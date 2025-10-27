
import os
import csv
import math
import heapq
import random
import argparse
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt

# ---------------------------
# Configuration (edit freely)
# ---------------------------
SIM_DURATION = 60            # total simulated minutes
RANDOM_SEED = 42             # reproducibility
N_TASKS = 20                 # number of tasks to generate
ARRIVAL_INTERVAL = (1, 4)    # minutes between task arrivals (uniform)
TASK_DURATION = (2, 8)       # minutes per task (uniform)
TASK_CPU_CORES = (1, 3)      # cores demanded by a task (uniform integer)

# Nodes/regions definitions
NODES = [
    # name, capacity cores, PUE, base_power kW, cpu_power_per_core kW
    ("eu-north-1", 8, 1.2, 0.4, 0.05),
    ("eu-west-1",  8, 1.3, 0.45, 0.055),
    ("us-east-1",  8, 1.4, 0.5, 0.06),
]

# Time-varying carbon intensity (gCO2/kWh) per region, piecewise constant.
# Tuples are (start_minute, g_per_kwh). The last entry applies until SIM_END.
CARBON_INTENSITY_SCHEDULE: Dict[str, List[Tuple[int, float]]] = {
    "eu-north-1": [(0, 90), (20, 70), (40, 120)],
    "eu-west-1":  [(0, 180), (30, 110)],
    "us-east-1":  [(0, 400), (10, 350), (30, 300), (50, 450)],
}

# Region latency (ms) for carbon+latency composite scheduling
REGION_LAT_MS = {
    "eu-north-1": 25,
    "eu-west-1": 35,
    "us-east-1": 95,
}
LAT_WEIGHT = 0.002  # gCO2 per millisecond (tune to study trade-off)


# ---------------------------
# Data Classes
# ---------------------------
@dataclass
class Task:
    id: int
    cpu_cores: int
    duration_min: int
    arrival_min: float


@dataclass
class TaskResult:
    task_id: int
    node: str
    start_min: float
    end_min: float
    cpu_cores: int
    duration_min: float
    queue_wait_min: float
    energy_kwh: float
    carbon_g: float


# ---------------------------
# Helpers
# ---------------------------
def piecewise_g_per_kwh(region: str, minute: float) -> float:
    sched = CARBON_INTENSITY_SCHEDULE[region]
    current = sched[0][1]
    for t, val in sched:
        if minute >= t:
            current = val
        else:
            break
    return current

def estimate_energy_kwh(pue: float, base_kw: float, cpu_kw_per_core: float,
                        cores: int, duration_min: float, node_util_before: float = 0.0, node_capacity: int = 8) -> float:
    """Non-linear server power: P(u) = P_idle + (P_peak - P_idle) * (alpha*u + (1-alpha)*u^2)."""
    duration_h = duration_min / 60.0
    u = min(1.0, node_util_before + (cores / node_capacity))
    alpha = 0.6
    f = alpha * u + (1 - alpha) * (u ** 2)
    idle_kw = base_kw
    peak_kw = base_kw + cpu_kw_per_core * node_capacity  # crude peak
    it_power_kw = idle_kw + (peak_kw - idle_kw) * f
    return it_power_kw * duration_h * pue


# ---------------------------
# Simulator (discrete event)
# ---------------------------
class Node:
    def __init__(self, name: str, capacity: int, pue: float, base_kw: float, cpu_kw_per_core: float):
        self.name = name
        self.capacity = capacity
        self.pue = pue
        self.base_kw = base_kw
        self.cpu_kw_per_core = cpu_kw_per_core
        self.in_use = 0  # cores currently in use

class Simulator:
    def __init__(self):
        self.now = 0.0
        self.event_queue: List[Tuple[float, int, str, dict]] = []
        self.event_id = 0

    def schedule(self, time_min: float, event_type: str, payload: dict):
        self.event_id += 1
        heapq.heappush(self.event_queue, (time_min, self.event_id, event_type, payload))

    def run(self, until: float):
        while self.event_queue and self.event_queue[0][0] <= until:
            t, _, etype, payload = heapq.heappop(self.event_queue)
            self.now = t
            yield etype, payload


class Policy(str, Enum):
    CARBON_GREEDY = "carbon_greedy"
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    CARBON_LATENCY_WEIGHTED = "carbon_latency_weighted"


class CarbonAwareScheduler:
    def __init__(self, nodes: List[Node], policy: Policy):
        self.nodes = nodes
        self.policy = policy
        self.results: List[TaskResult] = []
        self.waiting: List[Task] = []
        self._rr_idx = 0  # round robin index

    def _score_carbon(self, node: Node, task: Task, now: float) -> float:
        g = piecewise_g_per_kwh(node.name, now)
        energy = estimate_energy_kwh(node.pue, node.base_kw, node.cpu_kw_per_core,
                                     task.cpu_cores, task.duration_min,
                                     node_util_before=(node.in_use / node.capacity),
                                     node_capacity=node.capacity)
        return energy * g

    def _score_carbon_latency(self, node: Node, task: Task, now: float) -> float:
        carbon = self._score_carbon(node, task, now)
        lat = REGION_LAT_MS.get(node.name, 50)
        return carbon + LAT_WEIGHT * lat

    def _best_node(self, task: Task, now: float) -> Optional[Node]:
        candidates = [n for n in self.nodes if (n.capacity - n.in_use) >= task.cpu_cores]
        if not candidates:
            return None
        if self.policy == Policy.CARBON_GREEDY:
            return min(candidates, key=lambda n: self._score_carbon(n, task, now))
        if self.policy == Policy.CARBON_LATENCY_WEIGHTED:
            return min(candidates, key=lambda n: self._score_carbon_latency(n, task, now))
        if self.policy == Policy.ROUND_ROBIN:
            for _ in range(len(self.nodes)):
                idx = self._rr_idx % len(self.nodes)
                self._rr_idx += 1
                n = self.nodes[idx]
                if n in candidates:
                    return n
            return None
        if self.policy == Policy.RANDOM:
            return random.choice(candidates)
        return min(candidates, key=lambda n: self._score_carbon(n, task, now))

    def try_start_task(self, sim: Simulator, task: Task, max_wait: Optional[float] = None):
        node = self._best_node(task, sim.now)
        if node is None:
            if max_wait is not None and (sim.now - task.arrival_min) > max_wait:
                node = min(self.nodes, key=lambda n: self._score_carbon(n, task, sim.now))
            else:
                self.waiting.append(task)
                return
        start = sim.now
        node.in_use += task.cpu_cores
        finish = start + task.duration_min
        sim.schedule(finish, "task_finish", {"task": task, "node": node, "start": start})

    def on_task_finish(self, sim: Simulator, task: Task, node: Node, start: float):
        node.in_use -= task.cpu_cores
        g = piecewise_g_per_kwh(node.name, start)
        energy = estimate_energy_kwh(node.pue, node.base_kw, node.cpu_kw_per_core,
                                     task.cpu_cores, task.duration_min,
                                     node_util_before=(node.in_use / node.capacity),
                                     node_capacity=node.capacity)
        carbon = energy * g
        self.results.append(TaskResult(
            task_id=task.id, node=node.name,
            start_min=start, end_min=sim.now,
            cpu_cores=task.cpu_cores, duration_min=task.duration_min,
            queue_wait_min=max(0.0, start - task.arrival_min),
            energy_kwh=energy, carbon_g=carbon
        ))
        # TODO: try smarter ordering (e.g., sort by cores or deadline)
        if self.waiting:
            pending = self.waiting[:]
            self.waiting = []
            for w in pending:
                self.try_start_task(sim, w)


def generate_tasks(n: int, start_time: float, interval: Tuple[float, float],
                   duration: Tuple[int, int], cores: Tuple[int, int], seed: int = 42) -> List[Task]:
    random.seed(seed)
    tasks = []
    t = start_time
    for i in range(n):
        t += random.uniform(*interval)
        tasks.append(Task(
            id=i,
            cpu_cores=random.randint(*cores),
            duration_min=random.randint(*duration),
            arrival_min=t
        ))
    return tasks


def plot_carbon_intensity(sim_duration: int, schedule: Dict[str, List[Tuple[int, float]]], out_path: str):
    plt.figure()
    for region, sched in schedule.items():
        times = [t for t, _ in sched] + [sim_duration]
        vals = [v for _, v in sched] + [sched[-1][1]]
        plt.step(times, vals, where="post", label=region)
    plt.xlabel("Minute")
    plt.ylabel("gCO₂ / kWh")
    plt.title("Region Carbon Intensity (piecewise)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="carbon_greedy",
                        choices=[p.value for p in Policy],
                        help="Scheduling policy")
    parser.add_argument("--max-wait", type=float, default=None,
                        help="Optional max wait (minutes) before forcing placement")
    args = parser.parse_args()

    nodes = [Node(*cfg) for cfg in NODES]
    sched = CarbonAwareScheduler(nodes, policy=Policy(args.policy))
    sim = Simulator()

    tasks = generate_tasks(N_TASKS, start_time=0.0, interval=ARRIVAL_INTERVAL,
                           duration=TASK_DURATION, cores=TASK_CPU_CORES, seed=RANDOM_SEED)

    for task in tasks:
        sim.schedule(task.arrival_min, "task_arrival", {"task": task})

    for etype, payload in sim.run(until=SIM_DURATION):
        if etype == "task_arrival":
            t = payload["task"]
            sched.try_start_task(sim, t, max_wait=args.max_wait)
        elif etype == "task_finish":
            t = payload["task"]; node = payload["node"]; start = payload["start"]
            sched.on_task_finish(sim, t, node, start)

    art = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(art, exist_ok=True)

    # Summary CSV
    summary_csv = os.path.join(art, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id","node","start_min","end_min","cpu_cores","duration_min","queue_wait_min","energy_kwh","carbon_g"])
        for r in sched.results:
            w.writerow([r.task_id, r.node, round(r.start_min,2), round(r.end_min,2), r.cpu_cores,
                        r.duration_min, round(r.queue_wait_min,2), round(r.energy_kwh,6), round(r.carbon_g,3)])

    # Node totals
    totals: Dict[str, Dict[str, float]] = {}
    for r in sched.results:
        t = totals.setdefault(r.node, {"tasks":0, "energy_kwh":0.0, "carbon_g":0.0, "wait_sum":0.0})
        t["tasks"] += 1
        t["energy_kwh"] += r.energy_kwh
        t["carbon_g"] += r.carbon_g
        t["wait_sum"] += r.queue_wait_min

    totals_csv = os.path.join(art, "node_totals.csv")
    with open(totals_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node","tasks","energy_kwh","carbon_g","avg_wait_min"])
        for node, agg in totals.items():
            avg_wait = agg["wait_sum"] / max(1, agg["tasks"])
            w.writerow([node, agg["tasks"], round(agg["energy_kwh"],6), round(agg["carbon_g"],3), round(avg_wait,2)])

    # Plots
    plot_carbon_intensity(SIM_DURATION, CARBON_INTENSITY_SCHEDULE, os.path.join(art, "carbon_intensity.png"))

    # Bar charts
    nodes_list = list(totals.keys())
    energy_vals = [totals[n]["energy_kwh"] for n in nodes_list]
    carbon_vals = [totals[n]["carbon_g"] for n in nodes_list]

    plt.figure()
    plt.bar(nodes_list, energy_vals)
    plt.ylabel("kWh")
    plt.title("Energy per Node")
    plt.tight_layout()
    plt.savefig(os.path.join(art, "node_energy.png"))
    plt.close()

    plt.figure()
    plt.bar(nodes_list, carbon_vals)
    plt.ylabel("gCO₂")
    plt.title("Carbon per Node")
    plt.tight_layout()
    plt.savefig(os.path.join(art, "node_carbon.png"))
    plt.close()

    print("=== Run Summary ===")
    for node, agg in totals.items():
        avg_wait = agg["wait_sum"] / max(1, agg["tasks"])
        print(f"{node:>10s} | tasks={agg['tasks']:2d} | energy_kwh={agg['energy_kwh']:.4f} | carbon_g={agg['carbon_g']:.1f} | avg_wait={avg_wait:.2f} min")
    print(f"\nSaved artifacts to: {art}")


if __name__ == "__main__":
    main()
