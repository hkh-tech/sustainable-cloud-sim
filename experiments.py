
"""
Simple experiment runner to compare policies.
Writes artifacts/experiment_results.csv.
"""
import csv, subprocess, sys, pathlib

POLICIES = ["carbon_greedy", "round_robin", "random", "carbon_latency_weighted"]

def run():
    base = pathlib.Path(__file__).parent.parent
    art = base / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    out = art / "experiment_results.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy","total_energy_kwh","total_carbon_g"])
        for pol in POLICIES:
            subprocess.run([sys.executable, str(pathlib.Path(__file__).parent/"sim.py"), "--policy", pol], check=False)
            totals = (art/"node_totals.csv").read_text().strip().splitlines()[1:]
            energy = sum(float(line.split(",")[2]) for line in totals if line.strip())
            carbon = sum(float(line.split(",")[3]) for line in totals if line.strip())
            w.writerow([pol, f"{energy:.6f}", f"{carbon:.3f}"])
    print(f"Wrote {out}")

if __name__ == "__main__":
    run()
