Suggested commit sequence (for clarity, not mandatory):

1) Initial commit: minimal discrete-event simulator and carbon-greedy policy
2) Docs: add README with design notes and usage
3) Feat: add round-robin and random baseline policies
4) Feat: carbon+latency weighted policy and --policy/--max-wait flags
5) Model: non-linear utilization-aware power curve
6) Chore: add .gitignore, LICENSE, CONTRIBUTING
7) Exp: add experiments.py and write experiment_results.csv
