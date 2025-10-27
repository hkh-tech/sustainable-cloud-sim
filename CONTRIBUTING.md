# Contributing

Thanks for taking a look. This is intentionally small and readable.
If you change assumptions (power curve, carbon profiles, lat weights), please
explain the rationale in your commit message and README notes.

A few ideas:
- Add a memory/IO/NIC energy component, or a cooling term.
- Introduce data-locality and egress cost.
- Add mixed objectives and deadlines.
- Replace the greedy policy with a small look-ahead (even 1 step) to test deferral.

Code style: standard Python, short functions, comments over cleverness.
