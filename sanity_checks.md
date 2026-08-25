# Sanity checks

The automated suite in `tests/test_sanity.py` checks capability, fitness, workload,
candidate, message, deterministic-environment, `rho=1`, zero-degradation,
unavailability, full-candidate, `B_candidate <= B_full`, and `Q <= 1` invariants.
The simulator additionally asserts at runtime that drift leaves estimates unchanged.
Decision code receives estimates only; true capability is accessed solely after a
winner is returned. Feedback is represented by an arrival-indexed pending queue.
Workload is stored as remaining load and changes only by assignment, event-time
progress, interruption, or completion.

Latest verification: **PASS**, 4 tests passed. Full-run audit: 3,150 seed-runs,
seeds 0--29, zero task-accounting violations, and maximum observed Q = 1.0.
