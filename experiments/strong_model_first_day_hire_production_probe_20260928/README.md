# Strong Model first-day hire + production probe

This isolated experiment starts from the current Strong Model branch and does not edit its policy.

It uses the saved seed 92804001, seat 0 state at step 24 (day 1, hour 0, cash 463). It first checks that the current code reproduces the saved trace decision and that the trace action matches the following replay row. It then compares:

- the current chosen production plan and its existing commitments;
- that same plan with one `HIRE` order added to the same first action.

The script records one-turn official projection and the planner's central terminal cash forecast for both choices, plus when projected cash first rises above the boundary cash. A positive forecast difference is evidence only about this local model projection. It is not a Battle result or a strength claim.

Run with the same pinned official world used by the existing contract:

```sh
python -m pip install "git+https://github.com/Kaggle/kaggle-environments.git@d7729da06cc1382eb742d6980dc3180aa85caa28"
python experiments/strong_model_first_day_hire_production_probe_20260928/run_probe.py
```

The workflow artifact contains `result.json`. The probe intentionally fails if replay/trace alignment or current-policy reproduction fails.
