## Competition Scripts

These scripts are for the official AgiBot World Challenge / G2SIM workflow.

Main entrypoints:
- `server.sh`: serves a trained or smoke checkpoint through `serve_policy.py` for `G2SIM`.
- `openloop.py`: compares predicted actions against dataset actions for an already trained competition checkpoint.
- `download_reasoning2action_full.slurm`: downloads the full Reasoning2Action dataset.

Local ACoT smoke helpers:
- `compute_go2_acot_norm_stats.py`: computes local Go2 ACoT norm stats from extracted parquet data.
- `export_acot_init_checkpoint.py`: exports a step-0 competition ACoT checkpoint from an initialization source such as `pi05_base` or `pi05_libero`.

The shared policy server implementation still lives at `../serve_policy.py`.
