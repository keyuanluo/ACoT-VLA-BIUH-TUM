## Script Layout

Shared entrypoints stay in this directory:

- `compute_norm_stats.py`
- `plot_training_loss.py`
- `serve_policy.py`
- `train.py`
- `train.sh`

Task-specific scripts are grouped into:

- `competition/` for AgiBot World Challenge / G2SIM workflow
- `libero/` for earlier LIBERO-only workflow

Compatibility symlinks are kept at the old top-level paths so existing commands still work.
