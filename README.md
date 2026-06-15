# Chinese-Standard-Mahjong_try
try something

## v0.2.1

This release adds training metrics logging and plotting on top of the richer board-state representation.

Highlights:

- Added per-epoch `metrics.csv` output for SL training.
- Added a `matplotlib`-based `plot_metrics.py` helper to generate convergence curves from `metrics.csv`.
- Kept the richer board-state representation already introduced in the previous release.
- Continued support for NPU-aware supervised training with configurable `--device`, `--batch-size`, and OBS/local data paths.
- Kept RL training and inference aligned with the same enriched state format.

The result is a more traceable SL/RL pipeline with per-epoch metrics logging and plotting.

中文说明：

- 新增 SL 训练每个 epoch 的 `metrics.csv` 记录。
- 新增基于 `matplotlib` 的 `plot_metrics.py`，可从 `metrics.csv` 直接生成收敛曲线图。
- 保留上一版已经引入的更丰富局面状态表示。
- 监督训练继续支持 NPU，可通过 `--device`、`--batch-size` 和 `--data-dir` 灵活配置。
- 训练时会自动写出每个 epoch 的 `metrics.csv`，默认在 `model/metrics.csv`，也可用 `--metrics-file` 自定义路径。
- 安装依赖后可用 `python -m SL.plot_metrics --input model/metrics.csv --output model/metrics.png` 直接生成收敛曲线图。
- 可先执行 `pip install -r requirements.txt` 安装包括 `matplotlib` 在内的依赖。
- RL 侧继续和同一套新的状态格式保持一致。

这一版的目标是让 SL/RL 流程既有更完整的局面信息，也更方便回看训练过程和收敛趋势。
