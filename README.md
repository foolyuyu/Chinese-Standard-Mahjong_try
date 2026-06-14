# Chinese-Standard-Mahjong_try
try something

## v0.2

This release upgrades both SL and RL to a richer board-state representation.

Highlights:

- Expanded observation features to include fuller board context and global state.
- Updated SL data preprocessing, dataset loading, model input, and inference flow to match the new features.
- Added NPU-aware supervised training support with configurable `--device`, `--batch-size`, and OBS/local data paths.
- Kept RL training and inference aligned with the same enriched state format.

The result is a more expressive SL/RL pipeline built on the richer局面状态 version.

中文说明：

- SL 和 RL 都升级为更丰富的局面状态表示。
- SL 侧同步更新了数据预处理、数据集读取、模型输入和推理流程。
- 监督训练支持 NPU，可通过 `--device`、`--batch-size` 和 `--data-dir` 灵活配置。
- RL 侧也对齐了同一套新的状态格式，保证训练和推理一致。

这一版的目标是让 SL/RL 流程建立在更完整的局面信息之上。
