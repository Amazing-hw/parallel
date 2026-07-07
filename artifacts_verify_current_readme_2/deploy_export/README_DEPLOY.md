# Parallel 部署交接包

这个目录是从训练/分析工程中导出的独立部署交接包，可以直接交给工程化同事。

## 文件说明

- `model.json`：新增并联 XGBoost 模型 JSON。
- `method.json`：部署方法配置，包含特征顺序、填充值、阈值、guard 模式、fusion/veto 参数。
- `selected_features.json`：训练时确认的最终特征列表。
- `fill_values.json`：每个特征的缺失值/异常值填充值。
- `feature_extractor.py`：部署侧参考特征提取脚本，来源于项目 `s02_features.py`。
- `commercial_model.py`：冻结商用模型脚本，来源于项目 `s01_model.py`。
- `commercial_model_manifest.json`：商用模型冻结证据，用于核对树参数和特征是否变化。
- `deploy_inference.py`：最小 Python 推理参考，用于说明模型加载、特征顺序和 veto 逻辑。
- `deploy_manifest.json`：导出清单和文件 SHA256。

## 工程化重点

1. 商用模型仍然保留，并联模型只提供独立风险复核信号。
2. 默认 `shadow` 不改变最终输出，只记录风险和分歧。
3. `fusion_config.json` 的核心内容已经写入 `method.json` 的 `parallel` 字段。
4. 真正上线前应由端侧工程按 `method.json` 重写为目标语言实现，并用本目录文件做一致性核对。
