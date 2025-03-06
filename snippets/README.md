# 脱敏代码片段

本目录只放脱敏后的核心流程片段，用于展示工程设计思路。

这些文件不是完整业务源码，不能直接作为完整系统运行。

| 文件 | 展示内容 |
| --- | --- |
| `map-click-identify.js` | 地图点击识别、坐标校准、POI 匹配、AI 识别聚合、线索评分 |
| `ai-inference-service.py` | AI 推理服务契约、输入校验、安全边界、模型路由、结果归一化 |
| `pv-forecast-transformer.py` | 发电量预测特征工程、模型路由、公式降级、收益输出、误差评估 |

## 片段设计重点

- `map-click-identify.js` 展示地图拓客的主链路：坐标标准化、逆地理编码、周边 POI、卫星影像识别、线索评分和人工复核判断。
- `ai-inference-service.py` 展示 AI 推理服务如何对外提供统一接口，并把图片输入、模型路由、检测框后处理和屋顶业务状态统一成稳定 API。
- `pv-forecast-transformer.py` 展示发电量预测如何接收天气、容量和时间特征，并通过 Transformer / XGBoost / LightGBM / 公式模型输出发电量、收益和置信度。
