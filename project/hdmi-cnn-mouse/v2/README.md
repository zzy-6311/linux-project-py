# v2：四种输入分辨率的训练、测速与 RKNN 转换记录

实验日期：2026-11-08。目标是给第一版寻找更轻的检测输入，先在电脑上比较，再准备 RK3588 模型。**本版是实验归档，尚未替代 v1 的板端运行程序。**

四组都使用 YOLO26n：320×320、320×192、256×256、256×160。本文尺寸均为“宽×高”，PyTorch 张量最后两维则是“高、宽”。没有改成 RT-DETR，没有做模型剪枝。

## 文件保留了什么

| 位置 | 内容 |
| --- | --- |
| `compare_resolutions.py` | 四组独立训练、固定矩形输入、验证、原生 PT GPU 测速；两个子命令分别负责训练与只测速 |
| `weights/yolo26n.pt` | 四组共同的初始预训练权重 |
| `weights/320x320.pt` 等四个文件 | 各组原来的 `best.pt`，按输入尺寸命名 |
| `results/experiment.json` | 环境、训练用时、完整验证指标、权重路径与 SHA256 |
| `results/comparison.csv` | 第二轮测速的汇总表，适合直接查看或画图 |
| `results/benchmark.json` | 第二轮详细测速，包括每次采样值、均值、中位数、P95 |
| `results/benchmark_after_training.json` | 训练后第一轮测速，保留以便看重复测量的波动 |
| `results/四种尺寸/` | 每组的 50 轮 `results.csv`、原始 `args.yaml`、实际输入形状 `actual_shape.json` |
| `results/split/` | 当时的训练和验证图片清单，路径相对于原始数据集根目录 |
| `rknn/` | 320×192 转换代码、FP16/INT8 RKNN 权重、转换与模拟器对照记录 |

不重复保存训练图片、原视频、`last.pt`、训练截图、缓存、控制台日志、ONNX 中间文件或 RKNN 校准图片。原来的数据和实验目录没有删除。`args.yaml` 中的绝对路径是历史运行记录；复现请使用下面的入口，不直接拿这份 YAML 启动。

## 实验怎么做的

电脑使用 NVIDIA GeForce RTX 4060 Laptop GPU；PyTorch `2.9.1+cu128`，Ultralytics `8.4.204`。

原始图片为 1920×1080，单类别 `ball=0`。汇总原 `train/valid/test` 下的图片后，按文件名中的来源视频重新分组：视频 1、2 共 **607 张训练图**；视频 3 共 **300 张验证图**。同一视频不同时进入训练和验证。验证集用于挑选和比较模型，不是另外保留的最终测试集。

四组每次重新加载同一份 `yolo26n.pt`，各训练 50 轮、batch=16、seed=42、关闭早停、启用 AMP。优化器 AdamW，初始学习率 0.001、末期倍率 0.01。关闭 mosaic/mixup/cutmix，保留轻量平移、缩放、水平翻转和颜色增强。详细参数在代码和每组 `args.yaml` 中。

不同尺寸通过 `FixedTrainer` / `FixedValidator` 固定 `batch_shapes`，并在预处理之后断言张量大小。官方的数据与标签变换仍然保留，图片等比例缩放再补边，不强行拉伸。

例如 1920×1080 缩到宽 320 后，内容约为 320×180：320×320 需要较多上下补边，320×192 则主要是减少空白补边。后者的总输入像素减少 40%，但这不等于推理时间必然减少 40%。`args.yaml` 中的 `imgsz: 320` 也不代表实际还是正方形，实际形状以 `actual_shape.json` 为准。

## 训练结果

以下为各组最佳权重的 **FP32 验证指标**，均使用各自的固定输入尺寸；不是游戏命中率。

| 输入宽×高 | Precision | Recall | mAP50 | mAP50–95 | 训练用时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 320×320 | 99.88% | 98.22% | 99.35% | 88.74% | 531.7 秒 |
| 320×192 | 99.86% | 98.40% | 99.34% | 86.90% | 494.9 秒 |
| 256×256 | 99.88% | 98.59% | 99.44% | 85.84% | 500.4 秒 |
| 256×160 | 99.74% | 98.18% | 98.50% | 85.37% | 487.8 秒 |

四组都能检测小球。320×192 的 mAP50 接近 320×320，但更严格的定位指标 mAP50–95 下降约 1.84 个百分点。256×160 进一步减少输入，mAP50 也有下降。因此本轮选择 **320×192 作为下一步上板候选**，并不表示它已经被证明是所有硬件上的最优选择。

## 原生 PT 测速

没有转 ONNX/TensorRT，没有 `torch.compile`。使用原生 PyTorch eager 路径，推理时融合 Conv+BN，batch=1、CPU 线程数 4。每种形状和精度先预热 25 次，再进行三轮交错顺序测试，每轮各计时 50 次，共 150 次。

- **forward**：预处理后的输入已经在 GPU 上，用 CUDA Event 测模型前向。
- **pipeline**：从已经解码的内存图片调用 `model.predict()`，用 CPU 时钟和 CUDA 同步计时，包含补边、上传 GPU、归一化、模型与后处理。
- 不包含磁盘读图、视频解码、HDMI 采集、绘制、显示或鼠标动作。结果张量仍在 GPU，不额外复制到 CPU。
- 表中 FPS 为 `1000 / pipeline 平均毫秒`，是单张串行处理速度的换算值，不是显示器刷新率，也不是 RK3588 的速度。
- FP16 只做了测速；下表不代表已经测过 FP16 的完整 mAP。

第二轮记录（`benchmark.json` / `comparison.csv`）：

| 输入 | FP32 forward | FP32 pipeline | FP32 FPS | FP16 forward | FP16 pipeline | FP16 FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 320×320 | 17.34 ms | 22.70 ms | 44.05 | 20.27 ms | 24.18 ms | 41.36 |
| 320×192 | 15.80 ms | 19.65 ms | 50.89 | 18.82 ms | 23.12 ms | 43.26 |
| 256×256 | 15.09 ms | 19.63 ms | 50.94 | 19.59 ms | 24.33 ms | 41.11 |
| 256×160 | 16.72 ms | 20.02 ms | 49.95 | 18.95 ms | 23.74 ms | 42.12 |

训练结束后的第一轮记录（`benchmark_after_training.json`）：

| 输入 | FP32 pipeline / FPS | FP16 pipeline / FPS |
| --- | ---: | ---: |
| 320×320 | 19.28 ms / 51.88 | 21.03 ms / 47.55 |
| 320×192 | 19.47 ms / 51.36 | 20.98 ms / 47.67 |
| 256×256 | 19.26 ms / 51.92 | 20.92 ms / 47.81 |
| 256×160 | 20.28 ms / 49.30 | 22.73 ms / 43.99 |

两轮的排名和耗时有波动，所以不能只拿第二轮就断言矩形输入一定快多少。在这台电脑、这条原生 PT 路径下，缩小图片没有带来等比例加速，FP16 也没有比 FP32 更快。小模型的调度、算子执行和前后处理开销可能影响结果；没有据此推算 NPU 的实测速度。

## 320×192 的 RKNN 准备

已从本目录的 `weights/320x192.pt` 生成两个 RK3588 模型。RKNN Toolkit2 版本为 2.3.2，ONNX 中间导出 opset=12。转换保留 YOLO26 的 one-to-one 原始检测头，六个输出分别为三个 stride（8、16、32）下的回归和单类分类分支；`reg_max=1`，解码不做 NMS。

输入固定为 RGB uint8 NHWC `[1,192,320,3]`，等比例缩放、114 补边。归一化在 RKNN 中按 mean=0、std=255 处理，调用端不要再除以 255。INT8 从训练集按 seed=42 抽取 160 张校准图，使用 asymmetric_quantized-8、normal 算法、optimization_level=3。

导出时核对了原 PT、原始头解码与 ONNX 结果，然后用 PC 模拟器对照 **12 张验证图中的 36 个 PT 检测框**，置信度阈值 0.25、匹配 IoU≥0.5：

| 模型 | 相对 PT 少框 | 相对 PT 多框 | 已匹配框平均 IoU | 最大中心偏差（模型输入像素） |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 0 | 0 | 0.99946 | 0.00428 |
| INT8 | 2 | 0 | 0.97033 | 0.19442 |

INT8 少的两个框中，一个没有对应标注，另一个是真实小球：PT 得分约 0.261，量化后降到约 0.151，低于 0.25 阈值。因此不能说 INT8 完全不掉精度；仅匹配框的高 IoU 也不能代替整体精度。

这只是小样本一致性检查，**不是完整验证集 mAP，也不是上板性能测试**。详细数值与权重 SHA256 在 `rknn/conversion.json`。v1 的 `npu_model.py` 写死 320×320 网格，不能只替换模型文件就运行本版；后续还需要适配宽高、网格及坐标还原，再上板核对识别和速度。

## 以后怎么复现

本次整理没有重新训练、测速或转换，只保留了已有结果。运行依赖原实验环境的 PyTorch、Ultralytics、OpenCV、NumPy、PyYAML。归档代码合并了原 `train.py` 的数据分组函数，调整了文件路径与输出位置，没有改变四组训练参数、固定形状逻辑及测速方法。

在本机 PowerShell 中进入本 `v2` 目录，用已有虚拟环境执行：

```powershell
# 只对归档的四份权重重新测速，新结果保存到 v2/runs 下
& 'G:\python-project\linux-project\.venv\Scripts\python.exe' .\compare_resolutions.py benchmark --data-dir 'G:\python-project\linux-project\project\linux-hdmi-preview\yolo\data'

# 从同一份初始权重重新训练四组，然后测速；这一步会实际开始训练
& 'G:\python-project\linux-project\.venv\Scripts\python.exe' .\compare_resolutions.py train --data-dir 'G:\python-project\linux-project\project\linux-hdmi-preview\yolo\data'
```

换电脑后改 Python 路径和 `--data-dir` 即可；数据目录仍需有原始 `train/valid/test/images` 与对应 `labels`。不提供数据集时，可以阅读记录和使用权重，但无法重跑训练与同一组图片的测速。脚本不允许把新输出写到已经存在的目录，不会覆盖本归档的 `results`。

如果要重新转换 320×192，先在 Windows 导出中间文件，再在装有 RKNN Toolkit2 2.3.2 的 WSL 环境构建。下面两条命令分别在对应系统中、当前 `v2` 目录执行，指向同一个新建工作目录：

```powershell
# Windows：需要 onnx 和 onnxruntime；会生成 ONNX、校准图和 PT 参考输出
& 'G:\python-project\linux-project\.venv\Scripts\python.exe' .\rknn\convert.py export --data-dir 'G:\python-project\linux-project\project\linux-hdmi-preview\yolo\data' --output-dir .\rknn\work\rebuild
```

```bash
# WSL：先激活装有 Toolkit2 2.3.2 的环境，再进入本 v2 目录
python rknn/convert.py build --output-dir rknn/work/rebuild
```

新的模型与转换记录位于工作目录内，不覆盖 `rknn/` 中今天已经生成的两份模型。本归档不携带 ONNX 和校准图，因此 build 前必须先 export。两份已归档 RKNN 可直接留待板端适配，不需要现在重做转换。
