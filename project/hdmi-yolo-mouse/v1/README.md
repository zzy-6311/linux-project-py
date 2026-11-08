# v1：RK3588 HDMI + YOLO26n + 鼠标控制

在 RK3588 上接收电脑 HDMI 画面，使用 YOLO26n 的 RKNN INT8 模型检测小球，再通过 HDMI 输出画面、检测框和帧率。可选启用鼠标控制，通过自定义内核驱动向电脑发送相对移动和点击，用于 Aimlabs 小球训练场实验。

这是项目的**第一版板端方案**：固定 320×320 输入、RKNN INT8 推理、HDMI 预览与鼠标控制。2026-11-08 将原 `rk3588-vision-mouse` 目录整理到 `hdmi-yolo-mouse/v1`，并补入当时实际使用的模型权重。

本目录按用途分开：**`src` 放代码，`weights` 放模型，README 记录方案和用法**。不包含训练数据、驱动或 RKNN 动态库。今天重新训练的四种分辨率属于旁边的 [v2](../v2/README.md)，没有替换本版模型。

## 先看目录

```text
v1/
├── README.md                 # 第一版方案、环境和运行方法
├── src/                      # 在 RK3588 上运行的 Python 代码
│   ├── main.py               # 程序入口，把视频、识别和控制串起来
│   ├── npu_model.py          # YOLO 的 NPU 推理与检测框解码
│   ├── aim_control.py        # 选小球、计算鼠标位移并点击
│   ├── driver_control.py     # Python 与鼠标内核驱动交互
│   └── gst_support.py        # HDMI 视频采集和显示的底层接口
└── weights/                  # 实际使用的模型权重
    └── ball320-int8.rknn     # 320×320、YOLO26n、INT8
```

学习时先看 `src/main.py` 的整体流程，再看 `src/npu_model.py` 的识别过程和 `src/aim_control.py` 的鼠标控制。`driver_control.py` 是调用内核驱动的 Python 接口，不是驱动源码。

| 文件 | 作用 |
| --- | --- |
| `src/main.py` | 入口：HDMI 采集、显示、推理线程、HUD、自动控制调度 |
| `src/gst_support.py` | 使用 ctypes 调用 GStreamer C API，查询 HDMI 输入格式 |
| `src/npu_model.py` | 加载 RKNN Runtime、图片预处理、NPU 推理、检测框解码 |
| `src/aim_control.py` | 选择目标、计算相对位移、调用驱动、点击后排除上一目标区域 |
| `src/driver_control.py` | 查询、开启或关闭内核透传；也支持有限次数的位移注入 |
| `weights/ball320-int8.rknn` | 第一版实际使用的 320×320 INT8 模型，已经加入归档，可随 Git 上传 |

## 数据流程

```text
电脑 HDMI OUT → RK3588 HDMI IN
                        ├─ DMA-BUF 视频 → HDMI OUT → 显示器
                        └─ 缩小图片 → RGB 补边到 320×320 → NPU
                                                         ↓
                                              CPU 解码 → 检测框/HUD
                                                         ↓ 可选
                                              Python 选择目标
                                                         ↓ ioctl
鼠标接收器 → 内核转接驱动 → USB Device/OTG → 电脑
```

物理鼠标的接收与转发由内核完成。Python 不搬运物理鼠标事件，只提交自动控制指令。

## 已使用的环境

- 正点原子 ATK-DLRK3588，官方 Buildroot 系统，Linux 6.12.69。
- Python 3.14.5、NumPy 2.4、板载 OpenCV 4.13。
- GStreamer 1.28.2 及板厂提供的 `kmssink`，无需 Qt 桌面。
- RKNN Runtime 2.3.2，NPU 内核驱动 0.9.8。
- Python 直接使用 ctypes 调用 RKNN C API，不依赖 PyTorch、Ultralytics 或 RKNN Lite Python 包。
- 需要 `v4l2-ctl`、GStreamer 的 v4l2src/queue/tee/videorate/videoscale/videoconvert/appsink/appsrc/kmssink 插件，以及已挂载的 debugfs。

这是已验证环境的记录，不代表其他版本已经测试。程序在 Linux 板子上运行，不能直接作为 Windows PyCharm 本机程序运行。

## 部署前准备

将整个 `v1` 目录放到板子上，例如 `/root/hdmi-yolo-mouse/v1`，保留 `src`、`weights` 的层级。部署时再创建 `lib` 目录，放入与运行环境配套的 AArch64 `librknnrt.so`（本版使用 2.3.2）：

```text
hdmi-yolo-mouse/v1/
├── src/                       # 上面的五个 Python 文件
├── weights/
│   └── ball320-int8.rknn       # 已随本版归档
└── lib/                       # 仅部署时补充，不随本归档提供
    └── librknnrt.so           # AArch64 RKNN Runtime 2.3.2，Git 忽略
```

本项目使用的模型是固定 320×320、单类别、YOLO26 one-to-one 原始检测头，输出顺序为 `box8, cls8, box16, cls16, box32, cls32`，`reg_max=1`。不能直接替换成任意 YOLO RKNN 模型；输出不同时必须调整解码程序。

归档模型 SHA256：`ce17cfdf5d47239f564e5afa2705464501af6bdae547953c9dc2ba9b4400466f`。模型是从原 `linux-hdmi-preview/rknn/ball320-int8.rknn` 复制并校验的，没有重新转换。

输入为 RGB uint8 NHWC，等比例缩放并以 114 补边；模型转换时已配置 mean=0、std=255，程序不重复除以 255。输出通过 `want_float=1` 反量化，并根据实际布局转换，解码后恢复图片坐标。本模型使用 one-to-one 分支，不执行 NMS。

打开鼠标控制还需要提前部署并启用兼容的 `zzy_mouse_bridge` 驱动，存在 `/dev/zzy_mouse_bridge`，且 USB 输出已在电脑端枚举。普通 HID gadget 或最早的只读驱动版本不兼容。需要驱动支持统计 ABI 1（128 字节）、AI 状态 ABI 1（32 字节），以及会话、相对报告、取消等 ioctl。驱动不随本目录提供。

用 root 运行；启动前停止占用 HDMI 输入或 DRM 显示平面的旧预览/桌面程序，不要同时启动两份识别程序。

## 运行

进入板子上的项目目录，先只测试视频与检测：

```bash
cd /root/hdmi-yolo-mouse/v1
python3 -u src/main.py
```

显示等比例视频、绿色检测框、VIDEO/AI 帧率与检测耗时。单独预览不需要鼠标驱动。

启用鼠标控制前查询并开启透传：

```bash
python3 src/driver_control.py status
python3 src/driver_control.py start
python3 -u src/main.py --mouse-control
```

- 启动后仍为 `AUTO OFF`。
- 后退侧键（BTN_SIDE）按一次开启，再按一次关闭。
- 选择画面中心附近、置信度足够且不在排除圆内的小球。
- 相对移动后点击；点击完成后丢弃旧帧，只识别时间戳晚于动作结束界线的新画面。
- 橙色圆表示上一球在转动视角后的预计位置附近，半径按检测框外扩 25%，最小 12 个显示像素。它是几何近似，不是目标跟踪 ID；不同球进入圆内也会被排除。
- 手动鼠标动作或开关切换改变驱动 generation，使旧检测结果失效，并清除旧排除圆。
- Ctrl+C 结束程序并取消自动控制；物理鼠标透传由驱动独立管理。

其他运行示例：

```bash
# 限时预览 30 秒
python3 -u src/main.py --duration 30

# 指定兼容模型与检测分支最高帧率
python3 -u src/main.py --model ./weights/ball320-int8.rknn --ai-rate 60 --conf 0.25

# 修改游戏投影和灵敏度对应参数
python3 -u src/main.py --mouse-control --vfov 70.5328004329168 --yaw 0.070 --pitch 0.070
```

## 参数与硬件适配位置

`src/main.py` 的 `--conf` 是检测框显示/解码阈值，默认 0.25；`src/aim_control.py` 的 `confidence` 是自动点击筛选阈值，默认 0.65，两者作用不同。

当前鼠标参数位于 `AimControl.__init__`：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `move_settle` | 0.030 秒 | 移动完成后，给游戏处理输入的时间 |
| `click_hold` | 0.010 秒 | 左键按下保持时间 |
| `post_click` | 0.040 秒 | 点击后等待，再建立新帧时间界线 |
| `yaw` / `pitch` | 0.070 | 每个鼠标计数对应的水平/垂直角度 |
| `vfov` | 70.5328004329168 度 | 游戏垂直视场角 |

游戏灵敏度、FOV 或控制方式改变后，需要重新校准。USB 报告完成不等于游戏画面已经更新，不能认为等待越短越好。

目前针对已有板子保留以下硬件配置，换系统或接口时应先核对再修改 `src/main.py`：

- DRM 设备 `/dev/dri/card0`，显示状态 `/sys/kernel/debug/dri/0/summary`。
- 输出连接器 `HDMI-A-1`，connector-id=219。
- 视频 plane-id=59，叠加层 plane-id=196。
- `sync-mode`、`skip-vsync`、plane-properties 等依赖当前板厂 kmssink 的支持。
- 输入设备按 sysfs 名称 `stream_hdmirx` 自动寻找，不固定 video 编号。
- 保留 `/tmp/zzy-hdmi-pt.lock` 锁文件名称，与旧版本互斥。

## 状态与常见问题

`src/driver_control.py status` 中 `connected/configured/enabled` 分别表示物理鼠标已连接、USB 输出已配置、透传已开启。

USB 线重新连接或电脑重置 USB 后，驱动可能关闭透传。曾出现接收器在线、configured=1，但 enabled=0，导致鼠标不动。确认线缆与连接正常后执行 `python3 src/driver_control.py start` 恢复；这不会自动开启 AI。

`AUTO ON` 时新帧会经过时间戳检查，120ms 以上的旧图像被丢弃。日志中的 `stale_frames` 是过滤旧帧的累计数，不等于 USB 丢包；`fence_gap_ms` 表示动作时间界线之后被接受的帧时间差。

识别成功但点不准时，分别检查灵敏度/FOV、鼠标等待、场景变化和旧画面问题，不能只看模型是否画出了框。计数 `shots` 表示已发送点击，不代表游戏确认命中。

## 实测范围

在现有板子上，4K 60Hz 输入、1080p 60Hz 输出、关闭自动控制的约 74 秒采样中，视频约 60 FPS，检测约 60 FPS，NPU 调用约 8.62ms，检测处理约 11.97ms。这些不是完整端到端延迟，也不是每秒命中数。

300 张独立验证图上的 INT8 模型 AP50 约 98.93%，AP50–95 约 85.80%。这些是模型检测指标，不代表游戏命中率。自动控制的速度与准确率仍受输入延迟和参数影响。

以上性能来自第一版原程序的板端测试。2026-11-08 整理归档后，代码改放 `src`、模型改放 `weights`，同步修改了模型和动态库的定位路径；视频、推理和鼠标控制逻辑没有改动。整理后没有重新上板，也没有把电脑 v2 的测速结果当作板端性能。
