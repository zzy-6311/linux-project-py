#!/usr/bin/env python3
"""HDMI 输入/输出 + YOLO26 320 INT8 NPU 检测。运行在 RK3588，不需要桌面。"""
import argparse
import ctypes as C
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import threading
import time
import cv2  # 使用板子已有的无桌面 OpenCV，避免 pip GUI 版本依赖 X11
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # v1 根目录，不依赖启动时所在目录
from npu_model import NPUModel
from gst_support import Gst, GError, input_mode, input_format
from aim_control import AimControl


class GstMiniObject(C.Structure):
    # GStreamer 1.x 公共 ABI；来自 gstminiobject.h，避免猜测 PTS 字节偏移。
    _fields_ = [('type', C.c_size_t), ('refcount', C.c_int), ('lockstate', C.c_int),
                ('flags', C.c_uint), ('copy', C.c_void_p), ('dispose', C.c_void_p),
                ('free', C.c_void_p), ('priv_uint', C.c_uint), ('priv_pointer', C.c_void_p)]


class GstBuffer(C.Structure):
    _fields_ = [('mini_object', GstMiniObject), ('pool', C.c_void_p),
                ('pts', C.c_uint64), ('dts', C.c_uint64), ('duration', C.c_uint64),
                ('offset', C.c_uint64), ('offset_end', C.c_uint64)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=str(PROJECT_ROOT / 'weights' / 'ball320-int8.rknn'))
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--ai-rate', type=int, default=60, help='检测分支最高采集帧率')
    parser.add_argument('--duration', type=float, default=0, help='0 表示持续运行')
    parser.add_argument('--mouse-control', action='store_true', help='后退侧键切换 AI 开关，启动仍为 OFF')
    # 来自本机 Aimlabs 的 lookVertFov、lookIncrement 和 mouseSensitivity。
    # 改游戏灵敏度/FOV 后需同步参数，并重新验证实际命中。
    parser.add_argument('--vfov', type=float, default=70.5328004329168)
    parser.add_argument('--yaw', type=float, default=0.070, help='每个鼠标计数对应的水平角度')
    parser.add_argument('--pitch', type=float, default=0.070, help='每个鼠标计数对应的垂直角度')
    args = parser.parse_args()
    if not 0 < args.conf < 1 or not 1 <= args.ai_rate <= 60 or args.duration < 0:
        parser.error('参数范围错误')
    instance = open('/tmp/zzy-hdmi-pt.lock', 'w')
    fcntl.flock(instance, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cv2.setNumThreads(1)
    model = NPUModel(args.model)
    model.predict(np.zeros((180, 320, 3), np.uint8), args.conf)
    nodes = [p.parent.name for p in Path('/sys/class/video4linux').glob('video*/name')
             if p.read_text().strip() == 'stream_hdmirx']
    if len(nodes) != 1:
        raise RuntimeError('未找到唯一 HDMI 输入设备')
    device = '/dev/' + nodes[0]
    iw, ih, hz = input_mode(device)
    fmt = input_format(device)
    summary = Path('/sys/kernel/debug/dri/0/summary').read_text()
    mode = re.search(r'Connector:HDMI-A-1[\s\S]*?Display mode:\s*(\d+)x(\d+)p', summary)
    if not mode:
        raise RuntimeError('HDMI OUT 未启用')
    ow, oh = int(mode[1]), int(mode[2])
    scale = min(ow/iw, oh/ih)
    vw, vh = round(iw*scale), round(ih*scale)
    vx, vy = (ow-vw)//2, (oh-vh)//2
    scale = min(320/iw, 320/ih)
    sw, sh = max(4, round(iw*scale/4)*4), max(4, round(ih*scale/4)*4)
    gst = Gst()
    pull = gst.bind(gst.app, 'gst_app_sink_try_pull_sample', C.c_void_p, [C.c_void_p, C.c_uint64])
    get_buffer = gst.bind(gst.gst, 'gst_sample_get_buffer', C.c_void_p, [C.c_void_p])
    get_segment = gst.bind(gst.gst, 'gst_sample_get_segment', C.c_void_p, [C.c_void_p])
    to_running = gst.bind(gst.gst, 'gst_segment_to_running_time', C.c_uint64,
                          [C.c_void_p, C.c_int, C.c_uint64])
    running_now = gst.bind(gst.gst, 'gst_element_get_current_running_time', C.c_uint64, [C.c_void_p])
    extract = gst.bind(gst.gst, 'gst_buffer_extract', C.c_size_t,
                       [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t])
    fd = os.open('/dev/dri/card0', os.O_RDWR | os.O_CLOEXEC)
    pipeline = source = sink = bus = ai = worker = None
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    lock = threading.Lock()
    shared = {'count': 0, 'boxes': [], 'at': 0.0, 'ms': 0.0, 'npu_ms': 0.0, 'error': None,
              'stale_frames': 0, 'capture_age_ms': 0.0, 'fence_checks': 0, 'fence_gap_ms': None}
    control = None
    # 视频直接通过 DMA-BUF 交给显示控制器。CPU 仅处理缩小的检测画面。
    # 两条分支都只保留最新帧，推理慢时不会积压视频。
    description = (
        f'v4l2src device={device} io-mode=dmabuf ! '
        f'video/x-raw,format={fmt},width={iw},height={ih},framerate={round(hz*100)}/100 ! tee name=t '
        't. ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! '
        f'kmssink name=video fd={fd} connector-id=219 plane-id=59 '
        'fullscreen=true force-aspect-ratio=true sync=false sync-mode=none skip-vsync=true '
        'plane-properties="props,zpos=(uint)0,COLOR_ENCODING=(uint)1,COLOR_RANGE=(uint)0" '
        't. ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! '
        f'videorate drop-only=true max-rate={args.ai_rate} ! '
        f'videoscale n-threads=4 ! video/x-raw,format={fmt},width={sw},height={sh},pixel-aspect-ratio=1/1 ! '
        'videoconvert ! video/x-raw,format=BGR ! '
        'appsink name=ai max-buffers=1 drop=true sync=false wait-on-eos=false '
        'appsrc name=overlay is-live=true format=time do-timestamp=true block=false max-buffers=1 leaky-type=downstream '
        f'caps="video/x-raw,format=BGRA,width={ow},height={oh},framerate=15/1" ! '
        f'kmssink name=hud fd={fd} connector-id=219 plane-id=196 '
        'fullscreen=true sync=false sync-mode=none skip-vsync=true plane-properties="props,zpos=(uint)2"'
    )

    def infer():
        image = np.empty((sh, sw, 3), np.uint8)
        fence = 0
        pending_fence = False
        last_generation = None
        no_time = (1 << 64)-1
        try:
            while not stop.is_set():
                snapshot = control.state() if control else None
                if snapshot and snapshot['generation'] != last_generation:
                    last_generation = snapshot['generation']
                    fence = running_now(pipeline)
                    pending_fence = True
                sample = pull(ai, 200_000_000)
                if not sample:
                    continue
                at = time.monotonic()
                try:
                    buf = get_buffer(sample)
                    pts = GstBuffer.from_address(buf).pts
                    segment = get_segment(sample)
                    frame_time = to_running(segment, 3, pts) if segment and pts != no_time else no_time
                    now_time = running_now(pipeline)
                    if no_time in (frame_time, now_time, fence):
                        raise RuntimeError('采集时间戳不可用，已停止自动控制')
                    frame_age = (now_time-frame_time)/1e6
                    if frame_age < -100:
                        raise RuntimeError('采集时间戳与管线时钟不一致')
                    # 缩放/转换前积压的旧图像也必须丢掉；收到 sample 的时刻不代表采集时刻。
                    if frame_time < fence or (snapshot and snapshot['active'] and frame_age > 120):
                        with lock: shared['stale_frames'] += 1
                        continue
                    with lock:
                        shared['capture_age_ms'] = round(frame_age, 2)
                        if pending_fence:
                            shared['fence_checks'] += 1
                            shared['fence_gap_ms'] = round((frame_time-fence)/1e6, 2)
                            pending_fence = False
                    if extract(buf, 0, C.c_void_p(image.ctypes.data), image.nbytes) != image.nbytes:
                        raise RuntimeError('图像长度异常')
                finally:
                    gst.mini_unref(sample)
                boxes = model.predict(image, args.conf)
                if len(boxes):
                    boxes[:, [0, 2]] = boxes[:, [0, 2]] / sw * vw + vx
                    boxes[:, [1, 3]] = boxes[:, [1, 3]] / sh * vh + vy
                with lock:
                    shared.update(count=shared['count']+1, boxes=boxes.tolist(), at=at,
                                  ms=(time.monotonic()-at)*1000, npu_ms=model.last_ms)
                if control and control.act(boxes.tolist(), snapshot, stop):
                    # 点击释放、USB 完成及短暂等待之后立时间界线。
                    # 下一次推理只能使用这条界线之后采集的帧，不再复用旧结果。
                    fence = running_now(pipeline)
                    pending_fence = True
        except Exception as exc:
            with lock: shared['error'] = str(exc)

    try:
        if args.mouse_control:
            control = AimControl((vx, vy, vw, vh), vfov=args.vfov, yaw=args.yaw, pitch=args.pitch)
            print('AIM ready: AUTO OFF; press BTN_SIDE once to toggle. FOV/sensitivity require hit validation.', flush=True)
        print(json.dumps({'backend': 'RKNN 2.3.2 NPU INT8', 'runtime': model.version, 'input': [iw, ih, hz],
                          'output': [ow, oh], 'fit': [vx, vy, vw, vh]}), flush=True)
        error = C.POINTER(GError)()
        pipeline = gst.parse(description.encode(), C.byref(error))
        if error:
            message = error.contents.message.decode(errors='replace'); gst.error_free(error)
            raise RuntimeError(message)
        source, sink, ai = [gst.by_name(pipeline, n) for n in (b'overlay', b'video', b'ai')]
        bus = gst.bus(pipeline)
        if not all((pipeline, source, sink, ai, bus)):
            raise RuntimeError('GStreamer 元件创建失败')
        if gst.state(pipeline, 4) == 0:
            gst.check(bus)
            raise RuntimeError('HDMI 启动失败')
        worker = threading.Thread(target=infer, daemon=True)
        worker.start()
        start = last = last_video = time.monotonic()
        prev_video = prev_ai = change_video = 0
        video_fps = ai_fps = 0.0
        last_active = None
        base = np.zeros((oh, ow, 4), np.uint8)
        base[:, :, 3] = 255
        base[vy:vy+vh, vx:vx+vw, 3] = 0
        while not stop.is_set():
            now = time.monotonic()
            if args.duration and now-start >= args.duration: break
            gst.check(bus)
            stats = gst.stats(sink)
            count = stats.get('rendered', 0)
            if count != change_video: change_video, last_video = count, now
            if now-last_video > 10: raise RuntimeError('HDMI 视频停止超过 10 秒')
            with lock: result = dict(shared)
            mouse_state = control.state() if control else {'active': 0}
            if mouse_state['active'] != last_active:
                last_active = mouse_state['active']
                print(f'AIM switch: {"ON" if last_active else "OFF"}', flush=True)
            if result['error']: raise RuntimeError(result['error'])
            age = now-result['at'] if result['at'] else 999
            if now-start > 60 and age > 30: raise RuntimeError('模型没有持续产生结果')
            if now-last >= 1:
                video_fps = (count-prev_video)/(now-last)
                ai_fps = (result['count']-prev_ai)/(now-last)
                print(json.dumps({'video_fps': round(video_fps, 2), 'ai_fps': round(ai_fps, 2),
                                  'detect_ms': round(result['ms'], 2), 'npu_ms': round(result['npu_ms'], 2), 'result_age_ms': round(age*1000),
                                  'balls': len(result['boxes']), 'auto': mouse_state['active'],
                                  **{k: result[k] for k in ('capture_age_ms','stale_frames','fence_checks','fence_gap_ms')},
                                  'shots': control.shots if control else 0, **stats}), flush=True)
                last, prev_video, prev_ai = now, count, result['count']
            canvas = base.copy()
            if control and mouse_state['active'] and control.excluded_circle:
                cx, cy, radius = control.excluded_circle
                cv2.circle(canvas, (round(cx), round(cy)), round(radius), (0, 165, 255, 255), 1)
            if age < max(0.5, min(2, result['ms']/1000*2+0.2)):
                for x1, y1, x2, y2, confidence, cls in result['boxes']:
                    x1, y1, x2, y2 = map(round, (x1, y1, x2, y2))
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 255, 60, 255), 2)
                    cv2.drawMarker(canvas, ((x1+x2)//2, (y1+y2)//2), (0, 230, 255, 255), cv2.MARKER_CROSS, 14, 2)
                    cv2.putText(canvas, f'ball {confidence:.2f}', (x1, max(24, y1-7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 255, 60, 255), 2)
            cv2.rectangle(canvas, (12, 12), (min(ow-12, 1280), 172), (0, 0, 0, 255), -1)
            lines = [f'HDMI {iw}x{ih}@{hz:g} -> {ow}x{oh} | aspect preserved',
                     f'VIDEO: {video_fps:.1f} FPS | AI: {ai_fps:.1f} FPS | DETECT: {result["ms"]:.0f} ms',
                     f'YOLO26n 320 / NPU INT8 | balls: {len(result["boxes"])} | result age: {age*1000:.0f} ms'
                     if result['at'] else 'YOLO26n 320 / NPU INT8 | waiting for detection',
                     f'AUTO: {"ON" if mouse_state["active"] else "OFF"} | rear side button toggles | {control.last_action}'
                     if control else 'AUTO: disabled (preview only)']
            for i, line in enumerate(lines):
                cv2.putText(canvas, line, (28, 43+36*i), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (255, 255, 255, 255), 1)
            buf = gst.alloc(None, canvas.nbytes, None)
            if not buf: raise MemoryError('Overlay allocation failed')
            if gst.fill(buf, 0, C.c_void_p(canvas.ctypes.data), canvas.nbytes) != canvas.nbytes:
                gst.mini_unref(buf); raise RuntimeError('Overlay copy failed')
            if gst.push(source, buf) != 0: raise RuntimeError('Overlay push failed')
            stop.wait(max(0, 1/15-(time.monotonic()-now)))
    finally:
        stop.set()
        if control: control.cancel()
        if pipeline: gst.state(pipeline, 1)
        if worker: worker.join(10)
        if worker and worker.is_alive(): os._exit(1)
        if control: control.close()
        for ref in (bus, sink, source, ai, pipeline):
            if ref: gst.unref(ref)
        os.close(fd)
        instance.close()
        model.close()


if __name__ == '__main__':
    main()
