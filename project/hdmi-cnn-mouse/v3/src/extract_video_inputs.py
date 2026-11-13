"""从用户已有视频抽取原尺寸 BGR 帧，供同图多种 NPU 调度/IO 方式对比。"""
import argparse
from pathlib import Path
import hashlib
import json
import time

import cv2
import numpy as np


SOURCE = None
DEST = None


def main():
    global SOURCE,DEST
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    SOURCE=args.video_dir
    DEST=args.output_dir
    DEST.mkdir(parents=True,exist_ok=True)
    # 三个源视频均为 1920×1080；24 张原始 BGR 图约 142 MiB。
    # 不改变输入尺寸/颜色，不在这里做模型前处理。
    plans = [('1.mp4', 4), ('2.mp4', 4), ('3.mp4', 16)]
    output = DEST / 'video_frames.npz'
    metadata_path = DEST / 'video_frames.json'
    if output.exists() or metadata_path.exists():
        raise FileExistsError('保留已有样本与记录，请先检查现有文件。')
    sources, samples, frames = [], [], []
    started = time.perf_counter()
    expected_shape = None
    for filename, count in plans:
        path = SOURCE / filename
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f'无法打开 {path}')
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        sources.append(dict(path=str(path), bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            frame_count=total, fps=fps, duration_seconds=total/fps,
                            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            sample_count=count, split='validation' if filename == '3.mp4' else 'training'))
        try:
            indices = np.round(np.linspace((total-1)*.08, (total-1)*.92, count)).astype(int)
            for frame_index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f'读取失败 {filename} 帧 {frame_index}')
                frame = np.ascontiguousarray(frame)
                if expected_shape is None:
                    expected_shape = frame.shape
                if frame.shape != expected_shape:
                    raise RuntimeError(f'输入帧尺寸不一致：{frame.shape} / {expected_shape}')
                samples.append(dict(sample_id=len(samples), source=filename,
                                    frame_index=int(frame_index),
                                    nominal_time_seconds=float(frame_index/fps),
                                    raw_bgr_sha256=hashlib.sha256(frame).hexdigest()))
                frames.append(frame)
        finally:
            cap.release()
    all_frames = np.stack(frames)
    if all_frames.nbytes > 200_000_000:
        raise RuntimeError('原始输入缓存超过 200 MB，停止保存以控制内存。')
    np.savez_compressed(output, frames=all_frames)
    if output.stat().st_size > 100_000_000:
        output.unlink()
        raise RuntimeError('压缩样本超过 100 MB，已撤销本次新输出。')
    with np.load(output, allow_pickle=False) as data:
        verified = data['frames']
        if not np.array_equal(all_frames, verified):
            raise RuntimeError('样本无损回读校验失败')
    metadata = dict(
        purpose='同一批真实视频帧，比较 RKNN 核调度与内存 IO；非视频播放 FPS 测试。',
        timing_boundary='加载并解压 NPZ 后，从已解码 RAM BGR 图像开始计时；不含 eMMC 读取、MP4 解码或 HDMI。',
        format='uint8 BGR NHWC，原视频尺寸，无缩放、补边或颜色转换',
        sources=sources, samples=samples, shape=list(all_frames.shape),
        raw_bytes=int(all_frames.nbytes), npz_bytes=output.stat().st_size,
        npz_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        roundtrip_exact=True, preparation_seconds=time.perf_counter()-started)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: metadata[k] for k in ('shape', 'raw_bytes', 'npz_bytes', 'npz_sha256', 'roundtrip_exact')}, indent=2))
    print(output)
    print(metadata_path)


if __name__ == '__main__':
    main()
