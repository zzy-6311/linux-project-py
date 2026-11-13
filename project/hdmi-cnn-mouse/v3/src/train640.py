"""重新训练 YOLO26n 640×640；原训练算法不变，只把归档输入和新输出分开。

复用 v2 的 FixedTrainer/FixedValidator 与原始划分清单，不重复打包训练图片。
只在显式调用 main() 时训练，不安装依赖，不下载权重，不覆盖历史 results。
"""
import os
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import json
from pathlib import Path
import sys
import time
from multiprocessing import freeze_support

import torch
import yaml
from ultralytics import YOLO

ARCHIVE = Path(__file__).resolve().parents[1]
V2 = ARCHIVE.parent / 'v2'
sys.path.insert(0, str(V2))
import compare_resolutions as comparison

# Windows DataLoader 的 spawn 子进程也会导入此模块，必须在模块级固定尺寸。
comparison.TARGET_HW = (640, 640)


def prepare_split(data_dir, destination):
    """将历史相对清单重新定位到原数据集，顺序与 640 原实验完全一致。"""
    data_dir = data_dir.resolve()
    groups = {}
    for split, expected in (('train', 607), ('val', 300)):
        source = V2 / 'results' / 'split' / f'{split}.txt'
        rows = [line.strip() for line in source.read_text(encoding='utf8').splitlines() if line.strip()]
        paths = [(data_dir / line).resolve() for line in rows]
        if len(paths) != expected or len(set(paths)) != expected:
            raise ValueError(f'{split} 划分必须保持 {expected} 张且无重复')
        for path in paths:
            if not path.is_relative_to(data_dir) or not path.is_file():
                raise FileNotFoundError(f'数据集图片缺失或越出根目录: {path}')
            label = path.parent.parent / 'labels' / (path.stem + '.txt')
            if not label.is_file():
                raise FileNotFoundError(f'标签缺失: {label}')
        groups[split] = paths
    if set(groups['train']) & set(groups['val']):
        raise ValueError('训练与验证图片重叠')
    destination.mkdir()
    for split, paths in groups.items():
        (destination / f'{split}.txt').write_text(''.join(p.as_posix() + '\n' for p in paths), encoding='utf8')
    data = destination / 'data.yaml'
    data.write_text(yaml.safe_dump(dict(path=str(data_dir),
                                      train=str(destination / 'train.txt'),
                                      val=str(destination / 'val.txt'),
                                      names={0: 'ball'}), sort_keys=False), encoding='utf8')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True, help='原 YOLO 数据集，包含 train/valid/test 图片与 labels')
    parser.add_argument('--output', type=Path, required=True, help='必须是尚不存在的新工作目录')
    parser.add_argument('--initial-weights', type=Path, default=V2 / 'weights/yolo26n.pt')
    args = parser.parse_args()
    output = args.output.resolve()
    # 允许 v3/work 下的新运行目录；拒绝将结果写进归档的输入/源码/历史证据。
    for protected in ('weights', 'results', 'data', 'src'):
        if output.is_relative_to(ARCHIVE / protected):
            raise ValueError(f'输出不能位于归档 {protected} 目录内')
    if output.is_relative_to(ARCHIVE.parent / 'v1') or output.is_relative_to(ARCHIVE.parent / 'v2'):
        raise ValueError('不能覆盖 v1/v2 归档')
    if output.exists():
        raise FileExistsError(f'为保留旧结果，请使用新的 --output: {output}')
    if not args.initial_weights.is_file():
        raise FileNotFoundError(args.initial_weights)
    if not torch.cuda.is_available():
        raise RuntimeError('本实验必须使用 CUDA，不退回 CPU。')
    output.mkdir(parents=True, exist_ok=False)
    data = prepare_split(args.data_dir, output / 'split')
    metadata = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                    data=str(data), initial_weights=str(args.initial_weights.resolve()), epochs=50,
                    shape_HW=[640, 640], batch=16, seed=42)
    record = output / 'experiment.json'
    record.write_text(json.dumps(metadata, indent=2), encoding='utf8')
    model = YOLO(str(args.initial_weights))
    started = time.perf_counter()
    model.train(trainer=comparison.FixedTrainer, data=str(data),
                project=str(output), name='640x640', exist_ok=False,
                imgsz=640, rect=True, epochs=50, patience=0, batch=16,
                device=0, workers=4, cache=False, amp=True, seed=42,
                deterministic=True, optimizer='AdamW', lr0=0.001, lrf=0.01,
                pretrained=True, save=True, plots=False, multi_scale=0.0,
                close_mosaic=0, mosaic=0.0, mixup=0.0, cutmix=0.0,
                degrees=0.0, translate=0.05, scale=0.2,
                fliplr=0.5, flipud=0.0, hsv_h=0.01, hsv_s=0.2, hsv_v=0.2)
    metadata['train_seconds'] = time.perf_counter() - started
    metadata['best'] = str(model.trainer.best)
    best = YOLO(str(model.trainer.best))
    metrics = best.val(validator=comparison.FixedValidator, data=str(data),
                       imgsz=640, rect=True, batch=16, device=0, workers=4,
                       quantize=32, project=str(output), name='640x640_val',
                       plots=False, verbose=False)
    metadata['metrics_fp32'] = {key: float(value) for key, value in metrics.results_dict.items()}
    record.write_text(json.dumps(metadata, indent=2), encoding='utf8')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    freeze_support()
    main()
