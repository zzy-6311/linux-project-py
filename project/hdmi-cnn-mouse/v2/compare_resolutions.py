"""四种固定分辨率：独立训练，再用原生 .pt 比较 GPU 延迟；不导出 ONNX。

train 子命令训练四组并测速；benchmark 子命令只对已有权重测速。
归档后将原 train.py 的数据分组函数合并进来，数值参数和测速方法保持不变。
数据集由 --data-dir 指定；新结果只写入新目录，保留 results 中的历史记录。
宽高顺序只用于名字，PyTorch/LetterBox 的参数顺序为 (高, 宽)。
"""
import os
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import csv
import gc
import json
import re
import time
from copy import copy
from datetime import datetime
from pathlib import Path
from multiprocessing import freeze_support

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'weights' / 'yolo26n.pt'
DATASET = ROOT / 'data'  # 运行时由 --data-dir 替换；归档不重复保存图片。
CLASSES = ['ball']
TRAIN_VIDEOS = {'1', '2'}
VAL_VIDEOS = {'3'}

SHAPES = {'320x320': (320, 320), '320x192': (192, 320),
          '256x256': (256, 256), '256x160': (160, 256)}
TARGET_HW = (320, 320)
EPOCHS = 50  # 四组相同训练预算，关闭早停；不是宣称训练到各自最优极限。
BATCH = 16


def prepare_data(output):
    """从原 train/valid/test 目录汇总图片，按来源视频重划分，不移动图片或标签。"""
    import yaml
    if TRAIN_VIDEOS & VAL_VIDEOS:
        raise ValueError('训练和验证不能使用同一段视频')
    groups = {'train': [], 'val': []}
    for folder in ('train', 'valid', 'test'):
        for image in sorted((DATASET / folder / 'images').glob('*')):
            if image.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                continue
            match = re.match(r'(\d+)_mp4-', image.name)
            if not match or match[1] not in TRAIN_VIDEOS | VAL_VIDEOS:
                raise ValueError(f'未知视频来源，请更新视频分组：{image.name}')
            split = 'train' if match[1] in TRAIN_VIDEOS else 'val'
            groups[split].append(image.resolve().as_posix())
    output.mkdir(parents=True, exist_ok=True)
    for split, images in groups.items():
        if not images:
            raise ValueError(f'{split} 没有图片，请确认 --data-dir')
        (output / f'{split}.txt').write_text('\n'.join(images) + '\n', encoding='utf-8')
    data_file = output / 'data.yaml'
    data_file.write_text(yaml.safe_dump({
        'path': str(DATASET),
        'train': str((output / 'train.txt').resolve()),
        'val': str((output / 'val.txt').resolve()),
        'names': dict(enumerate(CLASSES)),
    }, allow_unicode=True, sort_keys=False), encoding='utf-8')
    print(f'训练集 {len(groups["train"])} 张，验证集 {len(groups["val"])} 张；按视频分组。')
    return data_file


def resolve_reference(path, base):
    """兼容原实验的绝对路径与归档中的相对路径。"""
    path = Path(path)
    return path if path.is_absolute() else base / path


def fixed_shape(dataset):
    """沿用官方等比例缩放/标签变换，只固定每批最终 letterbox 的画布尺寸。"""
    assert dataset.rect
    dataset.batch_shapes[:] = TARGET_HW
    return dataset


class FixedValidator(DetectionValidator):
    def build_dataset(self, img_path, mode='val', batch=None):
        self.args.rect = True
        return fixed_shape(super().build_dataset(img_path, mode, batch))

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        assert tuple(batch['img'].shape[-2:]) == TARGET_HW, batch['img'].shape
        return batch


class FixedTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        return fixed_shape(super().build_dataset(img_path, mode, batch))

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        shape = tuple(batch['img'].shape[-2:])
        assert shape == TARGET_HW, shape
        if not getattr(self, '_shape_recorded', False):
            (self.save_dir / 'actual_shape.json').write_text(
                json.dumps({'train_BCHW': list(batch['img'].shape),
                            'val_batch_shapes': self.test_loader.dataset.batch_shapes.tolist()}),
                encoding='utf-8')
            print(f'VERIFIED TRAIN SHAPE: {list(batch["img"].shape)}', flush=True)
            self._shape_recorded = True
        return batch

    def get_validator(self):
        return FixedValidator(self.test_loader, save_dir=self.save_dir,
                              args=copy(self.args), _callbacks=self.callbacks)


def train_all(output):
    global TARGET_HW
    data = prepare_data(output / 'split')
    metadata = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                'epochs': EPOCHS, 'batch': BATCH, 'seed': 42, 'initial_weights': str(WEIGHTS),
                'shapes_HW': SHAPES, 'data': str(data), 'results': {}}
    (output / 'experiment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    for name, TARGET_HW in SHAPES.items():
        print(f'\nSTART TRAINING {name}', flush=True)
        model = YOLO(str(WEIGHTS))  # 每组重新加载相同初始权重，不接着上一组微调。
        started = time.perf_counter()
        model.train(trainer=FixedTrainer, data=str(data), project=str(output), name=name,
                    exist_ok=False, imgsz=max(TARGET_HW), rect=True,
                    epochs=EPOCHS, patience=0, batch=BATCH, device=0, workers=4,
                    cache=False, amp=True, seed=42, deterministic=True,
                    optimizer='AdamW', lr0=0.001, lrf=0.01, pretrained=True,
                    save=True, plots=True, multi_scale=0.0, close_mosaic=0,
                    mosaic=0.0, mixup=0.0, cutmix=0.0,
                    degrees=0.0, translate=0.05, scale=0.2, fliplr=0.5, flipud=0.0,
                    hsv_h=0.01, hsv_s=0.2, hsv_v=0.2)
        metadata['results'][name] = {'best': str(model.trainer.best),
                                      'train_seconds': time.perf_counter() - started}
        # 用独立验证器按完全相同的指定尺寸复核 best.pt，避免默认 val 额外补边。
        best = YOLO(str(model.trainer.best))
        metrics = best.val(validator=FixedValidator, data=str(data), imgsz=max(TARGET_HW),
                           rect=True, batch=16, device=0, workers=4, quantize=32,
                           project=str(output), name=name + '_val', plots=True, verbose=False)
        metadata['results'][name]['metrics_fp32'] = {k: float(v) for k, v in metrics.results_dict.items()}
        (output / 'experiment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        del model, best, metrics
        gc.collect()
        torch.cuda.empty_cache()
    return metadata


def describe(values):
    a = np.array(values)
    return {'mean_ms': float(a.mean()), 'median_ms': float(np.median(a)),
            'p95_ms': float(np.percentile(a, 95)), 'fps_reciprocal_mean': float(1000 / a.mean())}


@torch.inference_mode()
def benchmark(experiment, output):
    """batch=1；真实验证图片；预热；CUDA 同步；三轮交错测试降低顺序影响。"""
    meta = json.loads((experiment / 'experiment.json').read_text(encoding='utf-8'))
    paths = (experiment / 'split' / 'val.txt').read_text(encoding='utf-8').splitlines()
    paths = [str(resolve_reference(p, DATASET)) for p in paths]
    selected = [paths[i] for i in np.linspace(0, len(paths)-1, 12, dtype=int)]
    images = [cv2.imread(p) for p in selected]
    assert all(im is not None for im in images)
    torch.set_num_threads(4)
    # 推理不继承训练的确定性开关；固定尺寸允许 cuDNN 预热选择算法。
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    records = {}
    # 读取/解码图片不计入测速；include：letterbox、H2D、归一化、推理、后处理。
    for precision in (32, 16):
        models = {}
        tensors = {}
        for name, hw in SHAPES.items():
            model = YOLO(str(resolve_reference(meta['results'][name]['best'], experiment)))
            kw = dict(imgsz=hw, rect=False, device=0, quantize=precision,
                      conf=0.25, max_det=300, verbose=False, save=False)
            for i in range(25):
                model.predict(images[i % len(images)], **kw)
            assert tuple(model.predictor.preprocess([images[0]]).shape[-2:]) == hw
            models[name] = (model, kw)
            tensor = model.predictor.preprocess([images[0]])
            assert tensor.is_cuda and tensor.dtype == (torch.float16 if precision == 16 else torch.float32)
            print(f'BENCH INPUT {name}: {list(tensor.shape)} {tensor.device} {tensor.dtype}', flush=True)
            tensors[name] = tensor
            records[(name, precision)] = {'forward': [], 'pipeline': []}
        for round_id in range(3):
            order = list(SHAPES)
            np.random.default_rng(42 + round_id).shuffle(order)
            for name in order:
                model, kw = models[name]
                tensor = tensors[name]
                for _ in range(10):
                    model.predictor.model(tensor)
                torch.cuda.synchronize()
                for i in range(50):
                    start_event, stop_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    model.predictor.model(tensor)
                    stop_event.record()
                    stop_event.synchronize()
                    records[(name, precision)]['forward'].append(start_event.elapsed_time(stop_event))
                for i in range(50):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    model.predict(images[i % len(images)], **kw)
                    torch.cuda.synchronize()
                    records[(name, precision)]['pipeline'].append((time.perf_counter()-start)*1000)
                print(f'BENCH round={round_id+1} {name} FP{precision}', flush=True)
        del models, tensors, model, tensor
        gc.collect()
        torch.cuda.empty_cache()
    summaries = []
    for (name, precision), times in records.items():
        summary = {'resolution_WH': name, 'precision': precision, 'samples': len(times['forward']),
                   'forward_gpu': describe(times['forward']), 'pipeline_wall': describe(times['pipeline']),
                   'accuracy': meta['results'][name]['metrics_fp32']}
        summaries.append(summary)
    (output / 'benchmark.json').write_text(json.dumps({'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__, 'source_experiment': str(experiment), 'results': summaries,
        'method': 'batch=1, 25 warmups, 3x50 samples, native eager PT, fused Conv+BN; no ONNX/TensorRT/torch.compile; decoded 1080p input; FP32 accuracy; CPU threads=4; deterministic=False; cudnn.benchmark=True; output tensors remain on GPU; no decode/render/capture included',
        'raw_ms': {f'{k[0]}_fp{k[1]}': v for k,v in records.items()}}, indent=2), encoding='utf-8')
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['width_x_height','compute_precision','forward_mean_ms','forward_median_ms','forward_p95_ms',
                         'pipeline_mean_ms','pipeline_median_ms','pipeline_p95_ms','pipeline_fps',
                         'mAP50_fp32','mAP50_95_fp32','detection_precision_fp32','recall_fp32'])
        for s in summaries:
            a=s['accuracy']; g=s['forward_gpu']; p=s['pipeline_wall']
            writer.writerow([s['resolution_WH'],s['precision'],g['mean_ms'],g['median_ms'],g['p95_ms'],p['mean_ms'],p['median_ms'],
                             p['p95_ms'],p['fps_reciprocal_mean'],a['metrics/mAP50(B)'],a['metrics/mAP50-95(B)'],
                             a['metrics/precision(B)'],a['metrics/recall(B)']])
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    global DATASET
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'benchmark'], help='train=训练四组后测速；benchmark=只测速')
    parser.add_argument('--data-dir', type=Path, required=True, help='原始 data 目录，包含 train/valid/test')
    parser.add_argument('--experiment-dir', type=Path, default=ROOT / 'results',
                        help='benchmark 读取的实验目录，默认使用本归档')
    parser.add_argument('--output-dir', type=Path, help='新的输出目录，必须尚不存在')
    args = parser.parse_args()
    DATASET = args.data_dir.expanduser().resolve()
    if not DATASET.is_dir():
        parser.error(f'数据目录不存在：{DATASET}')
    if not torch.cuda.is_available():
        raise RuntimeError('必须使用 CUDA GPU，不能悄悄退回 CPU。')
    output = (args.output_dir or ROOT / 'runs' / (
        args.mode + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if args.mode == 'benchmark':
        experiment = args.experiment_dir.resolve()
        if not (experiment / 'experiment.json').is_file():
            parser.error(f'缺少实验记录：{experiment}')
        metadata = json.loads((experiment / 'experiment.json').read_text(encoding='utf-8'))
        for result in metadata['results'].values():
            weight = resolve_reference(result['best'], experiment)
            if not weight.is_file():
                raise FileNotFoundError(weight)
    else:
        if not WEIGHTS.is_file():
            raise FileNotFoundError(WEIGHTS)
        experiment = output
    output.mkdir(parents=True, exist_ok=False)
    print(f'NEW OUTPUT DIRECTORY: {output}', flush=True)
    if args.mode == 'train':
        train_all(output)
    benchmark(experiment, output)


if __name__ == '__main__':
    freeze_support()
    main()
