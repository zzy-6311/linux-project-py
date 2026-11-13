"""Export freshly trained YOLO26n 640 and validate PT -> raw head -> ONNX.

Runs only when called explicitly, on CPU, without downloads or board access.
Archive edition: uses v3/data video frames and the v2 split; requires --data-dir.
All new files go to --output. Adds 12 held-out images to the 24 video frames.
Calibration uses 160 training images only, with identical 114-padded geometry.
"""
import os
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import types

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO

ARCHIVE = Path(__file__).resolve().parents[1]
SPLIT = ARCHIVE.parent / 'v2/results/split'
DEFAULT_PT = ARCHIVE / 'weights/yolo26n-640x640/best.pt'
VIDEO_ROOT = ARCHIVE / 'data'
ROOT = None  # 运行时指定新的 --output，不覆盖归档模型或结果。
DATA_DIR = None
from yolo_helpers import preprocess, decode_yolo, restore

SIZE = 640
CONFIDENCE = .25
OUTPUT_NAMES = ['box8', 'cls8', 'box16', 'cls16', 'box32', 'cls32']
EXPECTED = [(1, c, SIZE // stride, SIZE // stride)
            for stride in (8, 16, 32) for c in (4, 1)]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')
    temporary.replace(path)


def split_paths(split):
    # v2 保留了原清单的顺序，改路径不会改变校准抽样/验证抽样。
    rows = [line.strip() for line in (SPLIT / f'{split}.txt').read_text(encoding='utf8').splitlines() if line.strip()]
    paths = [(DATA_DIR / line).resolve() for line in rows]
    expected = {'train': 607, 'val': 300}[split]
    if len(paths) != expected or len(set(paths)) != expected:
        raise ValueError(f'Invalid archived {split} split: expected {expected}')
    for path in paths:
        if not path.is_relative_to(DATA_DIR) or not path.is_file():
            raise FileNotFoundError(path)
    return paths


def raw_forward(head, features):
    # Keep the trained one-to-one branch. Distances are already LTRB (reg_max=1).
    result = []
    for i in range(head.nl):
        result.extend((head.one2one_cv2[i](features[i]), head.one2one_cv3[i](features[i])))
    return tuple(result)


def as_six_columns(boxes):
    return np.column_stack((boxes, np.zeros(len(boxes), np.float32))).astype(np.float32)


def check_detection_sets(reference, actual, label, tolerance=.025):
    """Match by confidence+geometry, avoiding a false failure on score ties."""
    if reference.shape != actual.shape:
        raise AssertionError(f'{label}: different shapes {reference.shape} vs {actual.shape}')
    if not len(reference):
        return dict(count=0, max_box_abs_error=0., max_score_abs_error=0.)
    unused = set(range(len(actual)))
    box_error = score_error = 0.
    for ref in reference:
        candidates = np.asarray(sorted(unused), dtype=np.int64)
        cost = np.max(np.abs(actual[candidates, :4] - ref[:4]), axis=1)
        best = int(candidates[int(np.argmin(cost))])
        row = actual[best]
        box_error = max(box_error, float(np.max(np.abs(row[:4] - ref[:4]))))
        score_error = max(score_error, float(abs(row[4] - ref[4])))
        if box_error > tolerance or abs(row[4] - ref[4]) > 1e-3 or row[5] != ref[5]:
            raise AssertionError(f'{label}: detection mismatch {ref.tolist()} vs {row.tolist()}')
        unused.remove(best)
    return dict(count=len(reference), max_box_abs_error=box_error, max_score_abs_error=score_error)


def prepare_calibration():
    paths = split_paths('train')
    val_paths = set(split_paths('val'))
    assert len(paths) >= 160 and not (set(paths) & val_paths)
    selected = np.random.default_rng(42).choice(len(paths), 160, replace=False)
    directory = ROOT / 'calibration640'
    directory.mkdir(exist_ok=True)
    records = []
    for i, position in enumerate(selected):
        source = paths[int(position)]
        image = cv2.imread(str(source))
        assert image is not None, source
        rgb, geometry = preprocess(image, SIZE, SIZE)
        output = directory / f'{i:03d}.png'
        # PNG is saved BGR; RKNN's image reader supplies RGB then mean/std.
        assert cv2.imwrite(str(output), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        records.append(dict(index=i, source=str(source), source_sha256=sha256(source),
                            png=output.name, png_sha256=sha256(output), geometry=geometry))
    dataset = directory / 'dataset.txt'
    # 相对清单便于复制；build640.py 会按 WSL 当前目录生成实际绝对路径。
    dataset.write_text(''.join(f'{i:03d}.png\n' for i in range(160)), encoding='utf8')
    save_json(directory / 'manifest.json', dict(seed=42, count=160, split='train only',
                                               source_list=str(SPLIT / 'train.txt'),
                                               samples=records, dataset_sha256=sha256(dataset)))
    return dataset


def load_samples():
    video_path = VIDEO_ROOT / 'video_frames.npz'
    video_metadata = json.loads((VIDEO_ROOT / 'video_frames.json').read_text(encoding='utf8'))
    with np.load(video_path, allow_pickle=False) as data:
        videos = data['frames']
    assert len(videos) == 24
    samples = []
    for i, image in enumerate(videos):
        rgb, geometry = preprocess(image, SIZE, SIZE)
        samples.append((rgb, dict(index=i, kind='video', video_sample_id=i,
                                 video_source=video_metadata['samples'][i], geometry=geometry)))
    paths = split_paths('val')
    chosen = np.linspace(0, len(paths)-1, 12, dtype=int)
    assert len(set(chosen)) == 12
    for idx in chosen:
        path = paths[int(idx)]
        image = cv2.imread(str(path))
        assert image is not None, path
        rgb, geometry = preprocess(image, SIZE, SIZE)
        samples.append((rgb, dict(index=len(samples), kind='validation_image', image_path=str(path),
                                 image_sha256=sha256(path), geometry=geometry)))
    return samples


def main():
    global ROOT, DATA_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=DEFAULT_PT)
    parser.add_argument('--data-dir', type=Path, required=True, help='原 YOLO 图片数据集根目录')
    parser.add_argument('--output', type=Path, required=True, help='尚不存在的新导出工作目录')
    args = parser.parse_args()
    ROOT = args.output.resolve()
    DATA_DIR = args.data_dir.resolve()
    for protected in ('weights', 'results', 'data', 'src'):
        if ROOT.is_relative_to(ARCHIVE / protected):
            raise ValueError(f'输出不能位于归档 {protected} 目录内')
    if ROOT.is_relative_to(ARCHIVE.parent / 'v1') or ROOT.is_relative_to(ARCHIVE.parent / 'v2'):
        raise ValueError('不能覆盖 v1/v2 归档')
    if ROOT.exists():
        raise FileExistsError(f'Use a new --output directory: {ROOT}')
    assert args.weights.is_file(), f'Training weights missing: {args.weights}'
    assert DATA_DIR.is_dir(), DATA_DIR
    ROOT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    started = time.monotonic()
    samples = load_samples()
    weights_sha = sha256(args.weights)
    model = YOLO(str(args.weights)).model.cpu().float().eval()
    model.model[-1].end2end = True
    model.fuse()
    head = model.model[-1]
    assert head.nc == 1 and head.reg_max == 1 and head.end2end
    original_forward = head.forward
    arrays, sample_records = {}, []

    # First store official PT boxes and our six raw outputs for each real image.
    with torch.inference_mode():
        for rgb, metadata in samples:
            i = metadata['index']
            x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()[None]).float() / 255.
            head.forward = original_forward
            official = model(x)[0][0].cpu().numpy()
            official = official[official[:, 4] >= CONFIDENCE].astype(np.float32)
            assert official.ndim == 2 and official.shape[1] == 6
            head.forward = types.MethodType(raw_forward, head)
            raw = [output.cpu().numpy() for output in model(x)]
            assert [v.shape for v in raw] == EXPECTED
            own = as_six_columns(decode_yolo(raw, SIZE, SIZE, CONFIDENCE))
            agreement = check_detection_sets(official, own, f'PT/raw sample {i}', tolerance=.002)
            arrays[f'input_{i}'] = rgb
            arrays[f'boxes_{i}'] = official
            arrays[f'boxes_original_{i}'] = restore(official, metadata['geometry'])
            for j, output in enumerate(raw):
                arrays[f'raw_{i}_{j}'] = output
            sample_records.append(dict(metadata, pt_raw_agreement=agreement))
            print('PT_REFERENCE', i, metadata['kind'], len(official), flush=True)

    destination = ROOT / 'yolo640x640.onnx'
    # Fail instead of silently replacing a previous export from different weights.
    if destination.exists():
        raise FileExistsError(f'Export already exists; inspect before overwriting: {destination}')
    with torch.inference_mode():
        torch.onnx.export(model, torch.zeros(1, 3, SIZE, SIZE), str(destination),
                          input_names=['images'], output_names=OUTPUT_NAMES,
                          opset_version=12, dynamo=False, do_constant_folding=True)
    exported = onnx.load(str(destination))
    onnx.checker.check_model(exported)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(destination), options, providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape == [1, 3, SIZE, SIZE]
    max_abs = 0.
    for sample in sample_records:
        i = sample['index']
        rgb = arrays[f'input_{i}']
        x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.
        actual = session.run(None, {'images': x})
        assert [v.shape for v in actual] == EXPECTED
        errors = []
        for j, value in enumerate(actual):
            reference = arrays[f'raw_{i}_{j}']
            np.testing.assert_allclose(value, reference, atol=1e-3, rtol=1e-3,
                                       err_msg=f'sample {i} head {j}')
            errors.append(float(np.max(np.abs(value-reference))))
        sample['onnx_raw_max_abs_error'] = errors
        max_abs = max(max_abs, *errors)
        decoded = as_six_columns(decode_yolo(actual, SIZE, SIZE, CONFIDENCE))
        sample['pt_onnx_detection_agreement'] = check_detection_sets(arrays[f'boxes_{i}'], decoded,
                                                                   f'PT/ONNX sample {i}')
        print('ONNX_REFERENCE', i, max(errors), flush=True)
    # The source checkpoint must not have been overwritten while exporting.
    assert sha256(args.weights) == weights_sha
    reference_npz = ROOT / 'references.npz'
    np.savez_compressed(reference_npz, **arrays)
    reference_metadata = dict(count=len(samples), video_count=24, val_count=12, confidence=CONFIDENCE,
                              weights=str(args.weights), weights_sha256=weights_sha,
                              onnx_sha256=sha256(destination), reference_sha256=sha256(reference_npz),
                              outputs=dict(zip(OUTPUT_NAMES, EXPECTED)),
                              boxes_format='Nx6 xyxy,confidence,class; boxes_i in 640 canvas, boxes_original_i in original image',
                              input_format='input_i RGB UINT8 HWC 640x640; do not normalize before RKNN input',
                              video_frames_path=str(VIDEO_ROOT / 'video_frames.npz'),
                              video_frames_sha256=sha256(VIDEO_ROOT / 'video_frames.npz'),
                              validation_samples=sample_records)
    save_json(ROOT / 'references.json', reference_metadata)
    calibration = prepare_calibration()
    info = dict(status='verified', weights=str(args.weights), weights_sha256=weights_sha,
                onnx=str(destination), onnx_sha256=sha256(destination), onnx_opset=12,
                input_NCHW=[1, 3, SIZE, SIZE], outputs=dict(zip(OUTPUT_NAMES, EXPECTED)),
                decode='one-to-one LTRB / sigmoid; no DFL, no NMS, top300; undo centered114 letterbox',
                runtime_input='RGB UINT8 NHWC, mean0/std255 embedded in RKNN',
                validation_count=len(samples), max_onnx_raw_abs_error=max_abs,
                references=str(reference_npz), calibration_dataset=str(calibration), calibration_path_format='PNG paths relative to calibration640; build resolves actual paths',
                calibration_count=160, calibration_seed=42, seconds=time.monotonic()-started,
                board_validated=False)
    save_json(ROOT / 'export640.json', info)
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    main()
