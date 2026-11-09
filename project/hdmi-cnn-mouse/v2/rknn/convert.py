"""320×192 YOLO26n -> RK3588 RKNN，保留原来的六输出 one-to-one 协议。

Windows 训练虚拟环境执行 export，WSL 的 RKNN Toolkit2 环境执行 build。
export 需要 --data-dir；两步用 --output-dir 指向同一个新的工作目录。
第一步生成必要的 ONNX 中间文件和校准图片；第二步生成 FP16/INT8 并做模拟器检查。
不连接开发板；模拟器结果不代表板上速度或实机精度。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'work'
WORK = OUTPUT_DIR / 'intermediate'
EXPERIMENT = ROOT.parent / 'results'
WEIGHTS = ROOT.parent / 'weights' / '320x192.pt'
DATASET = ROOT.parent / 'data'  # export 时由 --data-dir 指定。
HEIGHT, WIDTH = 192, 320
OUTPUTS = ['box8', 'cls8', 'box16', 'cls16', 'box32', 'cls32']
SHAPES = [(1, c, HEIGHT // s, WIDTH // s) for s in (8, 16, 32) for c in (4, 1)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def letterbox_bgr(image):
    """保持长宽比，补边114；磁盘保存BGR，推理前转RGB。"""
    h, w = image.shape[:2]
    r = min(WIDTH / w, HEIGHT / h)
    nw, nh = round(w * r), round(h * r)
    left, top = round((WIDTH - nw) / 2 - 0.1), round((HEIGHT - nh) / 2 - 0.1)
    out = np.full((HEIGHT, WIDTH, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out


def decode(outputs, threshold=0.25):
    """reg_max=1：四路为LTRB距离；单类logit做sigmoid；one-to-one不做NMS。"""
    boxes = []
    assert len(outputs) == 6
    for i, stride in enumerate((8, 16, 32)):
        reg, logits = outputs[2*i:2*i+2]
        assert reg.shape == SHAPES[2*i] and logits.shape == SHAPES[2*i+1]
        scores = 1 / (1 + np.exp(-np.clip(logits[0, 0], -80, 80)))
        ys, xs = np.where(scores >= threshold)
        distances = reg[0, :, ys, xs]
        anchors = np.stack((xs + 0.5, ys + 0.5), axis=1)
        xyxy = np.concatenate((anchors - distances[:, :2], anchors + distances[:, 2:]), axis=1)*stride
        boxes.append(np.column_stack((xyxy, scores[ys, xs], np.zeros(len(xs)))))
    out = np.concatenate(boxes).astype(np.float32)
    return out[np.argsort(-out[:, 4])[:300]]


def compare_boxes(reference, result):
    """按IoU贪心一对一匹配；同时报告掉框/多框，不把仅匹配框误差当整体准确率。"""
    remaining = list(range(len(result)))
    matches = []
    for ref in reference:
        if not remaining:
            break
        candidates = result[remaining]
        lt = np.maximum(ref[:2], candidates[:, :2]); rb = np.minimum(ref[2:4], candidates[:, 2:4])
        inter = np.prod(np.maximum(rb-lt, 0), axis=1)
        area = np.prod(ref[2:4]-ref[:2]) + np.prod(candidates[:, 2:4]-candidates[:, :2], axis=1)
        ious = inter / np.maximum(area-inter, 1e-9)
        j = int(np.argmax(ious))
        if ious[j] < 0.5:
            continue
        pred = candidates[j]
        matches.append({'iou': float(ious[j]),
                        'center_error_input_px': float(np.linalg.norm((ref[:2]+ref[2:4]-pred[:2]-pred[2:4])/2)),
                        'score_error': float(abs(ref[4]-pred[4]))})
        remaining.pop(j)
    return {'reference_count': len(reference), 'result_count': len(result), 'matched': len(matches),
            'missed_vs_pt': len(reference)-len(matches), 'extra_vs_pt': len(result)-len(matches),
            'matches': matches}


def export():
    import types
    import torch
    import onnx
    import onnxruntime as ort
    import ultralytics
    from ultralytics import YOLO

    WORK.mkdir(parents=True, exist_ok=True)
    if (WORK / 'ball320x192.onnx').exists():
        raise FileExistsError('导出文件已存在，避免覆盖已有转换输入。')
    torch.set_num_threads(4)
    model = YOLO(str(WEIGHTS)).model.cpu().float().eval()
    # 必须先选分支再融合，否则默认推理设置可能把 one-to-one 分支删除。
    model.model[-1].end2end = True
    model.fuse()
    head = model.model[-1]
    assert head.nc == 1 and head.reg_max == 1 and head.end2end
    original_forward = head.forward

    def raw_forward(self, features):
        out = []
        for i in range(self.nl):
            out.extend((self.one2one_cv2[i](features[i]), self.one2one_cv3[i](features[i])))
        return tuple(out)

    # 归档清单相对于数据集根目录；不复制原始图片，不改变原来的采样顺序。
    train_paths = [str(DATASET / p) for p in
                   (EXPERIMENT/'split/train.txt').read_text(encoding='utf-8').splitlines()]
    val_paths = [str(DATASET / p) for p in
                 (EXPERIMENT/'split/val.txt').read_text(encoding='utf-8').splitlines()]
    assert not set(train_paths) & set(val_paths)
    chosen = np.random.default_rng(42).choice(len(train_paths), size=min(160, len(train_paths)), replace=False)
    calibration = WORK/'calibration'
    calibration.mkdir(exist_ok=True)
    calibration_sources = []
    for i, index in enumerate(chosen):
        path = train_paths[int(index)]
        frame = cv2.imread(path)
        assert frame is not None
        assert cv2.imwrite(str(calibration/f'{i:03d}.png'), letterbox_bgr(frame))
        calibration_sources.append(path)

    # 12张独立验证图：比较原PT解码结果、导出的原始头和ONNX。
    test_indices = np.linspace(0, len(val_paths)-1, 12, dtype=int)
    sample = {}
    samples_meta = []
    with torch.inference_mode():
        for i, index in enumerate(test_indices):
            path = val_paths[int(index)]
            rgb = letterbox_bgr(cv2.imread(path))[:, :, ::-1].copy()
            x = torch.from_numpy(rgb.transpose(2, 0, 1)[None].copy()).float()/255
            head.forward = original_forward
            original = model(x)[0][0].numpy()
            original = original[original[:, 4] >= .25]
            head.forward = types.MethodType(raw_forward, head)
            raw = [v.numpy() for v in model(x)]
            assert [v.shape for v in raw] == SHAPES
            decoded = decode(raw)
            agreement = compare_boxes(original, decoded)
            assert agreement['missed_vs_pt'] == agreement['extra_vs_pt'] == 0
            assert all(m['center_error_input_px'] < 1e-3 for m in agreement['matches'])
            sample[f'input_{i}'] = rgb
            sample[f'boxes_{i}'] = original
            for j, a in enumerate(raw):
                sample[f'raw_{i}_{j}'] = a
            samples_meta.append({'file': path, 'head_decode_check': agreement})
        torch.onnx.export(model, torch.zeros(1, 3, HEIGHT, WIDTH), str(WORK/'ball320x192.onnx'),
                          input_names=['images'], output_names=OUTPUTS,
                          opset_version=12, dynamo=False, do_constant_folding=True)
    onnx_model = onnx.load(str(WORK/'ball320x192.onnx'))
    onnx.checker.check_model(onnx_model)
    session_options = ort.SessionOptions(); session_options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(WORK/'ball320x192.onnx'), sess_options=session_options,
                                   providers=['CPUExecutionProvider'])
    raw_max_errors = []
    for i in range(len(test_indices)):
        x = sample[f'input_{i}'].transpose(2, 0, 1)[None].astype(np.float32)/255
        outputs = session.run(None, {'images': x})
        for j, arr in enumerate(outputs):
            raw_max_errors.append(float(np.max(np.abs(arr-sample[f'raw_{i}_{j}']))))
            np.testing.assert_allclose(arr, sample[f'raw_{i}_{j}'], atol=1e-3, rtol=1e-3)
        agreement = compare_boxes(sample[f'boxes_{i}'], decode(outputs))
        assert agreement['missed_vs_pt'] == agreement['extra_vs_pt'] == 0
        samples_meta[i]['onnx_check'] = agreement
    np.savez_compressed(WORK/'references.npz', **sample)
    manifest = {'source_pt': str(WEIGHTS), 'source_sha256': sha256(WEIGHTS),
                'source_metrics_fp32': json.loads((EXPERIMENT/'experiment.json').read_text())['results']['320x192']['metrics_fp32'],
                'source_metrics_note': 'Prior training comparison used its default inference branch. These metrics are not a fresh mAP evaluation of this explicitly selected one-to-one RKNN export.',
                'ultralytics': ultralytics.__version__, 'torch_export': torch.__version__,
                'target': 'rk3588', 'toolkit_version': '2.3.2', 'onnx_opset': 12,
                'onnx_ir': onnx_model.ir_version, 'onnx_sha256': sha256(WORK/'ball320x192.onnx'),
                'input_NCHW': [1, 3, HEIGHT, WIDTH], 'runtime_input': 'RGB uint8 NHWC [1,192,320,3], pass_through=0; mean=0 std=255 inside RKNN; do not divide by 255 again',
                'resize': 'aspect-preserving letterbox 114, width=320 height=192',
                'outputs': dict(zip(OUTPUTS, SHAPES)), 'decode': 'stride 8/16/32, LTRB distances, sigmoid single-class logits, one-to-one/no NMS; map boxes back via letterbox ratio and offsets',
                'calibration_sources': calibration_sources, 'validation_samples': samples_meta,
                'onnx_max_abs_error': max(raw_max_errors), 'board_validated': False,
                'important': 'Existing 320-square npu_model.py cannot directly consume this model: update input H/W, per-stride grid H/W and letterbox before board use.'}
    (OUTPUT_DIR/'conversion.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('EXPORT CHECK PASSED', 'max raw error=', max(raw_max_errors), flush=True)


def build():
    from rknn.api import RKNN
    import importlib.metadata
    assert importlib.metadata.version('rknn-toolkit2') == '2.3.2'
    os.chdir(WORK)
    dataset = WORK/'dataset.txt'
    dataset.write_text('\n'.join(str(p) for p in sorted((WORK/'calibration').glob('*.png'))) + '\n')
    manifest = json.loads((OUTPUT_DIR/'conversion.json').read_text(encoding='utf-8'))
    references = np.load(WORK/'references.npz')
    manifest['builds'] = {}
    for kind in ('fp16', 'int8'):
        destination = OUTPUT_DIR/f'ball320x192-{kind}.rknn'
        if destination.exists():
            raise FileExistsError(str(destination))
        rknn = RKNN(verbose=False)
        try:
            rknn.config(target_platform='rk3588', mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                        quantized_dtype='asymmetric_quantized-8', optimization_level=3)
            assert rknn.load_onnx(model=str(WORK/'ball320x192.onnx')) == 0
            assert rknn.build(do_quantization=kind == 'int8', dataset=str(dataset) if kind == 'int8' else None) == 0
            assert rknn.export_rknn(str(destination)) == 0
            item = {'file': destination.name, 'bytes': destination.stat().st_size,
                    'sha256': sha256(destination), 'do_quantization': kind == 'int8',
                    'board_validated': False}
            manifest['builds'][kind] = item
            (OUTPUT_DIR/'conversion.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
            # 无 target 的 init_runtime 启用 PC 模拟器，不需要板子。
            status = rknn.init_runtime()
            item['simulator_init_return'] = status
            if status == 0:
                checks = []
                for i in range(len(manifest['validation_samples'])):
                    outputs = rknn.inference(inputs=[references[f'input_{i}'][None]], data_format=['nhwc'])
                    assert outputs is not None and [a.shape for a in outputs] == SHAPES
                    assert all(np.isfinite(a).all() for a in outputs)
                    checks.append(compare_boxes(references[f'boxes_{i}'], decode(outputs)))
                matches = [m for c in checks for m in c['matches']]
                item['simulator'] = {'samples': len(checks), 'reference_boxes': sum(c['reference_count'] for c in checks),
                    'missed_vs_pt': sum(c['missed_vs_pt'] for c in checks), 'extra_vs_pt': sum(c['extra_vs_pt'] for c in checks),
                    'mean_iou': float(np.mean([m['iou'] for m in matches])) if matches else None,
                    'max_center_error_input_px': max((m['center_error_input_px'] for m in matches), default=None),
                    'checks': checks, 'note': 'Agreement with PT at confidence 0.25, IoU-match>=0.5 on 12 samples, not full mAP or real-board validation.'}
            (OUTPUT_DIR/'conversion.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
            print('BUILT', kind, json.dumps(item, ensure_ascii=False), flush=True)
        finally:
            rknn.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['export', 'build'])
    parser.add_argument('--data-dir', type=Path, help='仅 export 需要，原始数据集目录')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='export 创建新目录；build 指向这一步生成的同一目录')
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir.expanduser().resolve()
    WORK = OUTPUT_DIR / 'intermediate'
    if OUTPUT_DIR == ROOT:
        parser.error('不能覆盖归档目录，请使用新的工作目录')
    if args.step == 'export':
        if args.data_dir is None or not args.data_dir.is_dir():
            parser.error('export 需要有效的 --data-dir')
        DATASET = args.data_dir.expanduser().resolve()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
        export()
    else:
        if not (OUTPUT_DIR / 'conversion.json').is_file() or not (WORK / 'references.npz').is_file():
            parser.error('缺少 export 产物，请先在 Windows 执行 export')
        build()
