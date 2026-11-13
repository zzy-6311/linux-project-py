"""Compare real-board INT8 detections to PT; separately validate IO/core equivalence."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from rknn_runtime import RKModel
from io_variants import OptimizedRKModel
from yolo_helpers import decode_yolo

ROOT = Path(__file__).resolve().parent


def compare(reference, actual):
    remaining = list(range(len(actual)))
    pairs = []
    for ref in reference:
        if not remaining:
            break
        candidate = actual[remaining]
        lt = np.maximum(ref[:2], candidate[:, :2])
        rb = np.minimum(ref[2:4], candidate[:, 2:4])
        inter = np.prod(np.maximum(rb - lt, 0), axis=1)
        union = np.prod(ref[2:4] - ref[:2]) + np.prod(candidate[:, 2:4] - candidate[:, :2], axis=1) - inter
        ious = inter / np.maximum(union, 1e-9)
        best = int(np.argmax(ious))
        if ious[best] < 0.5:
            continue
        pred = candidate[best]
        pairs.append(dict(iou=float(ious[best]),
                          center_error_input_px=float(np.linalg.norm((ref[:2] + ref[2:4] - pred[:2] - pred[2:4]) / 2)),
                          score_error=float(abs(ref[4] - pred[4]))))
        remaining.pop(best)
    return dict(reference_count=len(reference), actual_count=len(actual), matched=len(pairs),
                missed=len(reference) - len(pairs), extra=len(actual) - len(pairs), pairs=pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=ROOT.parent/'weights/yolo26n-640x640/model-int8.rknn')
    parser.add_argument('--references',type=Path,default=ROOT.parent/'data/validation640')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    ref = np.load(args.references / 'references.npz',allow_pickle=False)
    meta = json.loads((args.references / 'references.json').read_text())
    model_path = args.model
    record = dict(model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  note='Agreement with PT on 24 cached video frames plus 12 held-out validation images, not dataset mAP.',
                  samples=[], equivalence=[])
    outputs_ref = []
    baseline = RKModel(model_path, '/usr/lib/librknnrt.so', 640, 640, 7)
    try:
        record['runtime'] = baseline.runtime_version
        record['driver'] = baseline.driver_version
        for item in meta['validation_samples']:
            i = item['index']
            outputs, timing = baseline.run(ref[f'input_{i}'])
            assert all(np.isfinite(v).all() for v in outputs)
            boxes = decode_yolo(outputs, 640, 640, 0.25)
            checked = compare(ref[f'boxes_{i}'], boxes)
            checked.update(index=i, kind=item['kind'], boxes=boxes.tolist())
            record['samples'].append(checked)
            outputs_ref.append(outputs)
    finally:
        baseline.close()
    # Same weights/inputs must retain outputs when core assignment or buffer ownership changes.
    for mask in (1, 2, 3, 4, 7):
        model = OptimizedRKModel(model_path, '/usr/lib/librknnrt.so', 640, 640, mask, variant='prealloc')
        max_error = 0.0
        try:
            for i in range(meta['count']):
                values, _ = model.run(ref[f'input_{i}'])
                for a, b in zip(outputs_ref[i], values):
                    max_error = max(max_error, float(np.max(np.abs(a - b))))
                    np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
        finally:
            model.close()
        record['equivalence'].append(dict(core_mask=mask, io='prealloc', raw_max_abs_error=max_error))
    pairs = [p for s in record['samples'] for p in s['pairs']]
    record['summary'] = dict(images=len(record['samples']),
                            reference_boxes=sum(s['reference_count'] for s in record['samples']),
                            actual_boxes=sum(s['actual_count'] for s in record['samples']),
                            matched=sum(s['matched'] for s in record['samples']),
                            missed=sum(s['missed'] for s in record['samples']),
                            extra=sum(s['extra'] for s in record['samples']),
                            mean_iou=float(np.mean([p['iou'] for p in pairs])) if pairs else None,
                            max_center_error_input_px=max((p['center_error_input_px'] for p in pairs), default=None))
    args.output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record['summary']), flush=True)
    print('CORE_AND_IO_EQUIVALENCE', json.dumps(record['equivalence']), flush=True)
    if not pairs:
        raise RuntimeError('No detections agree with PT; do not benchmark an invalid model.')


if __name__ == '__main__':
    main()
