"""Build INT8 YOLO26n640 using the existing WSL RKNN Toolkit2 2.5.0.

Uses the same quantization recipe as the 320x192 comparison. Does not install,
download, invoke a board, or alter weights. Only run after export640.py succeeds.
Archive edition: --work points to that fresh export; calibration paths are rebased.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time


ARCHIVE = Path(__file__).resolve().parents[1]
DISABLED = ['fuse_exmatmul_add_mul_exsoftmax13_exmatmul_to_sdpa']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True, help='export640.py 创建的工作目录，在 WSL 使用 /mnt/... 路径')
    args = parser.parse_args()
    root = args.work.resolve()
    for protected in ('weights', 'results', 'data', 'src'):
        if root.is_relative_to(ARCHIVE / protected):
            raise ValueError(f'工作目录不能位于归档 {protected} 内')
    if root.is_relative_to(ARCHIVE.parent / 'v1') or root.is_relative_to(ARCHIVE.parent / 'v2'):
        raise ValueError('不能覆盖 v1/v2 归档')
    destination = root / 'yolo640x640-int8.rknn'
    assert not destination.exists(), f'Refusing to overwrite existing model: {destination}'
    if (root / 'yolo640x640-int8.build.json').exists():
        raise FileExistsError('已有构建记录；请使用新的导出工作目录再构建')
    from rknn.api import RKNN
    assert version('rknn-toolkit2') == '2.5.0'
    metadata = json.loads((root / 'export640.json').read_text(encoding='utf8'))
    assert metadata['status'] == 'verified' and metadata['validation_count'] == 36
    model = root / 'yolo640x640.onnx'
    dataset = root / 'calibration640/dataset.txt'
    manifest_path = root / 'calibration640/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    assert manifest['count'] == 160 and len(manifest['samples']) == 160
    paths = []
    for sample in manifest['samples']:
        path = (root / 'calibration640' / sample['png']).resolve()
        assert path.is_relative_to(root / 'calibration640') and path.is_file(), path
        assert digest(path) == sample['png_sha256'], f'Calibration image changed: {path}'
        paths.append(path)
    assert len(set(paths)) == 160
    assert digest(model) == metadata['onnx_sha256']
    # 导出端可能是 Windows，构建端是 WSL；只重建路径，不重做/改变校准图。
    dataset.write_text(''.join(path.as_posix() + '\n' for path in paths), encoding='utf8')
    record = dict(status='started', kind='yolo26n', shape_WH='640x640', precision='int8',
                  toolkit=version('rknn-toolkit2'), onnx_sha256=digest(model),
                  weights_sha256=metadata['weights_sha256'], calibration_count=len(paths),
                  calibration_dataset_sha256=digest(dataset), calibration_manifest_sha256=digest(manifest_path),
                  calibration_png_hashes_verified=True, disabled_fusion_rules=DISABLED,
                  optimization_level=3, graph_weights_modified=False,
                  quantized_dtype='asymmetric_quantized-8', quantized_algorithm='normal',
                  quantization_recipe='Toolkit default normal/minmax, matching prior YOLO320x192',
                  mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                  workaround_source='https://github.com/airockchip/rknn-toolkit2/issues/581')
    rknn = RKNN(verbose=False)
    started = time.monotonic()
    try:
        # Keep unfused attention operations, avoiding the known 2.5 SDPA fusion bug.
        # The disabled optimization changes no trained tensors or model semantics.
        assert rknn.config(target_platform='rk3588', mean_values=[[0, 0, 0]],
                           std_values=[[255, 255, 255]], quantized_dtype='asymmetric_quantized-8',
                           quantized_algorithm='normal', optimization_level=3,
                           disable_rules=DISABLED) == 0
        assert rknn.load_onnx(model=str(model)) == 0
        assert rknn.build(do_quantization=True, dataset=str(dataset)) == 0
        assert rknn.export_rknn(str(destination)) == 0
        record.update(status='built', bytes=destination.stat().st_size, sha256=digest(destination))
    except Exception as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        rknn.release()
        record['seconds'] = time.monotonic() - started
        temporary = root / 'yolo640x640-int8.build.json.tmp'
        temporary.write_text(json.dumps(record, indent=2), encoding='utf8')
        temporary.replace(root / 'yolo640x640-int8.build.json')
        print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
