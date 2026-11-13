"""从归档原始采样重新生成总表；只读原记录，不访问板子、不运行模型。"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    'multicore-small/multicore-rtmdet256160.json',
    'multicore-small/multicore-yolo320192.json',
    'multicore-small/io-rtmdet256160.json',
    'multicore-small/steady-rtmdet256160.json',
    'multicore-small/steady-yolo320192.json',
    'multicore640/multicore640-baseline.json',
    'multicore640/multicore640-reuse.json',
    'multicore640/multicore640-io-pair.json',
]
MODELS = {
    '3011635986f967421f53e28ec5bcc7469a0cca1d730ecbede737147302ba4ec9': 'YOLO26n 320x192 INT8',
    '46df509a2e84dbbe931b716ec187bb123db25bd25fa7ce5d9ed964a816efff8e': 'RTMDet-tiny 256x160 MMSE INT8',
    '7993b641fde1d9fba69d7c20333da636b26b6b572a743e8a02eda8483ddbccbd': 'YOLO26n 640x640 INT8',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    # 用新目录保存，避免一次复现覆盖历史证据。
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, sources, total = [], [], 0
    for name in FILES:
        path = ROOT/'results'/name
        record = json.loads(path.read_text(encoding='utf8'))
        assert record['status'] == 'complete' and record['clock_restore']['exact'], name
        model = MODELS[record['model_sha256']]
        groups = defaultdict(list)
        for case in record['cases']:
            assert case['output_validation'] == 'all_frames_match_single_context_reference'
            ids = [sample['frame_id'] for sample in case['samples']]
            assert len(ids) == case['frames'] and sorted(ids) == list(range(case['frames']))
            groups[(tuple(case['core_masks']), case['mode'], case['io'])].append(case)
        sources.append(dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            model=model, model_sha256=record['model_sha256'],
                            clock_restore_exact=True))
        for (masks, mode, io), cases in groups.items():
            samples = [s for case in cases for s in case['samples']]
            count = len(samples)
            total += count
            # 分子分母分别求和，不能直接平均不同耗时轮次的FPS。
            row = dict(source=name, model=model, masks=','.join(map(str, masks)),
                       workers=len(masks), mode=mode, io=io, frames=count,
                       rounds=len(cases), fps=count*1000/sum(c['wall_ms'] for c in cases))
            for field in ('latency_ms', 'preprocess_ms', 'call_ms', 'run_ms', 'postprocess_ms'):
                values = np.asarray([s[field] for s in samples])
                row[field.replace('_ms', '_mean_ms')] = float(values.mean())
                if field == 'latency_ms':
                    row['latency_p95_ms'] = float(np.percentile(values, 95))
            row['api_io_mean_ms'] = row['call_mean_ms'] - row['run_mean_ms']
            rows.append(row)
    result = dict(total_measured_frames=total, sources=sources, rows=rows,
                  scope='24 cached decoded 1080p frames, repeated; no storage read, decoding, HDMI, display or mouse. FPS is concurrent throughput, latency is individual-frame time. run_ms includes runtime scheduling/waiting.',
                  grouping='Each original experiment remains a separate group; no mixing of full, rgb and reuse modes.')
    (args.output_dir/'all-results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    with (args.output_dir/'all-results.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(experiments=len(sources), rows=len(rows), measured_frames=total,
                          output=str(args.output_dir)), ensure_ascii=False))


if __name__ == '__main__':
    main()
