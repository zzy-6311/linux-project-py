"""Render sampled PT/INT8 detections for a conversion sanity check, without a GUI."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise FileExistsError(args.output)
args.output.parent.mkdir(parents=True,exist_ok=True)
refs = np.load(ROOT.parent / 'data/validation640/references.npz',allow_pickle=False)
data = json.loads((ROOT.parent / 'results/validation640/board-validation.json').read_text())
rows = []
for index in (8, 17, 26):
    frame = cv2.cvtColor(refs[f'input_{index}'], cv2.COLOR_RGB2BGR)
    halves = []
    for name, boxes, color in (('PT', refs[f'boxes_{index}'], (255, 200, 0)),
                                ('RKNN INT8', data['samples'][index]['boxes'], (0, 220, 0))):
        image = frame.copy()
        for box in boxes:
            x1, y1, x2, y2 = np.rint(box[:4]).astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            cv2.putText(image, f'{box[4]:.2f}', (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, .3, color, 1)
        cv2.putText(image, f'Sample {index}: {name} ({len(boxes)} boxes)', (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, color, 1)
        halves.append(image)
    rows.append(np.hstack(halves))
if not cv2.imwrite(str(args.output), np.vstack(rows)):raise RuntimeError('Failed to write preview')
