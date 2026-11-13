"""Compare every tensor and final box for all24 changing video frames, one IO variant."""
import argparse,json,traceback
from pathlib import Path
import cv2,numpy as np
from rknn_runtime import RKModel,attr_dict
from io_variants import OptimizedRKModel
from rtmdet_postprocess import preprocess,decode

p=argparse.ArgumentParser();p.add_argument('variant',choices=['prealloc','quantized','dma'])
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
if args.output.exists():raise FileExistsError(args.output)
args.output.parent.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parent
frames=np.load(root.parent/'data/video_frames.npz')['frames']
path=root.parent/'weights/rtmdet-tiny-256x160/model-mmse-int8.rknn'
baseline=RKModel(path,'/usr/lib/librknnrt.so',256,160,1)
candidate=None;record=dict(variant=args.variant,frames=[],status='started')
try:
    candidate=OptimizedRKModel(path,'/usr/lib/librknnrt.so',256,160,1,variant=args.variant)
    for i,frame in enumerate(frames):
        rgb,geo=preprocess(frame,256,160)
        ref,_=baseline.run(rgb)
        got,_=candidate.run(rgb)
        differences=[float(np.max(np.abs(a-b))) for a,b in zip(ref,got)]
        boxes0=decode(ref,256,160,geo);boxes1=decode(got,256,160,geo)
        equal=all(np.allclose(a,b,rtol=2e-5,atol=2e-5) for a,b in zip(ref,got))
        box_equal=boxes0.shape==boxes1.shape and np.allclose(boxes0,boxes1,atol=.02,rtol=2e-4)
        record['frames'].append(dict(index=i,max_abs=differences,tensors_equal=equal,boxes_equal=bool(box_equal)))
        if not equal or not box_equal:raise RuntimeError(f'Mismatch frame{i}: {differences}')
    record['status']='pass'
except BaseException as exc:
    record.update(status='error',error=repr(exc),traceback=traceback.format_exc())
    raise
finally:
    if candidate is not None:candidate.close()
    baseline.close()
    args.output.write_text(json.dumps(record,indent=2))
    print(json.dumps(dict(status=record['status'],variant=args.variant,frames=len(record['frames']),
                         error=record.get('error'))),flush=True)
