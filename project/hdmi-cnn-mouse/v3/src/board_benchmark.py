"""Isolated RK3588 batch=1 concurrency benchmark on cached real video frames.

No HDMI, mouse, service or model modifications. Every worker owns its context.
Full timing includes preprocessing and boxes. RGB timing skips preprocessing but
still includes RKNN I/O+run and box decoding. Neither is a pure rknn_run benchmark.
No pending queue: workers request the next frame only after finishing their own.
Frequency settings are restored in finally; results include errors and checks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from rknn_runtime import RKModel, snapshot, attr_dict
from rtmdet_postprocess import preprocess, decode


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(mean=float(a.mean()), p50=float(np.median(a)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), min=float(a.min()), max=float(a.max()))


class Clocks:
    def __init__(self):
        self.npu = Path('/sys/class/devfreq/fdab0000.npu')
        self.dmc = Path('/sys/class/devfreq/dmc')
        self.cpus = [p/'scaling_governor' for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*')]
        paths = self.cpus + [d/f for d in (self.npu, self.dmc) for f in ('governor','min_freq','max_freq')]
        self.saved = {str(p):p.read_text().strip() for p in paths}

    def set(self):
        for p in self.cpus:
            p.write_text('performance')
        for d, value in ((self.npu, '1000000000'), (self.dmc, '1560000000')):
            if value not in (d/'available_frequencies').read_text().split():
                raise RuntimeError(f'Unavailable test frequency {d}: {value}')
            (d/'max_freq').write_text(value)
            (d/'min_freq').write_text(value)
            (d/'governor').write_text('performance')

    def restore(self):
        errors = []
        for d in (self.npu, self.dmc):
            for f in ('min_freq','max_freq','governor'):
                p = d/f
                try: p.write_text(self.saved[str(p)])
                except Exception as exc: errors.append(f'{p}: {exc}')
        for p in self.cpus:
            try: p.write_text(self.saved[str(p)])
            except Exception as exc: errors.append(f'{p}: {exc}')
        restored = {p:Path(p).read_text().strip() for p in self.saved}
        return dict(saved=self.saved, restored=restored, errors=errors,
                    exact=restored==self.saved and not errors)


def make_model(args, mask, io):
    if io == 'baseline':
        return RKModel(args.model, args.library, args.width, args.height, mask)
    from io_variants import OptimizedRKModel
    return OptimizedRKModel(args.model, args.library, args.width, args.height, mask, variant=io)


class ReusablePreprocessor:
    """Reuse full RGB/pad buffers; cv2 writes resized pixels directly into pad ROI."""
    def __init__(self,width,height,geometry):
        self.geometry=geometry
        self.canvas=np.full((height,width,3),114,np.uint8)
        self.rgb=np.empty_like(self.canvas)
        self.nw,self.nh=geometry['resized_WH']
        self.left,self.top=geometry['left'],geometry['top']
        self.roi=self.canvas[self.top:self.top+self.nh,self.left:self.left+self.nw]

    def run(self,image):
        resized=cv2.resize(image,(self.nw,self.nh),dst=self.roi,interpolation=cv2.INTER_LINEAR)
        if not np.shares_memory(resized,self.canvas):
            raise RuntimeError('OpenCV did not use the reusable resize destination')
        cv2.cvtColor(self.canvas,cv2.COLOR_BGR2RGB,dst=self.rgb)
        return self.rgb,self.geometry


def run_case(args, frames, prepared, refs, masks, mode, io, round_index):
    stop = threading.Event()
    gate_time = {}
    barrier = threading.Barrier(len(masks)+1, action=lambda: gate_time.update(start=time.perf_counter_ns()))
    models = []
    try:
        for mask in masks:
            models.append(make_model(args,mask,io))
    except BaseException:
        for model in models:model.close()
        raise
    samples = []
    lock = threading.Lock()
    next_id = 0
    load_samples = []

    def monitor():
        while not stop.wait(.2):
            load_samples.append(snapshot())

    def worker(worker_id):
        nonlocal next_id
        os.sched_setaffinity(0, {4+worker_id%4})
        model=models[worker_id]
        local=[]
        try:
            reusable=ReusablePreprocessor(args.width,args.height,prepared[0][1]) if mode=='reuse' else None
            if reusable:
                for idx,frame in enumerate(frames):
                    rgb,geo=reusable.run(frame)
                    if geo!=prepared[idx][1] or not np.array_equal(rgb,prepared[idx][0]):
                        raise RuntimeError('Reusable preprocessor differs from original RGB pixels')
            def process(frame_id):
                idx=frame_id%len(frames)
                t0=time.perf_counter_ns()
                if mode=='full': rgb,geo=preprocess(frames[idx],args.width,args.height)
                elif mode=='reuse': rgb,geo=reusable.run(frames[idx])
                else: rgb,geo=prepared[idx]
                t1=time.perf_counter_ns()
                outputs,timing=model.run(rgb)
                t2=time.perf_counter_ns()
                boxes=decode(outputs,args.width,args.height,geo,confidence=.25)
                t3=time.perf_counter_ns()
                # Time excludes validation; every frame is still checked.
                expected=refs[idx]
                if boxes.shape!=expected.shape or not np.allclose(boxes,expected,rtol=2e-4,atol=.02):
                    raise RuntimeError(f'Box mismatch worker{worker_id}, frame{idx}: {boxes.tolist()} vs {expected.tolist()}')
                return dict(frame_id=frame_id,source_index=idx,worker=worker_id,
                            preprocess_ms=(t1-t0)/1e6,call_ms=timing['call_ms'],run_ms=timing['rknn_run_ms'],
                            postprocess_ms=(t3-t2)/1e6,latency_ms=(t3-t0)/1e6,
                            start_ns=t0,end_ns=t3,box_count=len(boxes))
            for k in range(args.warmup):process(k+worker_id)
            barrier.wait(timeout=90)
            while not stop.is_set():
                with lock:
                    if next_id>=args.frames:break
                    frame_id=next_id
                    next_id+=1
                local.append(process(frame_id))
            return local
        except BaseException:
            stop.set();barrier.abort()
            raise

    mon=threading.Thread(target=monitor,daemon=True)
    try:
        with ThreadPoolExecutor(max_workers=len(masks)) as pool:
            futures=[pool.submit(worker,i) for i in range(len(masks))]
            barrier.wait(timeout=90)
            start=gate_time['start']
            mon.start()
            for f in futures:samples.extend(f.result())
            end=time.perf_counter_ns()
        stop.set();mon.join(timeout=2)
        ids=[s['frame_id'] for s in samples]
        if len(ids)!=args.frames or len(set(ids))!=args.frames:
            raise RuntimeError('Missing or duplicate measured frame IDs')
        active_ns=max(s['end_ns'] for s in samples)-min(s['start_ns'] for s in samples)
        return dict(round=round_index,core_masks=masks,workers=len(masks),io=io,mode=mode,
                    frames=len(samples),wall_ms=(end-start)/1e6,fps=len(samples)/((end-start)/1e9),
                    active_fps=len(samples)/(active_ns/1e9),
                    statistics={key:stats([s[key] for s in samples]) for key in
                                ('latency_ms','preprocess_ms','call_ms','run_ms','postprocess_ms')},
                    output_validation='all_frames_match_single_context_reference',
                    samples=sorted(samples,key=lambda s:s['frame_id']),environment=load_samples)
    finally:
        stop.set()
        if mon.is_alive():mon.join(timeout=2)
        for model in models:model.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--inputs',type=Path,default=ROOT.parent/'data/video_frames.npz')
    p.add_argument('--library',default='/usr/lib/librknnrt.so')
    p.add_argument('--family',choices=['rtmdet','yolo'],default='rtmdet')
    p.add_argument('--width',type=int,default=256);p.add_argument('--height',type=int,default=160)
    p.add_argument('--frames',type=int,default=600);p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--configs',default='7;1;1,2,4;1,2,4,1,2,4')
    p.add_argument('--modes',default='full,rgb')
    p.add_argument('--io',default='baseline')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.family=='yolo':
        global preprocess,decode
        from yolo_helpers import preprocess,decode
    cv2.setNumThreads(1)
    def signaled(number,frame):raise KeyboardInterrupt(f'Signal {number}')
    signal.signal(signal.SIGTERM,signaled);signal.signal(signal.SIGINT,signaled)
    data=np.load(args.inputs, allow_pickle=False)
    frames=data['frames']
    prepared=[preprocess(frame,args.width,args.height) for frame in frames]
    output=args.output.resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    configs=[list(map(int,c.split(','))) for c in args.configs.split(';')]
    clocks=Clocks()
    record=dict(status='started',args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},input_shape=list(frames.shape),
                benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                input_metadata=json.loads(args.inputs.with_suffix('.json').read_text()),
                timing_scope='cached decoded video frames -> boxes; rgb mode omits resize. reuse mode includes exactpixel-equivalent reusable preprocessing. No file decode, HDMI or mouse. No pending work queue; one active frame per worker. Throughput includes output validation overhead.',
                initial_environment=snapshot(),cases=[])
    try:
        clocks.set()
        os.sched_setaffinity(0,{0,1,2,3,4,5,6,7})
        reference=make_model(args,7,'baseline')
        try:
            refs=[decode(reference.run(rgb)[0],args.width,args.height,geo,.25) for rgb,geo in prepared]
            record['reference_boxes']=[b.tolist() for b in refs]
            record['tensor_attrs']=[attr_dict(a) for a in reference.attrs]
            record['runtime_version']=reference.runtime_version
            record['driver_version']=reference.driver_version
        finally:reference.close()
        basecases=[(c,m,io) for io in args.io.split(',') for m in args.modes.split(',') for c in configs]
        for rnd in range(args.rounds):
            # Alternate the order to reduce monotonic warming/temperature bias.
            cases=basecases if rnd%2==0 else list(reversed(basecases))
            for masks,mode,io in cases:
                result=run_case(args,frames,prepared,refs,masks,mode,io,rnd+1)
                record['cases'].append(result)
                print(json.dumps(dict(round=rnd+1,masks=masks,mode=mode,io=io,
                                      fps=round(result['fps'],2),latency=result['statistics']['latency_ms'],
                                      npu_run=result['statistics']['run_ms'])),flush=True)
                output.write_text(json.dumps(record,indent=2))
        record['status']='complete'
    except BaseException as exc:
        record.update(status='error',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        record['clock_restore']=clocks.restore()
        record['final_environment']=snapshot()
        output.write_text(json.dumps(record,indent=2))
        print('SAVED',output,'restored',record['clock_restore']['exact'],flush=True)


if __name__=='__main__':main()
