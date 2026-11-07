"""YOLO26 one-to-one 原始检测头 + RKNN 2.3.2 C API，支持板子 Python 3.14。

输入：RGB uint8 NHWC，320×320，等比例缩放并以 114 补边。
输出：三个尺度各自的四路距离和单类 logit；CPU 解码，无 NMS。
"""
import ctypes as C
from pathlib import Path
import time
import cv2
import numpy as np

class Attr(C.Structure):
    _fields_=[('index',C.c_uint32),('n_dims',C.c_uint32),('dims',C.c_uint32*16),
              ('name',C.c_char*256),('n_elems',C.c_uint32),('size',C.c_uint32),
              ('fmt',C.c_int),('type',C.c_int),('qnt_type',C.c_int),('fl',C.c_int8),
              ('zp',C.c_int32),('scale',C.c_float),('w_stride',C.c_uint32),
              ('size_with_stride',C.c_uint32),('pass_through',C.c_uint8),('h_stride',C.c_uint32)]
class Input(C.Structure):
    _fields_=[('index',C.c_uint32),('buf',C.c_void_p),('size',C.c_uint32),
              ('pass_through',C.c_uint8),('type',C.c_int),('fmt',C.c_int)]
class Output(C.Structure):
    _fields_=[('want_float',C.c_uint8),('is_prealloc',C.c_uint8),('index',C.c_uint32),
              ('buf',C.c_void_p),('size',C.c_uint32)]

def letterbox(image):
    h,w=image.shape[:2]; ratio=min(320/w,320/h)
    nw,nh=round(w*ratio),round(h*ratio)
    left,top=round((320-nw)/2-.1),round((320-nh)/2-.1)
    result=np.full((320,320,3),114,np.uint8)
    result[top:top+nh,left:left+nw]=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(result[:,:,::-1]),ratio,left,top

def decode(outputs, threshold=.25):
    boxes=[]
    if len(outputs)!=6:raise ValueError('模型不是六输出 one-to-one 检测头')
    for i,stride in enumerate((8,16,32)):
        side=320//stride
        reg,logits=outputs[2*i:2*i+2]
        if reg.shape!=(1,4,side,side) or logits.shape!=(1,1,side,side):
            raise ValueError(f'输出形状错误：{reg.shape}, {logits.shape}')
        scores=1/(1+np.exp(-np.clip(logits[0,0],-80,80)))
        ys,xs=np.where(scores>=threshold)
        if not len(xs):continue
        distances=reg[0,:,ys,xs]  # (N,4)
        anchors=np.stack((xs+.5,ys+.5),axis=1)
        xyxy=np.concatenate((anchors-distances[:,:2],anchors+distances[:,2:]),axis=1)*stride
        boxes.append(np.column_stack((xyxy,scores[ys,xs],np.zeros(len(xs)))))
    if not boxes:return np.empty((0,6),np.float32)
    values=np.concatenate(boxes).astype(np.float32)
    values=values[np.isfinite(values).all(axis=1)]
    return values[np.argsort(-values[:,4])[:300]]

class NPUModel:
    def __init__(self,path,core_mask=7):
        self.ctx=C.c_uint64(0)
        here=Path(__file__).resolve().parent
        self.lib=C.CDLL(str(here/'librknnrt.so'))
        signatures={
            'rknn_init':([C.POINTER(C.c_uint64),C.c_void_p,C.c_uint32,C.c_uint32,C.c_void_p],C.c_int),
            'rknn_query':([C.c_uint64,C.c_int,C.c_void_p,C.c_uint32],C.c_int),
            'rknn_set_core_mask':([C.c_uint64,C.c_int],C.c_int),
            'rknn_inputs_set':([C.c_uint64,C.c_uint32,C.POINTER(Input)],C.c_int),
            'rknn_run':([C.c_uint64,C.c_void_p],C.c_int),
            'rknn_outputs_get':([C.c_uint64,C.c_uint32,C.POINTER(Output),C.c_void_p],C.c_int),
            'rknn_outputs_release':([C.c_uint64,C.c_uint32,C.POINTER(Output)],C.c_int),
            'rknn_destroy':([C.c_uint64],C.c_int)}
        for name,(args,result) in signatures.items():
            f=getattr(self.lib,name); f.argtypes=args;f.restype=result
        blob=Path(path).read_bytes(); data=C.create_string_buffer(blob)
        self.check(self.lib.rknn_init(C.byref(self.ctx),data,len(blob),0,None),'init')
        try:
            self.check(self.lib.rknn_set_core_mask(self.ctx,core_mask),'core mask')
            counts=(C.c_uint32*2)();self.query(0,counts)
            if list(counts)!=[1,6]:raise ValueError(f'输入输出数量不符：{list(counts)}')
            self.attrs=[]
            for i in range(6):
                a=Attr(index=i);self.query(2,a);self.attrs.append(a)
            a=Attr(index=0);self.query(1,a)
            self.input_shape=list(a.dims[:a.n_dims])
            if self.input_shape not in ([1,320,320,3],[1,3,320,320]):
                raise ValueError(f'输入形状不符：{self.input_shape}')
            version=C.create_string_buffer(512);self.query(5,version)
            self.version=version.raw[:256].split(b'\0')[0].decode()
            self.driver=version.raw[256:].split(b'\0')[0].decode()
        except Exception:self.close();raise
        self.last_ms=0.

    @staticmethod
    def check(code,operation):
        if code:raise RuntimeError(f'RKNN {operation} failed: {code}')
    def query(self,cmd,value):
        self.check(self.lib.rknn_query(self.ctx,cmd,C.byref(value),C.sizeof(value)),'query')
    def run(self,rgb):
        if rgb.shape!=(320,320,3) or rgb.dtype!=np.uint8:raise ValueError('输入必须为320 RGB uint8')
        rgb=np.ascontiguousarray(rgb)
        inputs=Input(0,rgb.ctypes.data,rgb.nbytes,0,3,1)
        start=time.perf_counter()
        self.check(self.lib.rknn_inputs_set(self.ctx,1,C.byref(inputs)),'input')
        self.check(self.lib.rknn_run(self.ctx,None),'run')
        values=(Output*6)(*(Output(1,0,i,None,0) for i in range(6)))
        self.check(self.lib.rknn_outputs_get(self.ctx,6,values,None),'outputs get')
        try:
            result=[]
            for a,v in zip(self.attrs,values):
                if v.size!=a.n_elems*4:raise ValueError('浮点输出长度错误')
                arr=np.ctypeslib.as_array(C.cast(v.buf,C.POINTER(C.c_float)),shape=(a.n_elems,)).copy()
                arr=arr.reshape(tuple(a.dims[:a.n_dims]))
                if a.fmt==1:arr=arr.transpose(0,3,1,2)
                result.append(arr)
            self.last_ms=(time.perf_counter()-start)*1000
            return result
        finally:self.check(self.lib.rknn_outputs_release(self.ctx,6,values),'outputs release')
    def predict(self,image,confidence=.25):
        rgb,ratio,left,top=letterbox(image)
        boxes=decode(self.run(rgb),confidence)
        if len(boxes):
            boxes[:,[0,2]]=(boxes[:,[0,2]]-left)/ratio
            boxes[:,[1,3]]=(boxes[:,[1,3]]-top)/ratio
            boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,image.shape[1])
            boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,image.shape[0])
        return boxes
    def close(self):
        if self.ctx.value:self.lib.rknn_destroy(self.ctx);self.ctx.value=0
