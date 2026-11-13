"""RKNN 2.5 C Runtime封装，复用已在本板验证的ctypes接口。无需安装Lite2。"""
import ctypes as C
import hashlib
from pathlib import Path
import sys
import time
import numpy as np

class Attr(C.Structure):
    _fields_ = [('index', C.c_uint32), ('n_dims', C.c_uint32),
                ('dims', C.c_uint32 * 16), ('name', C.c_char * 256),
                ('n_elems', C.c_uint32), ('size', C.c_uint32),
                ('fmt', C.c_int), ('type', C.c_int), ('qnt_type', C.c_int),
                ('fl', C.c_int8), ('zp', C.c_int32), ('scale', C.c_float),
                ('w_stride', C.c_uint32), ('size_with_stride', C.c_uint32),
                ('pass_through', C.c_uint8), ('h_stride', C.c_uint32)]


class Input(C.Structure):
    _fields_ = [('index', C.c_uint32), ('buf', C.c_void_p), ('size', C.c_uint32),
                ('pass_through', C.c_uint8), ('type', C.c_int), ('fmt', C.c_int)]


class Output(C.Structure):
    _fields_ = [('want_float', C.c_uint8), ('is_prealloc', C.c_uint8),
                ('index', C.c_uint32), ('buf', C.c_void_p), ('size', C.c_uint32)]


def attr_dict(a):
    return dict(index=int(a.index), name=bytes(a.name).decode(errors='replace'),
                shape=list(a.dims[:a.n_dims]), n_elems=int(a.n_elems),
                fmt=int(a.fmt), type=int(a.type), qnt_type=int(a.qnt_type),
                zero_point=int(a.zp), scale=float(a.scale))


class RKModel:
    def __init__(self, path, library, width, height, core_mask, flags=0):
        self.ctx = C.c_uint64(0)
        self.lib = C.CDLL(str(library))
        self.width, self.height = width, height
        signatures = {
            'rknn_init': ([C.POINTER(C.c_uint64), C.c_void_p, C.c_uint32, C.c_uint32, C.c_void_p], C.c_int),
            'rknn_query': ([C.c_uint64, C.c_int, C.c_void_p, C.c_uint32], C.c_int),
            'rknn_set_core_mask': ([C.c_uint64, C.c_int], C.c_int),
            'rknn_inputs_set': ([C.c_uint64, C.c_uint32, C.POINTER(Input)], C.c_int),
            'rknn_run': ([C.c_uint64, C.c_void_p], C.c_int),
            'rknn_outputs_get': ([C.c_uint64, C.c_uint32, C.POINTER(Output), C.c_void_p], C.c_int),
            'rknn_outputs_release': ([C.c_uint64, C.c_uint32, C.POINTER(Output)], C.c_int),
            'rknn_destroy': ([C.c_uint64], C.c_int),
        }
        for name, (args, result) in signatures.items():
            f = getattr(self.lib, name)
            f.argtypes, f.restype = args, result
        blob = Path(path).read_bytes()
        self.model_sha256 = hashlib.sha256(blob).hexdigest()
        buf = C.create_string_buffer(blob)
        try:
            self.check(self.lib.rknn_init(C.byref(self.ctx), buf, len(blob), flags, None), 'init')
            self.check(self.lib.rknn_set_core_mask(self.ctx, core_mask), 'set_core_mask')
            counts = (C.c_uint32 * 2)()
            self.query(0, counts)
            if counts[0] != 1:
                raise ValueError(f'Expected one image input, got {list(counts)}')
            self.count = int(counts[1])
            self.input_attr = Attr(index=0)
            self.query(1, self.input_attr)
            dims = list(self.input_attr.dims[:self.input_attr.n_dims])
            if dims not in ([1, height, width, 3], [1, 3, height, width]):
                raise ValueError(f'Model input {dims} disagrees with requested {width}x{height}')
            self.attrs = []
            for index in range(self.count):
                a = Attr(index=index)
                self.query(2, a)
                self.attrs.append(a)
            version = C.create_string_buffer(512)
            self.query(5, version)
            self.runtime_version = version.raw[:256].split(b'\0')[0].decode(errors='replace')
            self.driver_version = version.raw[256:].split(b'\0')[0].decode(errors='replace')
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(code, step):
        if code != 0:
            raise RuntimeError(f'RKNN {step} failed: {code}')

    def query(self, command, value):
        self.check(self.lib.rknn_query(self.ctx, command, C.byref(value), C.sizeof(value)), 'query')

    def run(self, rgb):
        if rgb.dtype != np.uint8 or rgb.shape != (self.height, self.width, 3) or not rgb.flags.c_contiguous:
            raise ValueError('Expected contiguous RGB uint8 HWC image')
        # UINT8=3, NHWC=1，pass_through=0 请求 Runtime 完成已配置的输入转换。
        item = Input(0, rgb.ctypes.data, rgb.nbytes, 0, 3, 1)
        values = (Output * self.count)(*(Output(1, 0, i, None, 0) for i in range(self.count)))
        start = time.perf_counter_ns()
        self.check(self.lib.rknn_inputs_set(self.ctx, 1, C.byref(item)), 'inputs_set')
        run_start = time.perf_counter_ns()
        self.check(self.lib.rknn_run(self.ctx, None), 'run')
        run_end = time.perf_counter_ns()
        # 仅成功取得输出后才 release；异常路径不释放尚未分配的输出。
        self.check(self.lib.rknn_outputs_get(self.ctx, self.count, values, None), 'outputs_get')
        try:
            result = []
            for a, value in zip(self.attrs, values):
                if value.size != a.n_elems * 4 or not value.buf:
                    raise ValueError('RKNN float output buffer size/pointer mismatch')
                arr = np.ctypeslib.as_array(C.cast(value.buf, C.POINTER(C.c_float)), shape=(a.n_elems,)).copy()
                arr = arr.reshape(tuple(a.dims[:a.n_dims]))
                # 只对四维NHWC特征图转置；RT的三维查询输出保留原有轴序。
                if a.n_dims == 4 and a.fmt == 1:
                    arr = arr.transpose(0, 3, 1, 2)
                result.append(arr)
        finally:
            self.check(self.lib.rknn_outputs_release(self.ctx, self.count, values), 'outputs_release')
        end = time.perf_counter_ns()
        return result, dict(call_ms=(end-start)/1e6, rknn_run_ms=(run_end-run_start)/1e6)

    def close(self):
        if self.ctx.value:
            code = self.lib.rknn_destroy(self.ctx)
            self.ctx.value = 0
            if code != 0:
                print(f'WARNING: rknn_destroy returned {code}', file=sys.stderr)


def snapshot():
    result = {}
    paths = [Path('/proc/loadavg'), Path('/sys/class/devfreq/dmc/governor'), Path('/sys/class/devfreq/dmc/cur_freq'), Path('/sys/class/devfreq/fdab0000.npu/governor'),
             Path('/sys/class/devfreq/fdab0000.npu/cur_freq'), Path('/sys/class/devfreq/fdab0000.npu/min_freq'),
             Path('/sys/class/devfreq/fdab0000.npu/max_freq'), Path('/sys/kernel/debug/rknpu/load'), Path('/sys/class/devfreq/fb000000.gpu/cur_freq')]
    paths += list(Path('/sys/class/thermal').glob('thermal_zone*/temp'))
    paths += list(Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq'))
    for p in paths:
        try:
            result[str(p)] = p.read_text().strip()
        except OSError as exc:
            result[str(p)] = str(exc)
    return result


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(mean_ms=float(a.mean()), median_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a, 95)), min_ms=float(a.min()), max_ms=float(a.max()),
                fps_reciprocal_mean=float(1000/a.mean()))

