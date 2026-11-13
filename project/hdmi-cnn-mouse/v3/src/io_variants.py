"""RKNN 2.5 I/O experiments; original application and model files stay unchanged.

Use one model object per worker thread. Returned arrays borrow persistent storage:
consume them before the next run() on that object, and never use them after close().
prealloc: Runtime float output -> preallocated NumPy storage, without .copy().
quantized: Runtime integer output -> preallocated storage -> NumPy dequantization.
dma: bound DMA input + float output, based on the vendor zero-copy example.
The dma input remains UINT8/pass_through=0: the model's mean/std is preserved.
It still copies RGB into DMA memory; it does NOT claim end-to-end zero-copy.
"""
import ctypes as C
import sys
import time

import numpy as np

from rknn_runtime import Attr, Input, Output, RKModel


class TensorMem(C.Structure):
    # RKNN Toolkit2 2.5.0 Linux rknn_api.h, struct rknn_tensor_memory.
    _fields_ = [
        ('virt_addr', C.c_void_p), ('phys_addr', C.c_uint64),
        ('fd', C.c_int32), ('offset', C.c_int32),
        ('size', C.c_uint32), ('flags', C.c_uint32),
        ('priv_data', C.c_void_p),
    ]


def clone_attr(attr):
    return Attr.from_buffer_copy(bytes(attr))


def logical_view(flat, attr):
    result = flat.reshape(tuple(attr.dims[:attr.n_dims]))
    if attr.n_dims == 4 and attr.fmt == 1:
        result = result.transpose(0, 3, 1, 2)
    return result


class OptimizedRKModel(RKModel):
    VARIANTS = ('prealloc', 'quantized', 'dma')
    _DTYPES = {
        0: np.float32, 1: np.float16, 2: np.int8, 3: np.uint8,
        4: np.int16, 5: np.uint16, 6: np.int32, 7: np.uint32,
        8: np.int64, 9: np.bool_,
    }

    def __init__(self, path, library, width, height, core_mask, variant='prealloc', flags=0):
        if variant not in self.VARIANTS:
            raise ValueError(f'Unknown IO variant: {variant}')
        # Async or manual-cache flags require a different buffer-lifetime protocol.
        if flags != 0:
            raise ValueError('IO experiment requires synchronous runtime with flags=0')
        self.variant = variant
        self._dma_mems = []
        self._flat_outputs = []
        self._raw_outputs = []
        self._views = []
        super().__init__(path, library, width, height, core_mask, flags=flags)
        try:
            self._input = Input(0, None, width * height * 3, 0, 3, 1)
            if variant == 'dma':
                self._prepare_dma()
            else:
                self._prepare_preallocated()
        except BaseException:
            self.close()
            raise

    def _prepare_preallocated(self):
        self._outputs = (Output * self.count)()
        for index, attr in enumerate(self.attrs):
            output = np.empty(int(attr.n_elems), dtype=np.float32)
            self._flat_outputs.append(output)
            self._views.append(logical_view(output, attr))
            if self.variant == 'prealloc':
                raw = output
                want_float = 1
            else:
                if attr.type not in self._DTYPES:
                    raise ValueError(f'Unsupported output dtype {attr.type}; no guessed conversion')
                raw = np.empty(int(attr.n_elems), dtype=self._DTYPES[attr.type])
                want_float = 0
                if attr.qnt_type not in (0, 1, 2):
                    raise ValueError(f'Unsupported quantization type {attr.qnt_type}')
                if attr.qnt_type == 2 and not (np.isfinite(attr.scale) and attr.scale > 0):
                    raise ValueError('Invalid affine output scale')
            self._raw_outputs.append(raw)
            self._outputs[index] = Output(want_float, 1, index, raw.ctypes.data, raw.nbytes)

    def _prepare_dma(self):
        signatures = {
            'rknn_create_mem': ([C.c_uint64, C.c_uint32], C.POINTER(TensorMem)),
            'rknn_destroy_mem': ([C.c_uint64, C.POINTER(TensorMem)], C.c_int),
            'rknn_set_io_mem': ([C.c_uint64, C.POINTER(TensorMem), C.POINTER(Attr)], C.c_int),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result

        # Use logical dimensions + input stride. The UINT8 conversion is the same
        # contract used by official rknn_dynshape_inference_zero_copy.cc.
        attr = clone_attr(self.input_attr)
        if attr.n_dims != 4:
            raise ValueError('DMA experiment only supports static batch-one image inputs')
        attr.n_dims = 4
        attr.dims[0], attr.dims[1], attr.dims[2], attr.dims[3] = 1, self.height, self.width, 3
        attr.type, attr.fmt, attr.pass_through = 3, 1, 0
        row_pixels = int(attr.w_stride) or self.width
        if row_pixels < self.width:
            raise ValueError(f'Unexpected DMA input w_stride: {row_pixels}')
        attr.h_stride = self.height
        input_bytes = self.height * row_pixels * 3
        memory = self._allocate(max(input_bytes, int(attr.size_with_stride)))
        self._dma_input_attr = attr
        self.check(self.lib.rknn_set_io_mem(self.ctx, memory, C.byref(attr)), 'set_io_mem input UINT8')
        self._dma_input_flat = np.ctypeslib.as_array(
            C.cast(memory.contents.virt_addr, C.POINTER(C.c_uint8)), shape=(input_bytes,))
        self._dma_input_flat.fill(0)
        self._dma_input = self._dma_input_flat.reshape(self.height, row_pixels, 3)[:, :self.width, :]
        self._dma_output_attrs = []
        for attr_original in self.attrs:
            attr = clone_attr(attr_original)
            # Logical tensor layout avoids guessing native NC1HWC2 padding.
            # Runtime converts/dequantizes to float, as in the official example.
            attr.type = 0
            expected_bytes = int(attr.n_elems) * 4
            memory = self._allocate(max(expected_bytes, int(attr.size_with_stride) * 4))
            self.check(self.lib.rknn_set_io_mem(self.ctx, memory, C.byref(attr)), 'set_io_mem float output')
            output = np.ctypeslib.as_array(
                C.cast(memory.contents.virt_addr, C.POINTER(C.c_float)), shape=(int(attr.n_elems),))
            self._flat_outputs.append(output)
            self._views.append(logical_view(output, attr))
            self._dma_output_attrs.append(attr)

    def _allocate(self, size):
        if size <= 0 or size > 0xffffffff:
            raise ValueError(f'Invalid DMA allocation size: {size}')
        memory = self.lib.rknn_create_mem(self.ctx, size)
        if not memory:
            raise MemoryError(f'rknn_create_mem({size}) failed')
        self._dma_mems.append(memory)
        if not memory.contents.virt_addr or memory.contents.size < size:
            raise MemoryError('RKNN returned invalid DMA memory')
        return memory

    def _dequantize(self):
        for attr, raw, output in zip(self.attrs, self._raw_outputs, self._flat_outputs):
            if attr.type in (0, 1) or attr.qnt_type == 0:
                np.copyto(output, raw, casting='unsafe')
            elif attr.qnt_type == 2:
                # Casting to float happens before subtraction: no int8 overflow.
                np.subtract(raw, np.float32(attr.zp), out=output, casting='unsafe')
                np.multiply(output, np.float32(attr.scale), out=output)
            elif attr.qnt_type == 1:
                np.multiply(raw, np.float32(2.0 ** (-int(attr.fl))), out=output, casting='unsafe')

    def run(self, rgb):
        if not self.ctx.value:
            raise RuntimeError('Model is closed')
        if rgb.dtype != np.uint8 or rgb.shape != (self.height, self.width, 3) or not rgb.flags.c_contiguous:
            raise ValueError('Expected contiguous RGB uint8 HWC image')
        start = time.perf_counter_ns()
        if self.variant == 'dma':
            np.copyto(self._dma_input, rgb)
        else:
            self._input.buf = rgb.ctypes.data
            self.check(self.lib.rknn_inputs_set(self.ctx, 1, C.byref(self._input)), 'inputs_set')
        run_start = time.perf_counter_ns()
        self.check(self.lib.rknn_run(self.ctx, None), 'run')
        run_end = time.perf_counter_ns()
        get_end = run_end
        convert_end = run_end
        if self.variant != 'dma':
            self.check(self.lib.rknn_outputs_get(self.ctx, self.count, self._outputs, None), 'outputs_get')
            get_end = time.perf_counter_ns()
            try:
                # is_prealloc=1 means release does not free our NumPy memory.
                # Check any unexpected pointer changes before dereferencing data.
                for output, raw in zip(self._outputs, self._raw_outputs):
                    if output.buf != raw.ctypes.data or output.size != raw.nbytes:
                        raise RuntimeError('Unexpected preallocated RKNN output address/size')
                if self.variant == 'quantized':
                    self._dequantize()
                convert_end = time.perf_counter_ns()
            finally:
                self.check(self.lib.rknn_outputs_release(self.ctx, self.count, self._outputs), 'outputs_release')
        # With flags=0, bound-memory cache synchronization is Runtime-managed.
        # Manual FLUSH-disable flags are intentionally not enabled in this test.
        end = time.perf_counter_ns()
        return self._views, {
            'call_ms': (end-start)/1e6,
            'rknn_run_ms': (run_end-run_start)/1e6,
            'input_ms': (run_start-start)/1e6,
            'outputs_get_ms': (get_end-run_end)/1e6,
            'dequantize_ms': (convert_end-get_end)/1e6,
            'release_ms': (end-convert_end)/1e6,
        }

    def close(self):
        # Views must not outlive their backing DMA allocations. Callers must not
        # retain a result after close(); retaining a result across run() mutates it.
        self._views = []
        self._flat_outputs = []
        self._raw_outputs = []
        self._dma_input = None
        self._dma_input_flat = None
        if getattr(self, 'ctx', None) is not None and self.ctx.value:
            for memory in reversed(self._dma_mems):
                code = self.lib.rknn_destroy_mem(self.ctx, memory)
                if code != 0:
                    print(f'WARNING: rknn_destroy_mem returned {code}', file=sys.stderr)
            self._dma_mems.clear()
            super().close()

