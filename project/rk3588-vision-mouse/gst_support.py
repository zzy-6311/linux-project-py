"""板载 GStreamer C API 与 HDMI 输入格式查询，不包含独立测试入口。"""
import ctypes as C
import ctypes.util
import re
import subprocess


class GError(C.Structure):
    _fields_ = [('domain', C.c_uint32), ('code', C.c_int), ('message', C.c_char_p)]


class Gst:
    def __init__(self):
        self.gst = C.CDLL(ctypes_library('gstreamer-1.0'))
        self.app = C.CDLL(ctypes_library('gstapp-1.0'))
        self.obj = C.CDLL(ctypes_library('gobject-2.0'))
        self.glib = C.CDLL(ctypes_library('glib-2.0'))
        self.init = self.bind(self.gst, 'gst_init', None, [C.c_void_p, C.c_void_p])
        self.parse = self.bind(self.gst, 'gst_parse_launch', C.c_void_p, [C.c_char_p, C.POINTER(C.POINTER(GError))])
        self.state = self.bind(self.gst, 'gst_element_set_state', C.c_int, [C.c_void_p, C.c_int])
        self.by_name = self.bind(self.gst, 'gst_bin_get_by_name', C.c_void_p, [C.c_void_p, C.c_char_p])
        self.bus = self.bind(self.gst, 'gst_element_get_bus', C.c_void_p, [C.c_void_p])
        self.pop_error = self.bind(self.gst, 'gst_bus_timed_pop_filtered', C.c_void_p, [C.c_void_p, C.c_uint64, C.c_uint])
        self.parse_error = self.bind(self.gst, 'gst_message_parse_error', None, [C.c_void_p, C.POINTER(C.POINTER(GError)), C.POINTER(C.c_void_p)])
        self.error_free = self.bind(self.glib, 'g_error_free', None, [C.POINTER(GError)])
        self.free = self.bind(self.glib, 'g_free', None, [C.c_void_p])
        self.unref = self.bind(self.gst, 'gst_object_unref', None, [C.c_void_p])
        self.mini_unref = self.bind(self.gst, 'gst_mini_object_unref', None, [C.c_void_p])
        self.alloc = self.bind(self.gst, 'gst_buffer_new_allocate', C.c_void_p, [C.c_void_p, C.c_size_t, C.c_void_p])
        self.fill = self.bind(self.gst, 'gst_buffer_fill', C.c_size_t, [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t])
        self.push = self.bind(self.app, 'gst_app_src_push_buffer', C.c_int, [C.c_void_p, C.c_void_p])
        # g_object_get is variadic: declare the fixed arguments and explicitly
        # pass pointer-typed remaining arguments below (important on AArch64).
        self.get = self.bind(self.obj, 'g_object_get', None, [C.c_void_p, C.c_char_p])
        self.get_u64 = self.bind(self.gst, 'gst_structure_get_uint64', C.c_int, [C.c_void_p, C.c_char_p, C.POINTER(C.c_uint64)])
        self.struct_free = self.bind(self.gst, 'gst_structure_free', None, [C.c_void_p])
        self.init(None, None)

    @staticmethod
    def bind(lib, name, result, args):
        f = getattr(lib, name)
        f.restype, f.argtypes = result, args
        return f

    def stats(self, sink):
        s = C.c_void_p()
        self.get(sink, b'stats', C.byref(s), C.c_void_p())
        result = {}
        if s:
            for name in ('rendered', 'dropped'):
                n = C.c_uint64()
                if self.get_u64(s, name.encode(), C.byref(n)):
                    result[name] = n.value
            self.struct_free(s)
        return result

    def check(self, bus):
        msg = self.pop_error(bus, 0, 2)  # GST_MESSAGE_ERROR
        if msg:
            err, debug = C.POINTER(GError)(), C.c_void_p()
            self.parse_error(msg, C.byref(err), C.byref(debug))
            message = err.contents.message.decode(errors='replace') if err else 'GStreamer error'
            detail = C.string_at(debug).decode(errors='replace') if debug else ''
            if err:
                self.error_free(err)
            if debug:
                self.free(debug)
            self.mini_unref(msg)
            raise RuntimeError(message + '\n' + detail)


def ctypes_library(name):
    lib = ctypes.util.find_library(name)
    if not lib:
        raise RuntimeError('Missing shared library: ' + name)
    return lib


def input_mode(device):
    text = subprocess.check_output(['v4l2-ctl', '-d', device, '--query-dv-timings'], text=True, timeout=8)
    def number(pattern):
        match = re.search(pattern, text)
        if not match:
            raise RuntimeError('Cannot identify HDMI signal: ' + text)
        return float(match[1])
    return (int(number(r'Active width:\s*(\d+)')), int(number(r'Active height:\s*(\d+)')),
            round(number(r'\(([\d.]+) frames per second\)'), 2))


def input_format(device):
    text = subprocess.check_output(['v4l2-ctl', '-d', device, '--get-fmt-video'], text=True, timeout=8)
    match = re.search(r"Pixel Format\s*:\s*'([^']+)'", text)
    formats = {'NV12': 'NV12', 'NV16': 'NV16', 'BGR3': 'BGR'}
    if not match or match[1] not in formats:
        raise RuntimeError('Unsupported input pixel format. Use SDR/8-bit output and inspect v4l2-ctl: ' + text)
    return formats[match[1]]

