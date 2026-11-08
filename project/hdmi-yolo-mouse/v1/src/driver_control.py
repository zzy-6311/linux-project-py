"""控制内核鼠标转接器：status/start/stop/inject。

本程序不读取物理鼠标事件，也不打开 /dev/hidg*。
start 退出后内核继续透传；inject 只按周期发相对位移指令。
"""
import argparse
import fcntl
import json
import math
import os
import struct
import sys
import time

DEVICE = "/dev/zzy_mouse_bridge"
STATS = struct.Struct("<7Ii6Q6q")
FIELDS = ("abi", "size", "connected", "configured", "enabled", "queued",
          "buttons", "last_error", "input_events", "submitted", "completed",
          "dropped", "injected", "usb_errors", "physical_x", "physical_y",
          "injected_x", "injected_y", "completed_x", "completed_y")


def request(direction, number, size):
    return (direction << 30) | (size << 16) | (ord("M") << 8) | number


def status(fd):
    data = bytearray(STATS.size)
    fcntl.ioctl(fd, request(2, 0, STATS.size), data, True)
    result = dict(zip(FIELDS, STATS.unpack(data)))
    if result["abi"] != 1 or result["size"] != STATS.size:
        raise RuntimeError("驱动 ABI 不匹配，需要内核转发版本 ABI 1。")
    return result


def set_enabled(fd, enabled):
    fcntl.ioctl(fd, request(1, 1, 4), struct.pack("<I", int(enabled)))


def inject_once(fd, dx, dy):
    fcntl.ioctl(fd, request(1, 2, 8), struct.pack("<ii", dx, dy))


def inject(fd, dx, dy, interval_ms, count):
    before = status(fd)
    if not all(before[k] for k in ("connected", "configured", "enabled")):
        raise RuntimeError("鼠标、USB 输出或透传未就绪；先执行 start。")
    sent = 0
    period = interval_ms / 1000
    deadline = time.monotonic() + period
    print(f"每 {interval_ms:g}ms 注入 X={dx:+d}, Y={dy:+d}，共 {count} 次；Ctrl+C 停止。", flush=True)
    try:
        for _ in range(count):
            time.sleep(max(0, deadline - time.monotonic()))
            inject_once(fd, dx, dy)
            sent += 1
            print(f"{sent}/{count} t={time.monotonic():.6f} 已提交位移指令", flush=True)
            # 若调度明显延迟，不追赶补发，避免突然成串移动。
            deadline = max(deadline + period, time.monotonic() + period)
    finally:
        print(f"注入停止：成功提交 {sent} 次，总量 X={sent*dx}, Y={sent*dy}；内核透传保持原状态。", flush=True)
        print(json.dumps(status(fd), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "start", "stop", "inject"))
    parser.add_argument("--dx", type=int, default=4)
    parser.add_argument("--dy", type=int, default=0)
    parser.add_argument("--interval-ms", type=float, default=200)
    parser.add_argument("--count", type=int, default=25)
    args = parser.parse_args()
    if not (-1000 <= args.dx <= 1000 and -1000 <= args.dy <= 1000):
        parser.error("本测试程序将每次位移限制在 -1000 到 1000")
    if not math.isfinite(args.interval_ms) or args.interval_ms < 10:
        parser.error("间隔至少 10ms 且必须为有限数值")
    if not 1 <= args.count <= 1000:
        parser.error("次数必须为 1 到 1000；不提供无限注入模式")
    fd = os.open(DEVICE, os.O_RDWR | os.O_CLOEXEC)
    try:
        status(fd)
        if args.action == "inject":
            inject(fd, args.dx, args.dy, args.interval_ms, args.count)
        else:
            if args.action in ("start", "stop"):
                set_enabled(fd, args.action == "start")
            print(json.dumps(status(fd), ensure_ascii=False, indent=2))
            if args.action == "start":
                print("内核透传已启用；本 Python 程序现在退出，鼠标仍可继续工作。")
    finally:
        os.close(fd)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("已停止 Python 控制程序；需要停止内核透传时另执行 stop。")
    except (OSError, RuntimeError) as error:
        print(f"控制失败：{error}", file=sys.stderr)
        sys.exit(1)
