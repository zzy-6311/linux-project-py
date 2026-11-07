"""Aimlabs 训练场控制：Python 选择目标，内核驱动接收/发送鼠标报告。

BTN_SIDE（通常是后退侧键）由驱动保留，按一次开启，再按一次关闭。
每次启动默认 OFF；不是按键精灵，不使用 Windows 光标绝对坐标。
"""
import errno
import fcntl
import math
import os
import struct
import time

STATE = struct.Struct('<8I')
STATS = struct.Struct('<7Ii6Q6q')


def request(direction, number, size=0):
    return direction << 30 | size << 16 | ord('M') << 8 | number


def aim_delta(box, viewport, vfov, yaw, pitch):
    """透视投影像素 -> 相机转角 -> HID 相对计数，不能直接把像素当计数。"""
    vx, vy, width, height = viewport
    x = (box[0]+box[2])/2 - (vx+width/2)
    y = (box[1]+box[3])/2 - (vy+height/2)
    focal = height / (2*math.tan(math.radians(vfov)/2))
    ax = math.degrees(math.atan2(x, focal))
    ay = math.degrees(math.atan2(y, math.hypot(focal, x)))
    return round(ax/yaw), round(ay/pitch)


class AimControl:
    def __init__(self, viewport, *, vfov=70.5328004329168,
                 yaw=0.070, pitch=0.070, confidence=0.65, move_settle=0.030,
                 click_hold=0.010, post_click=0.040):
        if not 10 < vfov < 150 or not 0.001 <= abs(yaw) <= 5 or not 0.001 <= abs(pitch) <= 5:
            raise ValueError('FOV/鼠标灵敏度参数错误')
        self.viewport, self.vfov = viewport, vfov
        self.yaw, self.pitch, self.confidence = yaw, pitch, confidence
        self.move_settle = move_settle
        self.click_hold = click_hold
        self.post_click = post_click
        self.fd = os.open('/dev/zzy_mouse_bridge', os.O_RDWR | os.O_CLOEXEC)
        self.last_action = 'OFF'
        self.shots = 0
        self.excluded_circle = None
        self.exclusion_generation = None
        try:
            fcntl.ioctl(self.fd, request(1, 3, 4), struct.pack('<I', 1))
            self.state()
        except Exception:
            os.close(self.fd)
            raise

    def state(self):
        data = bytearray(STATE.size)
        fcntl.ioctl(self.fd, request(2, 4, STATE.size), data, True)
        values = STATE.unpack(data)
        if values[:2] != (1, STATE.size):
            raise RuntimeError('AI 驱动接口版本不匹配')
        return dict(zip(('abi','size','active','generation','synthetic_buttons','side_code','commands','clicks'),values))

    def stats(self):
        data = bytearray(STATS.size)
        fcntl.ioctl(self.fd, request(2, 0, STATS.size), data, True)
        values = STATS.unpack(data)
        if values[:2] != (1, STATS.size): raise RuntimeError('驱动状态版本不匹配')
        return dict(zip(('abi','size','connected','configured','enabled','queued','buttons','last_error',
                         'input_events','submitted','completed','dropped','injected','usb_errors',
                         'physical_x','physical_y','injected_x','injected_y','completed_x','completed_y'),values))

    def valid(self, generation):
        current = self.state()
        return bool(current['active'] and current['generation'] == generation)

    def send(self, generation, dx=0, dy=0, buttons=0):
        fcntl.ioctl(self.fd, request(1,5,16), struct.pack('<IIii', generation, buttons, dx, dy))

    def cancel(self):
        fcntl.ioctl(self.fd, request(0,6))
        self.last_action = 'OFF'

    def wait_ready(self, generation, stop, timeout=0.3):
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline and not stop.is_set():
            if not self.valid(generation): return False
            s=self.stats()
            if not all(s[k] for k in ('connected','configured','enabled')) or s['last_error']:
                return False
            if not s['queued'] and s['submitted']==s['completed']: return True
            stop.wait(0.0005)  # 仅在 USB 尚未完成时短暂让出 CPU，不添加游戏等待
        return False

    def act(self, boxes, snapshot, stop):
        """只在同一开启周期的新结果上动作。完成点击后才允许调用下一次识别。"""
        if not snapshot['active'] or snapshot['generation'] != self.exclusion_generation:
            # 手动移动/点击或重新切换开关后，旧屏幕坐标不再适用。
            self.excluded_circle = None
        if not snapshot['active'] or stop.is_set(): return False
        generation = snapshot['generation']
        if not self.valid(generation):
            self.last_action='new frame needed'
            return False
        vx,vy,w,h=self.viewport
        candidates=[b for b in boxes if b[4]>=self.confidence and
                    vx <= (b[0]+b[2])/2 < vx+w and vy <= (b[1]+b[3])/2 < vy+h]
        if self.excluded_circle is not None:
            cx,cy,radius=self.excluded_circle
            candidates=[b for b in candidates
                        if ((b[0]+b[2])/2-cx)**2+((b[1]+b[3])/2-cy)**2 > radius**2]
        if not candidates:
            self.last_action='no target outside last-shot circle' if self.excluded_circle else 'no target'
            return False
        target=min(candidates,key=lambda b:((b[0]+b[2])/2-vx-w/2)**2+((b[1]+b[3])/2-vy-h/2)**2)
        dx,dy=aim_delta(target,self.viewport,self.vfov,self.yaw,self.pitch)
        if abs(dx)>2000 or abs(dy)>2000:
            self.cancel()
            self.last_action='target exceeds movement limit'
            return False
        moved=False
        pressed=False
        action_start=time.monotonic()
        try:
            if not self.wait_ready(generation,stop): return False
            if dx or dy:
                self.send(generation,dx,dy)
                moved=True
                self.excluded_circle=None  # 视角再次变化，旧圆失效。
            if not self.wait_ready(generation,stop): return moved
            # USB 完成不等于游戏已经处理移动，保留短暂应用响应时间。
            if self.move_settle and stop.wait(self.move_settle): return moved
            if stop.is_set() or not self.valid(generation): return moved
            self.send(generation,buttons=1)
            pressed=True
            if not self.wait_ready(generation,stop): return True
            if self.click_hold: stop.wait(self.click_hold)
            self.send(generation,buttons=0)
            pressed=False
            if not self.wait_ready(generation,stop): return True
            self.shots+=1
            # FPS 相对移动后，刚瞄准的球预计来到准星附近，而非停留在原检测坐标。
            # 按一个球的半径外扩 25%，抵消检测框和鼠标计数取整误差。
            radius=max(12.0, max(target[2]-target[0],target[3]-target[1])*0.625)
            self.excluded_circle=(vx+w/2,vy+h/2,radius)
            self.exclusion_generation=generation
            elapsed=(time.monotonic()-action_start)*1000
            self.last_action=f'shot {self.shots} dx={dx} dy={dy} move_click_ms={elapsed:.2f}'
            print(f'AIM {self.last_action} target={target[:5]}',flush=True)
            if self.post_click: stop.wait(self.post_click)
            return True
        except OSError as exc:
            if exc.errno in (errno.ESTALE,errno.EACCES,errno.EBUSY):
                self.last_action='cancelled / manual input'
                return moved
            self.cancel()
            raise
        finally:
            if pressed:
                try:self.send(generation,buttons=0)
                except OSError:pass  # 驱动另有 120ms 自动释放，close 也释放

    def close(self):
        if self.fd is not None:
            try:self.cancel()
            finally:os.close(self.fd);self.fd=None
