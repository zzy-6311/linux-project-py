"""RTMDet-tiny RKNN 前后处理，仅依赖 NumPy/OpenCV。

ONNX/RKNN 输入为 RGB；导出时已等价交换首层卷积输入通道。
补边在右侧/底部。RKNN负责mean/std，调用方不能再除255。
六个输出依次为cls8,reg8,cls16,reg16,cls32,reg32，均NCHW。
cls是logits，reg已经乘过stride。网格中心偏移为0，不是YOLO的0.5。
"""
import cv2
import numpy as np

MEAN_RGB = [123.675, 116.28, 103.53]
STD_RGB = [58.395, 57.12, 57.375]
OUTPUT_NAMES = ["cls_s8", "reg_s8", "cls_s16", "reg_s16", "cls_s32", "reg_s32"]


def preprocess(image_bgr, width, height):
    """返回RGB uint8 HWC及还原坐标需要的缩放信息。"""
    oh, ow = image_bgr.shape[:2]
    scale = min(width / ow, height / oh)
    nw, nh = int(ow * scale + 0.5), int(oh * scale + 0.5)
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((height, width, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return rgb, dict(original_WH=[ow, oh], target_WH=[width, height], resized_WH=[nw, nh],
                     scale_xy=[nw / ow, nh / oh], left=0, top=0, pad=114, color="RGB")


def normalize_rgb(rgb):
    """仅用于PC ONNX/PyTorch参考；板端RKNN已在模型中处理归一化。"""
    tensor = (rgb.astype(np.float32) - np.asarray(MEAN_RGB, dtype=np.float32)) / np.asarray(STD_RGB, dtype=np.float32)
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])


def _nms(boxes, scores, iou_threshold=0.65, limit=300):
    areas = np.maximum(boxes[:, 2]-boxes[:, 0], 0) * np.maximum(boxes[:, 3]-boxes[:, 1], 0)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size and len(keep) < limit:
        current = int(order[0])
        keep.append(current)
        other = order[1:]
        x1 = np.maximum(boxes[current, 0], boxes[other, 0])
        y1 = np.maximum(boxes[current, 1], boxes[other, 1])
        x2 = np.minimum(boxes[current, 2], boxes[other, 2])
        y2 = np.minimum(boxes[current, 3], boxes[other, 3])
        intersection = np.maximum(x2-x1, 0) * np.maximum(y2-y1, 0)
        iou = intersection / np.maximum(areas[current] + areas[other] - intersection, 1e-12)
        order = other[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def decode(outputs, width, height, geometry, confidence=0.25):
    """六个NCHW原始输出 -> 原图坐标Nx5：[x1,y1,x2,y2,score]。"""
    if len(outputs) != 6:
        raise ValueError(f"需要6个输出，实际{len(outputs)}")
    all_boxes, all_scores = [], []
    for index, stride in enumerate((8, 16, 32)):
        cls = np.asarray(outputs[index*2], dtype=np.float32)
        reg = np.asarray(outputs[index*2+1], dtype=np.float32)
        fh, fw = height//stride, width//stride
        if cls.shape != (1, 1, fh, fw) or reg.shape != (1, 4, fh, fw):
            raise ValueError(f"stride{stride}输出shape错误：{cls.shape}, {reg.shape}")
        logits = cls[0, 0].reshape(-1)
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
        valid = np.flatnonzero(scores > confidence)
        if not valid.size:
            continue
        # nms_pre=30000；当前尺寸所有网格数都小于30000，不需截断。
        distances = reg[0].transpose(1, 2, 0).reshape(-1, 4)[valid]
        points_x = (valid % fw).astype(np.float32) * stride
        points_y = (valid // fw).astype(np.float32) * stride
        boxes = np.stack((points_x-distances[:, 0], points_y-distances[:, 1],
                          points_x+distances[:, 2], points_y+distances[:, 3]), axis=1)
        # 与官方Pad后img_shape裁剪一致；先裁剪到补边画布，再按真实缩放比还原。
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, width)
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, height)
        sx, sy = geometry["scale_xy"]
        boxes *= np.asarray([1/sx, 1/sy, 1/sx, 1/sy], dtype=np.float32)
        positive = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        all_boxes.append(boxes[positive])
        all_scores.append(scores[valid][positive])
    if not all_boxes:
        return np.empty((0, 5), dtype=np.float32)
    boxes, scores = np.concatenate(all_boxes), np.concatenate(all_scores)
    keep = _nms(boxes, scores)
    return np.column_stack((boxes[keep], scores[keep])).astype(np.float32)
