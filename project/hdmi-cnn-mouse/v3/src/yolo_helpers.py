"""Exact reusable functions from previously validated YOLO2.5 benchmark."""
import cv2
import numpy as np

def preprocess(image, width, height):
    h, w = image.shape[:2]
    ratio = min(width/w, height/h)
    rw, rh = max(1, round(w*ratio)), max(1, round(h*ratio))
    left, top = (width-rw)//2, (height-rh)//2
    canvas = np.full((height, width, 3), 114, np.uint8)
    canvas[top:top+rh, left:left+rw] = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    geometry = dict(original_WH=[w, h], target_WH=[width, height], resized_WH=[rw, rh],
                    left=left, top=top, scale_xy=[rw/w, rh/h], color='RGB', pad=114)
    return np.ascontiguousarray(canvas[:, :, ::-1]), geometry

def sigmoid(logits):
    return 1/(1+np.exp(-np.clip(logits, -80, 80)))

def decode_yolo(outputs, width, height, confidence):
    if len(outputs) != 6:
        raise ValueError(f'YOLO expected 6 raw outputs, got {len(outputs)}')
    rows = []
    for i, stride in enumerate((8, 16, 32)):
        gh, gw = height//stride, width//stride
        reg, logit = outputs[2*i:2*i+2]
        if reg.shape != (1, 4, gh, gw) or logit.shape != (1, 1, gh, gw):
            raise ValueError(f'Unexpected YOLO stride {stride} shapes: {reg.shape}, {logit.shape}')
        score = sigmoid(logit[0, 0])
        ys, xs = np.where(score >= confidence)
        if not len(xs):
            continue
        distance = reg[0, :, ys, xs]
        anchors = np.stack((xs+.5, ys+.5), axis=1)
        xyxy = np.concatenate((anchors-distance[:, :2], anchors+distance[:, 2:]), axis=1)*stride
        rows.append(np.column_stack((xyxy, score[ys, xs])))
    if not rows:
        return np.empty((0, 5), np.float32)
    result = np.concatenate(rows).astype(np.float32)
    return result[np.argsort(-result[:, 4])[:300]]

def restore(boxes, geometry):
    boxes = boxes.copy()
    if len(boxes):
        boxes[:, [0, 2]] = (boxes[:, [0, 2]]-geometry['left'])/geometry['scale_xy'][0]
        boxes[:, [1, 3]] = (boxes[:, [1, 3]]-geometry['top'])/geometry['scale_xy'][1]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, geometry['original_WH'][0])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, geometry['original_WH'][1])
    return boxes

def decode(outputs,width,height,geometry,confidence=.25):
    return restore(decode_yolo(outputs,width,height,confidence),geometry)
