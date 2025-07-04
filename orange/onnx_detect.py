import onnxruntime as ort
import numpy as np
import cv2

# COCO class names (80 classes)
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

def draw_yolo_detections(frame, preds, class_names, input_size=640, conf_thresh=0.4, nms_thresh=0.5):
    h, w, _ = frame.shape
    boxes = []
    scores = []
    class_ids = []

    for pred in preds:
        conf = pred[4]
        if conf < conf_thresh:
            continue
        class_conf = pred[5:]
        class_id = np.argmax(class_conf)
        if class_conf[class_id] < conf_thresh:
            continue

        cx, cy, bw, bh = pred[:4]
        cx *= w / input_size
        cy *= h / input_size
        bw *= w / input_size
        bh *= h / input_size

        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        boxes.append([x1, y1, int(bw), int(bh)])
        scores.append(float(class_conf[class_id]))
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, nms_thresh)

    for i in indices:
        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        x, y, bw, bh = boxes[i]
        label = f"{class_names[class_ids[i]]}: {scores[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def main():
    session = ort.InferenceSession("orange/yolov5s.onnx")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (640, 640))  # JANGAN DIRUBAH
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = img.astype(np.float16)  # convert to float16 sesuai ekspektasi model

        outputs = session.run(None, {'images': img})
        preds = outputs[0][0]  # ambil batch pertama

        frame = draw_yolo_detections(frame, preds, class_names)

        cv2.imshow("YOLOv5s ONNX Runtime", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
