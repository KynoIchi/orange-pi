import onnxruntime as ort
import numpy as np
import cv2

# Hanya class person
class_names = ["person"]

def draw_yolo_detections(frame, preds, input_size=640, conf_thresh=0.4, nms_thresh=0.5):
    h, w, _ = frame.shape
    boxes, scores, class_ids = [], [], []

    for pred in preds:
        conf = pred[4]
        if conf < conf_thresh:
            continue
        class_conf = pred[5:]
        class_id = np.argmax(class_conf)
        if class_id != 0 or class_conf[class_id] < conf_thresh:
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
    person_count = len(indices)

    for i in indices:
        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        x, y, bw, bh = boxes[i]
        label = f"person: {scores[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"Person count: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def main():
    session = ort.InferenceSession("orange/yolov5n.onnx")
    cam_path = 'rtsp://admin:Damin1234@192.168.12.16:554/'  # or 0 for webcam

    fps = get_video_fps(cam_path)
    if fps is None:
        fps = 15  # fallback
    delay = max(1, int(1000 / fps))

    cap = cv2.VideoCapture(cam_path)  # Webcam, ganti cam_path jika RTSP
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = img.astype(np.float16)  # atau float32 jika model tidak pakai half

        preds = session.run(None, {'images': img})[0][0]
        frame = draw_yolo_detections(frame, preds)

        cv2.imshow("Person Detection", frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
