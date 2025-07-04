import onnxruntime as ort
import numpy as np
import cv2

# Load model C:/Users/fadel/Downloads/BackUpSyarif/onnx/yolov5xu.onnx
model_path = "C:/Users/fadel/Downloads/BackUpSyarif/onnx/yolov5xu.onnx"
session = ort.InferenceSession(model_path)

# Prepare input
input_name = session.get_inputs()[0].name
img = cv2.imread('orange/Person.jpg')
h_img, w_img = img.shape[:2]

input_size = 640

input_width = 1088
input_height = 1920

blob = cv2.dnn.blobFromImage(img, 1/255.0, (input_width, input_height), swapRB=True, crop=False)
# ONNX Runtime expects (N,C,H,W)
outputs = session.run(None, {input_name: blob})

# === Inference ===
outputs = session.run(None, {input_name: blob})

# === Postprocess ===
predictions = outputs[0][0]  # remove batch dim → (num_preds, 85)

boxes = []
confidences = []
class_ids = []

conf_thresh = 0.3
nms_thresh = 0.5

for pred in predictions:
    x, y, w, h = pred[:4]
    obj_conf = pred[4]
    class_scores = pred[5:]
    class_id = np.argmax(class_scores)
    class_conf = class_scores[class_id]
    conf = obj_conf * class_conf
    
    if conf > conf_thresh:
        # YOLO bbox: center x,y,w,h → x1,y1,x2,y2 (scale ke ukuran asli)
        x1 = int((x - w / 2) * w_img / input_size)
        y1 = int((y - h / 2) * h_img / input_size)
        x2 = int((x + w / 2) * w_img / input_size)
        y2 = int((y + h / 2) * h_img / input_size)
        
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(float(conf))
        class_ids.append(class_id)

# Non-Max Suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)

# Draw results
for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = f"ID:{class_ids[i]} {confidences[i]:.2f}"
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Show + save
cv2.imshow('YOLO ONNX Deteksi', img)
cv2.imwrite('hasil_deteksi.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
