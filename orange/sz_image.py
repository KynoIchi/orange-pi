import onnxruntime as ort
import numpy as np
import onnx
import cv2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Load model
model_path = 'onnx/yolov8s.onnx'
onnx_model = onnx.load(model_path)
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

# Load image
img = cv2.imread('orange/Kabinet.jpg')

# Get ONNX input shape
input_shape = onnx_model.graph.input[0].type.tensor_type.shape.dim
input_height = input_shape[2].dim_value
input_width = input_shape[3].dim_value

print(f'Expected input size: {input_width}x{input_height}')

# Resize image
sz_img = cv2.resize(img, (input_width, input_height))

# Prepare input
blob = cv2.dnn.blobFromImage(sz_img, 1/255.0, (input_width, input_height), swapRB=True, crop=False)

# Inference
outputs = session.run(None, {input_name: blob})
predictions = outputs[0][0]

# Postprocess
boxes = []
confidences = []
class_ids = []
conf_thresh = 0.3
nms_thresh = 0.5

for pred in predictions:
    x, y, w, h = pred[:4]
    obj_conf = sigmoid(pred[4])
    class_scores = sigmoid(pred[5:])
    class_id = np.argmax(class_scores)
    class_conf = class_scores[class_id]
    conf = obj_conf * class_conf
    
    if conf > conf_thresh:
        # Convert center x,y,w,h to x1,y1
        x1 = int((x - w / 2))
        y1 = int((y - h / 2))
        boxes.append([x1, y1, int(w), int(h)])
        confidences.append(float(conf))
        class_ids.append(class_id)

# NMS
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)

# Draw
for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = f"ID:{class_ids[i]} {confidences[i]:.2f}"
    cv2.rectangle(sz_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(sz_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Show
cv2.imshow('YOLOv8 ONNX Detection', sz_img)
cv2.imwrite('orange/hasil_deteksi.jpg', sz_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
