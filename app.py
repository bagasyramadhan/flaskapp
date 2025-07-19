from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import json


app = Flask(__name__)

# Load label map
with open("model/label_mapping.json", "r") as f:
    label_map = json.load(f)

# Rebuild model architecture
num_classes = len(label_map) + 1  # +1 untuk background
model = fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load state_dict dari file
state_dict = torch.load("model/detection.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Transform input image
transform = T.Compose([
    T.ToTensor()
])

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    results = []
    for box, score, label in zip(outputs['boxes'], outputs['scores'], outputs['labels']):
        if score >= 0.5:  # default confidence threshold
            label_name = label_map.get(str(label.item()), "Unknown")
            results.append({
                'label': label_name,
                'score': round(float(score), 3),
                'box': [round(float(x), 2) for x in box]
            })

    return jsonify({'result': results})

if __name__ == "__main__":
    app.run(debug=True)
