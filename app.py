from __future__ import annotations

import io
import os
import base64
from typing import List, Optional

from flask import Flask, render_template, request, redirect, url_for, flash
from collections import OrderedDict

from PIL import Image

import torch
import torch.nn.functional as F
"""Avoid importing torchvision transforms that require NumPy.
We implement preprocessing with pure PIL + PyTorch to remove the NumPy dependency.
"""


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")


# Class labels inferred from dataset directories
CLASS_LABELS: List[str] = [
    "basophil",
    "eosinophil",
    "erythroblast",
    "ig",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
]


MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "cnn_model.pth"))
DEVICE = torch.device("cpu")

# Normalization configuration (default: disabled to match raw training)
NORMALIZE = os.environ.get("IMG_NORMALIZE", "false").lower() in {"1", "true", "yes"}
MEAN_ENV = os.environ.get("IMG_MEAN", "0.485,0.456,0.406")
STD_ENV = os.environ.get("IMG_STD", "0.229,0.224,0.225")
IMAGE_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGE_STD: list[float] = [0.229, 0.224, 0.225]
if NORMALIZE:
    try:
        IMAGE_MEAN = [float(x) for x in MEAN_ENV.split(",")]
        IMAGE_STD = [float(x) for x in STD_ENV.split(",")]
        if len(IMAGE_MEAN) != 3 or len(IMAGE_STD) != 3:
            raise ValueError
    except Exception:
        # Keep defaults if parsing fails
        pass


class CNN_Model(torch.nn.Module):
    """CNN architecture copied from training notebook (blood.ipynb).

    Assumes inputs are RGB images resized to 128x128.
    After three MaxPool(2,2) operations: 128->64->32->16, with 128 channels.
    """

    def __init__(self, num_classes: int):
        super(CNN_Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc1 = torch.nn.Linear(128 * 16 * 16, 256)
        self.fc2 = torch.nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))  # -> 64x64
        x = self.pool(torch.relu(self.conv2(x)))  # -> 32x32
        x = self.pool(torch.relu(self.conv3(x)))  # -> 16x16
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def load_model(model_path: str) -> Optional[torch.nn.Module]:
    """Attempt to load a PyTorch model stored at model_path.

    Tries torch.jit.load first; if that fails, falls back to torch.load.
    Returns None if loading fails.
    """
    if not os.path.exists(model_path):
        app.logger.error(f"Model file not found at: {model_path}")
        return None

    # Try TorchScript first (works without original Python class)
    try:
        model = torch.jit.load(model_path, map_location=DEVICE)
        model.eval()
        app.logger.info("Loaded model via torch.jit.load")
        return model
    except Exception as e:
        app.logger.warning(f"torch.jit.load failed: {e}")

    # Fallback to standard torch.load (may require original class definition)
    try:
        model = torch.load(model_path, map_location=DEVICE)
        # Some checkpoints store {'model': state_dict, ...}
        state_dict: Optional[OrderedDict] = None
        if isinstance(model, dict) and "model" in model and not isinstance(model["model"], torch.nn.Module):
            state_dict = model["model"]
        elif isinstance(model, dict) and "state_dict" in model:
            state_dict = model["state_dict"]
        elif isinstance(model, OrderedDict):
            state_dict = model

        if state_dict is not None:
            app.logger.info("Loaded state_dict; instantiating CNN_Model to load weights")
            instantiated = CNN_Model(num_classes=len(CLASS_LABELS))
            missing, unexpected = instantiated.load_state_dict(state_dict, strict=False)
            if missing:
                app.logger.warning(f"Missing keys when loading state_dict: {missing}")
            if unexpected:
                app.logger.warning(f"Unexpected keys when loading state_dict: {unexpected}")
            instantiated.eval()
            return instantiated

        if isinstance(model, torch.nn.Module):
            if hasattr(model, "eval"):
                model.eval()
            app.logger.info("Loaded model via torch.load")
            return model
        else:
            app.logger.error("Unsupported checkpoint format. Provide TorchScript or state_dict.")
            return None
    except Exception as e:
        app.logger.error(f"torch.load failed: {e}")
        return None


MODEL = load_model(MODEL_PATH)


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Preprocess a PIL image for model inference without NumPy.

    - Resize with PIL
    - Convert to RGB uint8 buffer
    - Convert to torch tensor in CHW layout scaled to [0, 1]
    """
    # Match training resolution in notebook (128x128 before pooling)
    target_size = (128, 128)
    img_resized = img.resize(target_size, Image.BILINEAR)

    # Ensure RGB
    if img_resized.mode != "RGB":
        img_resized = img_resized.convert("RGB")

    width, height = img_resized.size
    # Read raw bytes and construct a tensor (copy to avoid non-writable buffer warnings)
    byte_tensor = torch.tensor(bytearray(img_resized.tobytes()), dtype=torch.uint8)
    chw = byte_tensor.view(height, width, 3).permute(2, 0, 1).contiguous()
    tensor = chw.float().div(255.0).unsqueeze(0)  # [1, 3, H, W]
    # Optional normalization (disabled by default)
    if NORMALIZE:
        mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGE_STD, dtype=torch.float32).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
    return tensor


def image_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.route("/", methods=["GET"])  # Home page with upload form
def index():
    model_ready = MODEL is not None
    return render_template("index.html", model_ready=model_ready)


@app.route("/about", methods=["GET"])  # About page
def about():
    return render_template("about.html")


MAX_UPLOAD_MB = 10
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/predict", methods=["POST"])  # Handle image upload and prediction
def predict():
    if MODEL is None or not hasattr(MODEL, "__call__"):
        flash("Model is not runnable. Provide a TorchScript model or load with the original architecture (not just a state_dict).")
        return redirect(url_for("index"))

    if "image" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not _allowed_file(file.filename):
        flash("Unsupported file type. Please upload JPG or PNG.")
        return redirect(url_for("index"))

    # Limit size by reading at most MAX_UPLOAD_MB
    file.stream.seek(0, io.SEEK_END)
    size_bytes = file.stream.tell()
    file.stream.seek(0)
    if size_bytes > MAX_UPLOAD_MB * 1024 * 1024:
        flash(f"File too large. Max {MAX_UPLOAD_MB} MB.")
        return redirect(url_for("index"))

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        flash("Invalid image file.")
        return redirect(url_for("index"))

    input_tensor = preprocess_image(img).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(input_tensor)  # expected shape [1, num_classes]
        if not torch.is_tensor(outputs):
            flash("Model output is not a tensor. Please verify the model.")
            return redirect(url_for("index"))

        # Ensure 2D [batch, classes]
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)

        probs = F.softmax(outputs, dim=1).cpu().squeeze(0).tolist()

    # Align predictions to CLASS_LABELS length if mismatched
    num_outputs = len(probs)
    if num_outputs != len(CLASS_LABELS):
        app.logger.warning(
            f"Model num_classes ({num_outputs}) != labels ({len(CLASS_LABELS)}). Truncating or padding labels."
        )
        if num_outputs < len(CLASS_LABELS):
            labels = CLASS_LABELS[:num_outputs]
        else:
            # Extend with generic labels
            labels = CLASS_LABELS + [f"class_{i}" for i in range(len(CLASS_LABELS), num_outputs)]
    else:
        labels = CLASS_LABELS

    # Top prediction
    top_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    top_label = labels[top_idx]
    top_prob = float(probs[top_idx])

    # Prepare data for template
    preview_b64 = image_to_base64(img)
    predictions = list(zip(labels, [float(p) for p in probs]))
    predictions.sort(key=lambda x: x[1], reverse=True)

    return render_template(
        "result.html",
        predicted_label=top_label,
        predicted_confidence=top_prob,
        predictions=predictions,
        preview_b64=preview_b64,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)


