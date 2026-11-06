import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).parent
MODELS_DIR = BASE / "models"
CONF_THRESHOLD = 0.5

with open(BASE / "care_advice.json", encoding="utf-8") as f:
    ADVICE = json.load(f)


class Stage:
    """One model: loads real files if present, otherwise mocks."""

    def __init__(self, name, mock_labels):
        self.name = name
        self.mock_labels = mock_labels
        self.model = None
        self.labels = mock_labels
        self.input_size = 224

        folder = MODELS_DIR / name
        model_path = folder / "model.keras"
        labels_path = folder / "labels.json"

        if model_path.exists() and labels_path.exists():
            import tensorflow as tf  # imported only when needed
            self.model = tf.keras.models.load_model(model_path)
            cfg = json.loads(labels_path.read_text(encoding="utf-8"))
            self.labels = cfg["labels"]
            self.input_size = cfg.get("input_size", 224)

    @property
    def is_mock(self):
        return self.model is None

    def predict(self, img: Image.Image):
        if self.is_mock:
            return random.choice(self.labels), round(random.uniform(0.6, 0.95), 2)

        x = img.resize((self.input_size, self.input_size))
        x = np.asarray(x, dtype="float32") / 127.5 - 1.0  # MobileNetV2 style
        probs = self.model.predict(x[None, ...], verbose=0)[0]
        i = int(np.argmax(probs))
        return self.labels[i], round(float(probs[i]), 2)


plant_stage = Stage("plant", ["snake_plant", "aloe_vera", "money_plant"])
health_stage = Stage("health", ["healthy", "unhealthy"])
condition_stage = Stage("condition", ["yellowing", "brown_spots", "wilting"])


def get_advice(plant, condition):
    entry = ADVICE.get(plant, ADVICE["default"])
    if condition is None:
        return entry["healthy"]
    return entry["conditions"].get(condition, ADVICE["default"]["conditions"][condition])


def run_pipeline(img: Image.Image):
    plant, plant_conf = plant_stage.predict(img)
    health, health_conf = health_stage.predict(img)

    condition, condition_conf = None, None
    if health == "unhealthy":
        condition, condition_conf = condition_stage.predict(img)

    mocked = [s.name for s in (plant_stage, health_stage, condition_stage) if s.is_mock]

    if plant_conf < CONF_THRESHOLD or health_conf < CONF_THRESHOLD:
        return {
            "status": "uncertain",
            "message": "Not sure about this image. Try a clearer, closer photo.",
            "mocked_stages": mocked,
        }

    return {
        "status": "ok",
        "plant": {"name": plant, "confidence": plant_conf},
        "health": {"label": health, "confidence": health_conf},
        "condition": (
            {"name": condition, "confidence": condition_conf} if condition else None
        ),
        "advice": get_advice(plant, condition),
        "mocked_stages": mocked,
    }