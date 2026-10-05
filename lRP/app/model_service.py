import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "trained_model.joblib"
METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"


class ModelService:
    def __init__(self, pipeline, metadata: dict):
        self.pipeline = pipeline
        self.metadata = metadata

    @classmethod
    def load(cls):
        if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
            return None
        return cls(joblib.load(MODEL_PATH), json.loads(METADATA_PATH.read_text()))

    def _ordered_array(self, values: list[float]):
        features = self.metadata["features"]
        if len(values) != len(features):
            raise ValueError(f"Expected {len(features)} feature values")
        return pd.DataFrame(
            [np.asarray(values, dtype=float)],
            columns=self.metadata["feature_order"],
        )

    def predict(self, values: list[float]) -> int:
        return int(self.pipeline.predict(self._ordered_array(values))[0])

    def predict_probabilities(self, values: list[float]) -> dict[int, float]:
        probabilities = self.pipeline.predict_proba(self._ordered_array(values))[0]
        return {
            int(class_value): float(probability)
            for class_value, probability in zip(self.pipeline.classes_, probabilities)
        }