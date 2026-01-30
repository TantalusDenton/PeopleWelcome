"""ResNet50-based multi-label image classifier for per-AI tag prediction.

This module provides transfer learning using ResNet50 pretrained on ImageNet,
with custom classification heads for each AI's unique tag set.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add App-logic to path for database imports
APP_LOGIC_DIR = Path(__file__).resolve().parents[1] / "App-logic"
sys.path.insert(0, str(APP_LOGIC_DIR))

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Classifier will not work.")

# Model storage directory
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Image configuration
IMG_SIZE = 224
IMG_CHANNELS = 3
BATCH_SIZE = 32


class ResNetClassifier:
    """ResNet50-based multi-label classifier for image tagging."""

    def __init__(
        self,
        ai_id: str,
        model_dir: Optional[Path] = None,
        threshold: float = 0.5
    ):
        """
        Initialize classifier for a specific AI.

        Args:
            ai_id: Unique identifier for the AI
            model_dir: Directory to store model weights (default: models/)
            threshold: Prediction confidence threshold (default: 0.5)
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required for ResNetClassifier")

        self.ai_id = ai_id
        self.model_dir = model_dir or MODELS_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold

        self.model: Optional[Model] = None
        self.labels: List[str] = []
        self.label_to_idx: Dict[str, int] = {}

        # Paths for this AI's model
        self.weights_path = self.model_dir / ai_id / "resnet_weights.h5"
        self.config_path = self.model_dir / ai_id / "config.json"

        # Try to load existing model
        self._try_load_model()

    def _try_load_model(self) -> bool:
        """Try to load existing model weights and config."""
        if self.weights_path.exists() and self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                    self.labels = config.get("labels", [])
                    self.label_to_idx = {l: i for i, l in enumerate(self.labels)}

                if self.labels:
                    self._build_model(len(self.labels))
                    self.model.load_weights(str(self.weights_path))
                    logger.info(f"Loaded model for AI {self.ai_id} with {len(self.labels)} labels")
                    return True
            except Exception as e:
                logger.warning(f"Failed to load model for AI {self.ai_id}: {e}")

        return False

    def _build_model(self, num_labels: int) -> None:
        """
        Build ResNet50 model with custom classification head.

        Args:
            num_labels: Number of output labels for multi-label classification
        """
        # Load ResNet50 without the top classification layer
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
        )

        # Freeze base model layers for transfer learning
        base_model.trainable = False

        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.3)(x)
        # Sigmoid for multi-label classification
        outputs = Dense(num_labels, activation="sigmoid")(x)

        self.model = Model(inputs=base_model.input, outputs=outputs)

    def _save_model(self) -> str:
        """Save model weights and configuration."""
        if self.model is None:
            raise ValueError("No model to save")

        # Create AI-specific directory
        ai_model_dir = self.model_dir / self.ai_id
        ai_model_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        self.model.save_weights(str(self.weights_path))

        # Save configuration
        config = {
            "labels": self.labels,
            "threshold": self.threshold,
            "img_size": IMG_SIZE,
            "ai_id": self.ai_id
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved model for AI {self.ai_id} to {ai_model_dir}")
        return str(self.weights_path)

    def train(
        self,
        images: List[Tuple[str, np.ndarray]],
        tags: List[List[str]],
        epochs: int = 30,
        learning_rate: float = 1e-4,
        validation_split: float = 0.2,
        fine_tune: bool = False
    ) -> Dict[str, Any]:
        """
        Train the classifier on images with multi-label tags.

        Args:
            images: List of (image_id, image_array) tuples
            tags: List of tag lists corresponding to each image
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            validation_split: Fraction of data for validation
            fine_tune: Whether to fine-tune base layers after initial training

        Returns:
            Training history and metrics
        """
        if len(images) == 0:
            raise ValueError("No images provided for training")

        if len(images) != len(tags):
            raise ValueError("Number of images must match number of tag lists")

        # Collect all unique labels
        all_labels = set()
        for tag_list in tags:
            all_labels.update(tag_list)

        self.labels = sorted(list(all_labels))
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}
        num_labels = len(self.labels)

        logger.info(f"Training on {len(images)} images with {num_labels} unique labels")

        # Build model
        self._build_model(num_labels)

        # Prepare data
        X = np.array([img for _, img in images])
        y = self._encode_tags(tags)

        # Split into training and validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode="nearest"
        )

        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", self._macro_f1]
        )

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )
        ]

        # Train with frozen base
        logger.info("Phase 1: Training with frozen base layers...")
        history = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            epochs=epochs // 2,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        # Optional fine-tuning
        if fine_tune and len(X_train) >= 100:
            logger.info("Phase 2: Fine-tuning base layers...")

            # Unfreeze some layers
            for layer in self.model.layers[-30:]:
                layer.trainable = True

            # Recompile with lower learning rate
            self.model.compile(
                optimizer=Adam(learning_rate=learning_rate / 10),
                loss="binary_crossentropy",
                metrics=["accuracy", self._macro_f1]
            )

            history_fine = self.model.fit(
                train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                epochs=epochs // 2,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )

            # Merge histories
            for key in history.history:
                history.history[key].extend(history_fine.history[key])

        # Save model
        model_path = self._save_model()

        return {
            "status": "completed",
            "ai_id": self.ai_id,
            "num_images": len(images),
            "num_labels": num_labels,
            "labels": self.labels,
            "epochs_trained": len(history.history.get("loss", [])),
            "final_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history.get("val_loss", [0])[-1]),
            "model_path": model_path
        }

    def predict(
        self,
        image: np.ndarray,
        return_all: bool = False
    ) -> Dict[str, Any]:
        """
        Predict tags for a single image.

        Args:
            image: Image array (224x224x3)
            return_all: If True, return all tag scores; otherwise only above threshold

        Returns:
            Dictionary with predicted tags and confidence scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Ensure correct shape
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        # Normalize if needed
        if image.max() > 1.0:
            image = image / 255.0

        # Predict
        predictions = self.model.predict(image, verbose=0)[0]

        # Format results
        results = []
        for i, (label, score) in enumerate(zip(self.labels, predictions)):
            if return_all or score >= self.threshold:
                results.append({
                    "tag": label,
                    "confidence": float(score),
                    "above_threshold": score >= self.threshold
                })

        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "ai_id": self.ai_id,
            "predictions": results,
            "threshold": self.threshold,
            "total_labels": len(self.labels)
        }

    def predict_batch(
        self,
        images: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Predict tags for multiple images.

        Args:
            images: List of image arrays

        Returns:
            List of prediction dictionaries
        """
        return [self.predict(img) for img in images]

    def _encode_tags(self, tags: List[List[str]]) -> np.ndarray:
        """Convert tag lists to binary matrix."""
        num_samples = len(tags)
        num_labels = len(self.labels)
        y = np.zeros((num_samples, num_labels), dtype=np.float32)

        for i, tag_list in enumerate(tags):
            for tag in tag_list:
                if tag in self.label_to_idx:
                    y[i, self.label_to_idx[tag]] = 1.0

        return y

    @staticmethod
    def _macro_f1(y_true, y_pred):
        """Compute macro F1 score as a metric."""
        y_pred_binary = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
        tp = tf.reduce_sum(y_pred_binary * y_true, axis=0)
        fp = tf.reduce_sum(y_pred_binary * (1 - y_true), axis=0)
        fn = tf.reduce_sum((1 - y_pred_binary) * y_true, axis=0)

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        return tf.reduce_mean(f1)


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess an image for the classifier.

    Args:
        image_path: Path to image file

    Returns:
        Preprocessed image array
    """
    img = tf.keras.utils.load_img(
        image_path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    return img_array


def preprocess_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load and preprocess an image from bytes.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Preprocessed image array
    """
    img = tf.image.decode_image(image_bytes, channels=IMG_CHANNELS)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img_array = img.numpy() / 255.0
    return img_array


def get_classifier(ai_id: str) -> ResNetClassifier:
    """
    Get or create a classifier instance for an AI.

    Args:
        ai_id: AI identifier

    Returns:
        ResNetClassifier instance
    """
    return ResNetClassifier(ai_id)


def train_classifier_for_ai(
    ai_id: str,
    images_with_tags: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Train a classifier for an AI using its tagged images.

    Args:
        ai_id: AI identifier
        images_with_tags: List of dicts with 'image_path' and 'tags' keys

    Returns:
        Training results
    """
    classifier = get_classifier(ai_id)

    # Prepare data
    images = []
    tags = []
    for item in images_with_tags:
        try:
            img_array = preprocess_image(item["image_path"])
            images.append((item.get("image_id", ""), img_array))
            tags.append(item["tags"])
        except Exception as e:
            logger.warning(f"Failed to process image {item.get('image_path')}: {e}")

    if not images:
        return {"status": "failed", "error": "No valid images to train on"}

    return classifier.train(images, tags)


def run_inference(
    ai_id: str,
    image_path: str
) -> Dict[str, Any]:
    """
    Run inference on an image using an AI's trained model.

    Args:
        ai_id: AI identifier
        image_path: Path to image file

    Returns:
        Prediction results
    """
    classifier = get_classifier(ai_id)

    if classifier.model is None:
        return {"status": "error", "message": "No trained model found for this AI"}

    img_array = preprocess_image(image_path)
    return classifier.predict(img_array)
