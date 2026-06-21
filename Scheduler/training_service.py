"""Background training service for ResNet classifiers.

This module provides background training job processing for per-AI classifiers,
integrating with the database and S3 storage systems.
"""

import asyncio
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add paths for imports
BASE_DIR = Path(__file__).resolve().parents[1]
APP_LOGIC_DIR = BASE_DIR / "App-logic"
CLASSIFIER_DIR = BASE_DIR / "ImageClassifier"
sys.path.insert(0, str(APP_LOGIC_DIR))
sys.path.insert(0, str(CLASSIFIER_DIR))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import database services
import db_service
import s3_service

# Import classifier (with error handling)
try:
    from resnet_classifier import (
        ResNetClassifier,
        preprocess_image,
        preprocess_image_from_bytes,
        get_classifier,
    )
    CLASSIFIER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ResNet classifier not available: {e}")
    CLASSIFIER_AVAILABLE = False


class TrainingService:
    """Background service for processing AI training jobs."""

    def __init__(
        self,
        poll_interval: float = 5.0,
        max_workers: int = 1
    ):
        """
        Initialize the training service.

        Args:
            poll_interval: Seconds between polling for new jobs
            max_workers: Maximum concurrent training jobs (keep at 1 for GPU safety)
        """
        self.poll_interval = poll_interval
        self.max_workers = max_workers
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_job: Optional[Dict[str, Any]] = None

    def start(self) -> None:
        """Start the background training service."""
        if self._running:
            logger.warning("Training service already running")
            return

        if not CLASSIFIER_AVAILABLE:
            logger.error("Cannot start training service: classifier not available")
            return

        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        logger.info("Training service started")

    def stop(self) -> None:
        """Stop the background training service."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Training service stopped")

    def _worker_loop(self) -> None:
        """Main worker loop that polls for and processes training jobs."""
        while self._running:
            try:
                self._process_next_job()
            except Exception as e:
                logger.error(f"Error in training worker loop: {e}")

            time.sleep(self.poll_interval)

    def _process_next_job(self) -> None:
        """Process the next pending training job."""
        # Get pending jobs
        jobs = db_service.get_pending_training_jobs()

        if not jobs:
            return

        # Process the oldest pending job
        job = jobs[0]
        job_id = job["id"]
        ai_id = job["ai_id"]

        logger.info(f"Starting training job {job_id} for AI {ai_id}")
        self._current_job = job

        # Update job status
        db_service.update_training_job_status(job_id, "training")

        try:
            # Run training
            result = self._train_ai(ai_id)

            if result.get("status") == "completed":
                # Update AI's model path
                model_path = result.get("model_path")
                if model_path:
                    db_service.update_ai_model_path(ai_id, model_path)

                db_service.update_training_job_status(job_id, "completed")
                logger.info(f"Training job {job_id} completed successfully")
            else:
                error_msg = result.get("error", "Unknown error")
                db_service.update_training_job_status(job_id, "failed", error_msg)
                logger.error(f"Training job {job_id} failed: {error_msg}")

        except Exception as e:
            error_msg = str(e)
            db_service.update_training_job_status(job_id, "failed", error_msg)
            logger.error(f"Training job {job_id} failed with exception: {e}")

        finally:
            self._current_job = None

    def _train_ai(self, ai_id: str) -> Dict[str, Any]:
        """
        Train a classifier for a specific AI.

        Args:
            ai_id: AI identifier

        Returns:
            Training result dictionary
        """
        # Get all images and tags for this AI
        images = db_service.get_images_by_ai(ai_id, limit=10000)

        if not images:
            return {"status": "failed", "error": "No images found for AI"}

        # Prepare training data
        training_data = []

        for image in images:
            image_id = image["id"]
            s3_key = image["s3_key"]

            # Get tags for this image
            tags = db_service.get_tags_for_image_by_ai(image_id, ai_id)

            if not tags:
                continue  # Skip images without tags

            # Download image from S3
            try:
                image_data = self._download_image(s3_key)
                if image_data is not None:
                    training_data.append({
                        "image_id": image_id,
                        "image_data": image_data,
                        "tags": tags
                    })
            except Exception as e:
                logger.warning(f"Failed to download image {image_id}: {e}")
                continue

        if not training_data:
            return {"status": "failed", "error": "No valid training data (images with tags)"}

        logger.info(f"Training AI {ai_id} with {len(training_data)} images")

        # Create classifier and train
        classifier = get_classifier(ai_id)

        # Prepare data for classifier
        images_list = [
            (item["image_id"], preprocess_image_from_bytes(item["image_data"]))
            for item in training_data
        ]
        tags_list = [item["tags"] for item in training_data]

        # Train with fine-tuning if we have enough data
        fine_tune = len(images_list) >= 100

        result = classifier.train(
            images=images_list,
            tags=tags_list,
            epochs=30,
            fine_tune=fine_tune
        )

        return result

    def _download_image(self, s3_key: str) -> Optional[bytes]:
        """
        Download image from S3.

        Args:
            s3_key: S3 object key

        Returns:
            Image bytes or None if failed
        """
        try:
            import boto3
            s3_client = s3_service.get_s3_client()
            response = s3_client.get_object(
                Bucket=s3_service.AWS_BUCKET_NAME,
                Key=s3_key
            )
            return response["Body"].read()
        except Exception as e:
            logger.warning(f"Failed to download {s3_key}: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the training service."""
        return {
            "running": self._running,
            "current_job": self._current_job,
            "poll_interval": self.poll_interval
        }


# Singleton instance
_training_service: Optional[TrainingService] = None


def get_training_service() -> TrainingService:
    """Get or create the singleton training service instance."""
    global _training_service
    if _training_service is None:
        _training_service = TrainingService()
    return _training_service


def start_training_service() -> None:
    """Start the background training service."""
    service = get_training_service()
    service.start()


def stop_training_service() -> None:
    """Stop the background training service."""
    global _training_service
    if _training_service:
        _training_service.stop()


def queue_training_job(ai_id: str) -> Dict[str, Any]:
    """
    Queue a training job for an AI.

    Args:
        ai_id: AI identifier

    Returns:
        Job information
    """
    job = db_service.create_training_job(ai_id)
    return {"status": "queued", "job": job}


def run_inference_for_ai(ai_id: str, image_id: str) -> Dict[str, Any]:
    """
    Run inference on an image using an AI's trained model.

    Args:
        ai_id: AI identifier
        image_id: Image identifier

    Returns:
        Inference results
    """
    if not CLASSIFIER_AVAILABLE:
        return {"status": "error", "message": "Classifier not available"}

    # Get AI info
    ai = db_service.get_ai(ai_id)
    if not ai:
        return {"status": "error", "message": "AI not found"}

    if not ai.get("model_path"):
        return {"status": "error", "message": "AI has no trained model"}

    # Get image info
    image = db_service.get_image(image_id)
    if not image:
        return {"status": "error", "message": "Image not found"}

    # Download image
    try:
        s3_client = s3_service.get_s3_client()
        response = s3_client.get_object(
            Bucket=s3_service.AWS_BUCKET_NAME,
            Key=image["s3_key"]
        )
        image_bytes = response["Body"].read()
    except Exception as e:
        return {"status": "error", "message": f"Failed to download image: {e}"}

    # Run inference
    try:
        classifier = get_classifier(ai_id)
        if classifier.model is None:
            return {"status": "error", "message": "Model not loaded"}

        img_array = preprocess_image_from_bytes(image_bytes)
        result = classifier.predict(img_array, return_all=True)

        return {
            "status": "success",
            "ai_id": ai_id,
            "image_id": image_id,
            "predictions": result["predictions"]
        }
    except Exception as e:
        return {"status": "error", "message": f"Inference failed: {e}"}


# Auto-start service when module is imported (optional)
def auto_start():
    """Auto-start the training service."""
    if os.getenv("AUTO_START_TRAINING_SERVICE", "").lower() == "true":
        start_training_service()
