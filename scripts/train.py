import os
import mlflow
import mlflow.tensorflow  # required for proper integration
from ultralytics import YOLO

# ✅ Set MLflow tracking URI based on environment (GitHub Actions or local)
if os.getenv("GITHUB_ACTIONS") == "true":
    mlflow.set_tracking_uri("file:./mlruns")  # Relative path for GitHub Actions
else:
    mlflow.set_tracking_uri("file:///C:/Users/Syed Burhan Ahmad/Desktop/Academia/Other/Internship TNG/Object-Detection-Pipeline/mlruns")

# ✅ Set the experiment name
mlflow.set_experiment("YOLOv8-Zyn-Training")

# ✅ Start MLflow run
with mlflow.start_run():
    # Log training configuration
    mlflow.log_param("epochs", 10)
    mlflow.log_param("imgsz", 416)
    mlflow.log_param("batch", 4)
    mlflow.log_param("model", "yolov8n.yaml")

    # Load YOLO model
    model = YOLO("yolov8n.yaml")

    # Train model
    results = model.train(
        data="/datasets/data/data.yaml",  # Ensure this path is correct
        epochs=10,
        imgsz=416,
        batch=4
    )

    # Log performance metrics
    metrics = results.results_dict
    mlflow.log_metric("metrics/mAP50", metrics.get("metrics/mAP50", 0))
    mlflow.log_metric("metrics/precision", metrics.get("metrics/precision", 0))
    mlflow.log_metric("metrics/recall", metrics.get("metrics/recall", 0))

    # Log trained model weights as artifact
    model_path = "runs/detect/train/weights/best.pt"
    if os.path.exists(model_path):
        mlflow.log_artifact(model_path)
    else:
        print(f"⚠️ Warning: Model weights not found at {model_path}")

    print("✅ Training complete and logged to MLflow")
