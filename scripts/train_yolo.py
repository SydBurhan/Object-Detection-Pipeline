import os
import mlflow
from ultralytics import YOLO

# ✅ Set MLflow tracking URI
if os.getenv("GITHUB_ACTIONS") == "true":
    mlflow.set_tracking_uri("file:./mlruns")  # GitHub Actions
else:
    mlflow.set_tracking_uri("file:///C:/Users/Syed Burhan Ahmad/Desktop/Academia/Other/Internship TNG/Object-Detection-Pipeline/mlruns")  # Local path

# ✅ Create or set experiment
mlflow.set_experiment("YOLOv8-Zyn-Training")

# ✅ Start MLflow run
with mlflow.start_run():
    # Parameters
    epochs = 10
    imgsz = 416
    batch = 4
    model_arch = "yolov8n.yaml"
    data_yaml = "data.yaml"

    # Log parameters
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("imgsz", imgsz)
    mlflow.log_param("batch", batch)
    mlflow.log_param("model", model_arch)

    # Load and train YOLO model
    model = YOLO(model_arch)
    results = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch)

    # Logging metrics
    try:
        metrics = results.metrics  # ultralytics>=8.1.0
        if metrics:
            mlflow.log_metric("metrics/mAP50", metrics.get("metrics/mAP50", 0))
            mlflow.log_metric("metrics/precision", metrics.get("metrics/precision", 0))
            mlflow.log_metric("metrics/recall", metrics.get("metrics/recall", 0))
        else:
            print("⚠️ Metrics not available.")
    except Exception as e:
        print(f"⚠️ Failed to log metrics: {e}")

    # Log trained model weights
    model_path = "runs/detect/train/weights/best.pt"
    if os.path.exists(model_path):
        mlflow.log_artifact(model_path)
    else:
        print(f"⚠️ Model weights not found at {model_path}")

    print("✅ Training complete and logged to MLflow.")
