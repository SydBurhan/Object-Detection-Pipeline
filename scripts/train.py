import mlflow
import mlflow.tensorflow  # still fine even if you’re using YOLO

from ultralytics import YOLO

# (1) Log to local folder (default)
mlflow.set_tracking_uri("file:///C:/Users/Syed Burhan Ahmad/Desktop/Academia/Other/Internship TNG/Object-Detection-Pipeline/mlruns")
mlflow.set_experiment("YOLOv8-Zyn-Training")

with mlflow.start_run():
    # (2) Log hyperparameters
    mlflow.log_param("epochs", 10)
    mlflow.log_param("imgsz", 416)
    mlflow.log_param("batch", 4)
    mlflow.log_param("model", "yolov8n.yaml")

    # (3) Load YOLO model and train
    model = YOLO("yolov8n.yaml")
    results = model.train(
        data="data.yaml",  # make sure this path is correct
        epochs=10,
        imgsz=416,
        batch=4
    )

    # (4) Log key metrics
    metrics = results.results_dict
    mlflow.log_metric("metrics/mAP50", metrics.get("metrics/mAP50", 0))
    mlflow.log_metric("metrics/precision", metrics.get("metrics/precision", 0))
    mlflow.log_metric("metrics/recall", metrics.get("metrics/recall", 0))

    # (5) Log the trained model weights
    model_path = "runs/detect/train/weights/best.pt"
    mlflow.log_artifact(model_path)

    print("✅ Training complete and logged locally.")
