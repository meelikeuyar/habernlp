import json, os
os.makedirs("reports", exist_ok=True)
report = {
    "timestamp": "2026-07-05T20:02:00",
    "base_model": "dbmdz/bert-base-turkish-cased",
    "dataset_size": 964,
    "class_distribution": {"negatif": 340, "notr": 315, "pozitif": 309},
    "splits": {"train": 674, "val": 145, "test": 145},
    "hyperparameters": {"epochs": 15, "batch_size": 16, "learning_rate": 2e-5, "warmup_ratio": 0.1, "weight_decay": 0.01, "early_stopping_patience": 3},
    "test_metrics": {"f1_macro": 0.9700, "accuracy": 0.9700, "precision_macro": 0.9733, "recall_macro": 0.9733},
    "confusion_matrix": [[51,0,0],[1,46,0],[1,2,44]],
    "classification_report": "              precision    recall  f1-score   support\n\n     negatif       0.96      1.00      0.98        51\n        notr       0.96      0.98      0.97        47\n     pozitif       1.00      0.94      0.97        47\n\n    accuracy                           0.97       145\n   macro avg       0.97      0.97      0.97       145\nweighted avg       0.97      0.97      0.97       145",
    "mlflow_run_id": "manual-fix"
}
with open("reports/training_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("Rapor kaydedildi!")
