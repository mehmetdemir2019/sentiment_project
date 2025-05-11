# src/evaluate.py
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def full_report(model, X_test, y_test, name: str = "Model",
                save_cm: bool = True,
                cm_path: str = "outputs/confusion_matrix.png"):
    """
    • Tahmin yapar
    • Accuracy, Precision, Recall, F1 basar
    • classification_report döker
    • Opsiyonel confusion matrix kaydeder
    """
    y_pred = model.predict(X_test)

    #— Temel metrikler
    acc = accuracy_score(y_test, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )

    print(f"\n{name} — Accuracy: {acc:.4f} | Precision: {prec:.4f} "
          f"| Recall: {recall:.4f} | F1: {f1:.4f}\n")
    print(classification_report(y_test, y_pred, digits=3))

    #— Confusion Matrix
    if save_cm:
        os.makedirs(os.path.dirname(cm_path), exist_ok=True)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix — {name}")
        plt.xlabel("Predicted"), plt.ylabel("True")
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()

    #— Pandas-friendly çıktı da döndürelim
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": recall,
        "f1": f1,
    }
