
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


def show_metrics(X_test, y_test):

    model = joblib.load(
        "breast_cancer_model.joblib"
    )

    predictions = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Benign", "Malignant"]
        )
    )
