
import sys
import joblib
import pandas as pd

from breast_cancer import is_valid_input
from breast_cancer.features import FEATURES


values = list(map(float, sys.argv[1:]))


if len(values) != 30:

    print("Please enter 30 measurements")

else:

    radius_mean = values[0]
    texture_mean = values[1]
    perimeter_mean = values[2]
    area_mean = values[3]

    if not is_valid_input(
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean
    ):

        print("Invalid input")

    else:

        model = joblib.load(
            "breast_cancer_model.joblib"
        )

        data = pd.DataFrame(
            [values],
            columns=FEATURES
        )

        prediction = model.predict(data)[0]

        probabilities = model.predict_proba(data)[0]

        confidence = max(probabilities)

        if prediction == 1:
            result = "Malignant"
        else:
            result = "Benign"

        print("Prediction:", result)
        print("Confidence:", round(confidence, 4))
