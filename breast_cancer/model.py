
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

import joblib


def train_and_save_model(X, y):

    models = {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42)
        ),

        "DecisionTree": DecisionTreeClassifier(
            random_state=42
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    best_model = None
    best_name = ""
    best_score = 0

    for name, model in models.items():

        scores = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="accuracy"
        )

        score = scores.mean()

        print(name, "CV accuracy:", round(score, 4))

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    best_model.fit(X, y)

    joblib.dump(
        best_model,
        "breast_cancer_model.joblib"
    )

    print("Best model:", best_name)
    print("Best CV accuracy:", round(best_score, 4))

    return best_model
