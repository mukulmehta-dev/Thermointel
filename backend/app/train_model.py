import os
import sys
import json
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
)

sys.path.insert(
    0,
    os.path.dirname(
        os.path.abspath(__file__)
    ),
)

import db
import ml_features


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models",
)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "thermal_classifier.joblib",
)

JSON_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "thermal_classifier.json",
)

LABEL_ENCODER_PATH = os.path.join(
    MODELS_DIR,
    "label_encoder.joblib",
)

CLASSES_PATH = os.path.join(
    MODELS_DIR,
    "classes.json",
)

MINIMUM_ROWS_REQUIRED = 30

MINIMUM_ROWS_PER_CLASS = 2


def load_training_dataframe():
    rows = db.get_all_events_for_training()

    if not rows:
        raise RuntimeError(
            "No events found in the database. "
            "Run the backend and let it collect "
            "data before training."
        )

    dataframe = pd.DataFrame(rows)

    dataframe["landcover"] = dataframe[
        "landcover"
    ].fillna("unknown")

    dataframe["landcover"] = dataframe[
        "landcover"
    ].apply(ml_features.normalize_landcover)

    print(
        "\nLandcover distribution in "
        "stored data:"
    )

    print(
        dataframe["landcover"].value_counts()
    )

    return dataframe


def build_features(dataframe):
    return ml_features.build_features(
        dataframe
    )


def train():
    print(
        "Loading events from database..."
    )

    dataframe = load_training_dataframe()

    total_rows = len(dataframe)

    print(
        f"Loaded {total_rows} stored events."
    )

    if total_rows < MINIMUM_ROWS_REQUIRED:
        print(
            "\nWARNING: Fewer than "
            f"{MINIMUM_ROWS_REQUIRED} events "
            "available. The model will train, "
            "but accuracy will be unreliable "
            "with this little data. Consider "
            "letting the backend run longer "
            "before relying on this model.\n"
        )

    class_counts = dataframe[
        "classification"
    ].value_counts()

    print(
        "\nClass distribution:"
    )

    print(class_counts)

    valid_classes = class_counts[
        class_counts >= MINIMUM_ROWS_PER_CLASS
    ].index

    dropped_classes = class_counts[
        class_counts < MINIMUM_ROWS_PER_CLASS
    ]

    if len(dropped_classes) > 0:
        print(
            "\nDropping classes with fewer "
            f"than {MINIMUM_ROWS_PER_CLASS} "
            "examples (not enough to learn "
            "from yet):"
        )

        print(dropped_classes)

    dataframe = dataframe[
        dataframe["classification"].isin(
            valid_classes
        )
    ]

    if dataframe["classification"].nunique() < 2:
        raise RuntimeError(
            "Not enough distinct classes with "
            "sufficient examples to train a "
            "classifier yet. Let the backend "
            "collect more varied data (more "
            "days, more locations) and try "
            "again."
        )

    features = build_features(dataframe)

    label_encoder = LabelEncoder()

    labels = label_encoder.fit_transform(
        dataframe["classification"]
    )

    can_stratify = (
        pd.Series(labels)
        .value_counts()
        .min()
        >= 2
    )

    (
        features_train,
        features_test,
        labels_train,
        labels_test,
    ) = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=(
            labels if can_stratify else None
        ),
    )

    print(
        "\nTraining XGBoost classifier..."
    )

    from sklearn.utils.class_weight import (
        compute_sample_weight,
    )

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=labels_train,
    )

    num_classes = len(
        label_encoder.classes_
    )

    eval_metric = (
        "logloss"
        if num_classes == 2
        else "mlogloss"
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric=eval_metric,
        random_state=42,
    )

    model.fit(
        features_train,
        labels_train,
        sample_weight=sample_weights,
    )

    predictions = model.predict(
        features_test
    )

    accuracy = accuracy_score(
        labels_test,
        predictions,
    )

    print(
        f"\nTest accuracy: {accuracy:.2%}"
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            labels_test,
            predictions,
            target_names=(
                label_encoder.classes_
            ),
            zero_division=0,
        )
    )

    feature_importances = sorted(
        zip(
            features.columns,
            model.feature_importances_,
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    print(
        "\nFeature importance "
        "(what the model relies on most):"
    )

    for name, importance in feature_importances:
        print(
            f"  {name}: {importance:.3f}"
        )

    os.makedirs(
        MODELS_DIR,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    try:
        model.get_booster().save_model(
            JSON_MODEL_PATH
        )
    except Exception as error:
        print(f"Warning: Failed to save booster JSON: {error}")

    joblib.dump(
        label_encoder,
        LABEL_ENCODER_PATH,
    )

    with open(CLASSES_PATH, "w", encoding="utf-8") as file:
        json.dump(list(label_encoder.classes_), file, indent=2)

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )

    print(
        "Label encoder saved to: "
        f"{LABEL_ENCODER_PATH}"
    )

    print(
        "Classes JSON saved to: "
        f"{CLASSES_PATH}"
    )

    print(
        "\nDone. Restart the backend to "
        "start using this trained model."
    )


if __name__ == "__main__":
    train()