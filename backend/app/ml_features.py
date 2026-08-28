import pandas as pd

LANDCOVER_CATEGORIES = [
    "industrial",
    "forest",
    "farmland",
    "unknown",
]

FEATURE_COLUMNS = [
    "brightness",
    "confidence",
    "frp",
    "intelligence_score",
    "industrial_likelihood",
]


def normalize_landcover(value):
    if value in LANDCOVER_CATEGORIES:
        return value

    return "unknown"


def build_features(dataframe):
    dataframe = dataframe.copy()

    dataframe["landcover"] = dataframe[
        "landcover"
    ].apply(normalize_landcover)

    landcover_dummies = pd.get_dummies(
        dataframe["landcover"],
        prefix="landcover",
    )

    for category in LANDCOVER_CATEGORIES:
        column_name = f"landcover_{category}"

        if column_name not in landcover_dummies.columns:
            landcover_dummies[column_name] = 0

    features = pd.concat(
        [
            dataframe[FEATURE_COLUMNS],
            landcover_dummies[
                [
                    f"landcover_{category}"
                    for category in LANDCOVER_CATEGORIES
                ]
            ],
        ],
        axis=1,
    )

    return features