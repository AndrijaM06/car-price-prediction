import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET_COLUMN = "price_usd"

NUMERIC_FEATURES = [
    "year",
    "mileage_km",
    "volume_cm3",
    "car_age",
    "mileage_per_year",
    "engine_volume_liters",
    "is_newer_car",
    "is_high_mileage",
]

CATEGORICAL_FEATURES = [
    "make",
    "fuel_type",
    "color",
    "transmission",
    "drive_unit",
    "segment",
]

ORDINAL_FEATURES = [
    "condition",
]

CONDITION_ORDER = ["for parts", "with damage", "with mileage"]


def get_all_feature_columns() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES


def split_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[get_all_feature_columns()].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def _build_numeric_transformer() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return numeric_transformer


def _build_categorical_transformer() -> Pipeline:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return categorical_transformer


def _build_ordinal_transformer() -> Pipeline:
    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(categories=[CONDITION_ORDER])),
        ]
    )

    return ordinal_transformer


def build_preprocessor() -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", _build_numeric_transformer(), NUMERIC_FEATURES),
            ("cat", _build_categorical_transformer(), CATEGORICAL_FEATURES),
            ("ord", _build_ordinal_transformer(), ORDINAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor
