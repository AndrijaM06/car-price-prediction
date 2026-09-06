import pandas as pd

CLEANED_DATA_PATH = "data/cars_cleaned.csv"
FEATURES_DATA_PATH = "data/cars_features.csv"

REFERENCE_YEAR = 2026
HIGH_MILEAGE_THRESHOLD_KM = 300_000
NEWER_CAR_YEAR_THRESHOLD = 2015


def _add_car_age_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["car_age"] = REFERENCE_YEAR - df["year"]

    return df


def _add_mileage_per_year_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    safe_age = df["car_age"].clip(lower=1)
    df["mileage_per_year"] = df["mileage_km"] / safe_age

    return df


def _add_engine_volume_liters_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["engine_volume_liters"] = df["volume_cm3"] / 1000

    return df


def _add_is_newer_car_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_newer_car"] = (df["year"] >= NEWER_CAR_YEAR_THRESHOLD).astype(int)

    return df


def _add_is_high_mileage_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_high_mileage"] = (
        df["mileage_km"] > HIGH_MILEAGE_THRESHOLD_KM
    ).astype(int)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df_features = (
        df
        .pipe(_add_car_age_feature)
        .pipe(_add_mileage_per_year_feature)
        .pipe(_add_engine_volume_liters_feature)
        .pipe(_add_is_newer_car_feature)
        .pipe(_add_is_high_mileage_feature)
        .reset_index(drop=True)
    )

    return df_features


def main() -> None:
    print("Loading cleaned dataset...")
    df_cleaned = pd.read_csv(CLEANED_DATA_PATH)
    print(f"Cleaned dataset shape: {df_cleaned.shape}")

    print("Building features...")
    df_features = build_features(df_cleaned)
    print(f"Feature-engineered dataset shape: {df_features.shape}")

    print("Saving feature-engineered dataset...")
    df_features.to_csv(FEATURES_DATA_PATH, index=False)

    print(f"Feature-engineered dataset saved to: {FEATURES_DATA_PATH}")


if __name__ == "__main__":
    main()
