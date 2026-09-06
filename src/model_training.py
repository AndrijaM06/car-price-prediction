import time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.data_preprocessing import build_preprocessor, split_features_and_target

DATA_PATH = "data/cars_features.csv"
MODELS_DIR = "models"

MODELS = {
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(random_state=42),
    "random_forest": RandomForestRegressor(random_state=42),
    "svm": SVR(),
}


def main() -> None:
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Splitting features and target...")
    X, y = split_features_and_target(df)
    print(X.shape, y.shape)

    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    for model_name, regressor in MODELS.items():
        print(f"\n--- Training: {model_name} ---")

        model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("regressor", regressor),
            ]
        )

        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start_time
        print(f"Training took {elapsed:.1f} seconds")

        model_path = f"{MODELS_DIR}/{model_name}_model.joblib"
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")

    print("\nAll models trained and saved.")

    print("\nMaking a few sample predictions with the linear regression model...")
    linear_model = joblib.load(f"{MODELS_DIR}/linear_regression_model.joblib")

    sample_X = X_test.sample(10, random_state=42)
    sample_y = y_test.loc[sample_X.index]
    sample_predictions = linear_model.predict(sample_X)

    prediction_preview = pd.DataFrame(
        {
            "actual_price_usd": sample_y.values,
            "predicted_price_usd": sample_predictions,
        }
    )
    print(prediction_preview)


if __name__ == "__main__":
    main()
