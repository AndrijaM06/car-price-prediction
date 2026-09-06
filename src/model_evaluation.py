import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data_preprocessing import split_features_and_target

DATA_PATH = "data/cars_features.csv"
MODELS_DIR = "models"
FINAL_MODEL_PATH = "models/car_price_model.joblib"

MODEL_FILES = {
    "Linear Regression": "linear_regression_model.joblib",
    "Decision Tree": "decision_tree_model.joblib",
    #"Random Forest": "random_forest_model.joblib",
    "SVM": "svm_model.joblib",
}


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "y_pred": y_pred,
    }


def main() -> None:
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Splitting features and target...")
    X, y = split_features_and_target(df)

    print("Creating the same train/test split as during training...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("\nEvaluating all trained models...\n")
    results = []
    predictions_by_model = {}

    for model_name, filename in MODEL_FILES.items():
        model_path = f"{MODELS_DIR}/{filename}"
        model = joblib.load(model_path)

        metrics = evaluate_model(model, X_test, y_test)
        predictions_by_model[model_name] = metrics["y_pred"]

        results.append(
            {
                "model": model_name,
                "mae": metrics["mae"],
                "mse": metrics["mse"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
            }
        )

        print(
            f"{model_name:20s} | MAE: {metrics['mae']:8.2f} | "
            f"RMSE: {metrics['rmse']:8.2f} | R2: {metrics['r2']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(by="mae", ascending=True)
    results_df = results_df.reset_index(drop=True)

    print("\nComparison table (sorted by MAE, best first):")
    print(results_df)

    # Biramo najbolji model kao onaj sa najmanjim MAE.
    best_model_name = results_df.iloc[0]["model"]
    best_model_filename = MODEL_FILES[best_model_name]
    print(f"\nBest model based on MAE: {best_model_name}")

    print("\nDetaljna analiza grešaka najboljeg modela...")
    best_predictions = predictions_by_model[best_model_name]

    prediction_analysis = pd.DataFrame(
        {
            "actual_price_usd": y_test.values,
            "predicted_price_usd": best_predictions,
        }
    )
    prediction_analysis["error_usd"] = (
        prediction_analysis["actual_price_usd"]
        - prediction_analysis["predicted_price_usd"]
    )
    prediction_analysis["absolute_error_usd"] = prediction_analysis["error_usd"].abs()

    print("\nPrimeri predikcija (10 nasumičnih redova):")
    print(prediction_analysis.sample(10, random_state=42))

    print("\n10 najvećih grešaka:")
    print(
        prediction_analysis.sort_values("absolute_error_usd", ascending=False).head(10)
    )

    # Čuvamo najbolji model kao finalni model projekta.
    print(f"\nSaving best model ({best_model_name}) as final model...")
    best_model = joblib.load(f"{MODELS_DIR}/{best_model_filename}")
    joblib.dump(best_model, FINAL_MODEL_PATH)
    print(f"Final model saved to: {FINAL_MODEL_PATH}")

    # Čuvamo i tabelu sa rezultatima poređenja, korisno za README i izveštaj.
    comparison_path = f"{MODELS_DIR}/model_comparison_results.csv"
    results_df.to_csv(comparison_path, index=False)
    print(f"Comparison results saved to: {comparison_path}")


if __name__ == "__main__":
    main()
