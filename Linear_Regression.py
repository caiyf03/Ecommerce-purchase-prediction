import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from data_split import load_and_prepare_data, print_split_summary


def build_preprocessor(
    X_train: pd.DataFrame,
    use_categorical_features: bool = True,
) -> ColumnTransformer:
    """
    Build preprocessing pipeline:
    - numeric: median impute + standardize
    - categorical: most frequent impute + one-hot encode (optional)
    """
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    print("\nNumeric features:", len(numeric_features))
    print("Categorical features:", len(categorical_features))
    if categorical_features:
        print("Categorical columns:", categorical_features)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = [
        ("num", numeric_transformer, numeric_features),
    ]

    if use_categorical_features and categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    return preprocessor


def run_linear_regression_baseline(
    file_path: str,
    target_col: str = "purchased",
    threshold: float = 0.5,
    load_model: bool = False,
    use_event_features: bool = False,
    event_feature_path: str = r"F:\CIS5450\event_feature_table_v3.csv",
    use_categorical_features: bool = True,   # ⭐ 新开关
):
    # 1. load and split
    X_train, X_test, y_train, y_test, df = load_and_prepare_data(
        file_path=file_path,
        target_col=target_col,
        drop_cols=["total_purchases", "purchase_rate", "total_events", "cart_rate"],
        test_size=0.2,
        random_state=42,
        stratify=True,
    )

    # 2. merge event features
    if use_event_features:
        print("\n" + "=" * 80)
        print("ADDING EVENT-LEVEL FEATURES")
        print("=" * 80)

        event_df = pd.read_csv(event_feature_path)
        assert "user_id" in event_df.columns, "event_feature_table must contain user_id"

        X_train = X_train.merge(event_df, on="user_id", how="left")
        X_test = X_test.merge(event_df, on="user_id", how="left")

        # remove leakage features
        leakage_cols = [
            "purchase",
            "purchase_count",
            "purchase_per_event",
            "cart_to_purchase_rate",
            "time_to_first_purchase",
            "fast_purchase",
            "user_avg_category_conversion",
            "user_max_category_conversion",
            "user_min_category_conversion",
            "user_std_category_conversion",
            "top_category_conversion",
            "high_conversion_category_ratio",
        ]
        X_train = X_train.drop(columns=[c for c in leakage_cols if c in X_train.columns])
        X_test = X_test.drop(columns=[c for c in leakage_cols if c in X_test.columns])

        # remove duplicated / highly redundant columns
        duplicate_cols = [
            "view",                 # already have total_views
            "cart",                 # already have total_carts
            "num_products_y",
            "num_categories_y",
            "avg_price_y",
            "max_price_y",
            "min_price_y",
            "total_events",         # redundant
        ]
        X_train = X_train.drop(columns=[c for c in duplicate_cols if c in X_train.columns])
        X_test = X_test.drop(columns=[c for c in duplicate_cols if c in X_test.columns])

        print(f"Merged train shape: {X_train.shape}")
        print(f"Merged test shape : {X_test.shape}")

    # 3. drop user_id after merge
    X_train = X_train.drop(columns=["user_id"], errors="ignore")
    X_test = X_test.drop(columns=["user_id"], errors="ignore")

    print_split_summary(X_train, X_test, y_train, y_test)

    print("\nFeature columns:")
    print(f"Total raw columns before preprocessing: {len(X_train.columns)}")

    # 4. preprocessing
    preprocessor = build_preprocessor(
        X_train=X_train,
        use_categorical_features=use_categorical_features,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    model_suffix = []
    model_suffix.append("event" if use_event_features else "basic")
    model_suffix.append("cat" if use_categorical_features else "nocat")
    model_name = f"linear_regression_{'_'.join(model_suffix)}.pkl"

    if load_model:
        print("\n" + "=" * 80)
        print("LOADING EXISTING MODEL ...")
        print("=" * 80)
        model = joblib.load(model_name)
    else:
        print("\n" + "=" * 80)
        print("TRAINING ...")
        print("=" * 80)
        model.fit(X_train, y_train)
        joblib.dump(model, model_name)
        print(f"\nModel saved as {model_name}")

    # 5. predict
    y_pred_continuous = model.predict(X_test)
    y_pred_score = np.clip(y_pred_continuous, 0, 1)
    y_pred_label = (y_pred_score >= threshold).astype(int)

    # 6. metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_score))
    mae = mean_absolute_error(y_test, y_pred_score)
    r2 = r2_score(y_test, y_pred_score)

    acc = accuracy_score(y_test, y_pred_label)
    prec = precision_score(y_test, y_pred_label, zero_division=0)
    rec = recall_score(y_test, y_pred_label, zero_division=0)
    f1 = f1_score(y_test, y_pred_label, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_score)

    print("\n" + "=" * 80)
    print("LINEAR REGRESSION RESULTS")
    print("=" * 80)
    print(f"Threshold : {threshold}")
    print()
    print("[Regression-style metrics]")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"R²  : {r2:.6f}")
    print()
    print("[Classification metrics]")
    print(f"Accuracy : {acc:.6f}")
    print(f"Precision: {prec:.6f}")
    print(f"Recall   : {rec:.6f}")
    print(f"F1-score : {f1:.6f}")
    print(f"ROC-AUC  : {auc:.6f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_label, digits=6))

    cm = confusion_matrix(y_test, y_pred_label)
    print("\nConfusion Matrix:")
    print(cm)

    # 7. coefficients
    coef_df = None
    try:
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        coefficients = model.named_steps["regressor"].coef_

        coef_df = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "abs_coefficient": np.abs(coefficients),
            }
        ).sort_values(by="abs_coefficient", ascending=False)

        print("\n" + "=" * 80)
        print("TOP FEATURES BY ABSOLUTE COEFFICIENT")
        print("=" * 80)
        print(coef_df.head(20))
    except Exception as e:
        print("\nCould not extract coefficient table:", e)

    # 8. save outputs
    pred_df = X_test.copy()
    pred_df["actual"] = y_test.values
    pred_df["pred_score"] = y_pred_score
    pred_df["pred_label"] = y_pred_label

    pred_path = f"lr_predictions_{'_'.join(model_suffix)}.csv"
    coef_path = f"lr_coefficients_{'_'.join(model_suffix)}.csv"

    pred_df.to_csv(pred_path, index=False)
    if coef_df is not None:
        coef_df.to_csv(coef_path, index=False)

    print("\nSaved files:")
    print(f"- {pred_path}")
    if coef_df is not None:
        print(f"- {coef_path}")

    # 9. plots
    plt.figure(figsize=(8, 5))
    plt.hist(y_pred_score, bins=50)
    plt.title("Distribution of Linear Regression Predicted Scores")
    plt.xlabel("Predicted score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(np.arange(len(y_test[:2000])), y_test[:2000], label="Actual", alpha=0.6, s=10)
    plt.scatter(np.arange(len(y_pred_score[:2000])), y_pred_score[:2000], label="Predicted score", alpha=0.6, s=10)
    plt.title("Actual Labels vs Predicted Scores (First 2000 Samples)")
    plt.xlabel("Sample index")
    plt.ylabel("Value / Score")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "model": model,
        "coef_df": coef_df,
        "pred_df": pred_df,
        "metrics": {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
        },
    }


if __name__ == "__main__":
    FILE_PATH = r"F:\CIS5450\compressed_data.csv"

    run_linear_regression_baseline(
        FILE_PATH,
        target_col="purchased",
        threshold=0.5,
        load_model=False,
        use_event_features=True,
        event_feature_path=r"F:\CIS5450\event_feature_table_v3.csv",
        use_categorical_features=True, 
    )