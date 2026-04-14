import os
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import OrdinalEncoder

from data_split import load_and_prepare_data, print_split_summary


force_retrain = False


def build_xgboost(scale_pos_weight: float) -> XGBClassifier:
    """
    Build an XGBoost classifier.
    """
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    return model


def encode_categorical_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    use_categorical_features: bool,
):
    """
    Encode object/category columns for XGBoost if enabled.
    Uses OrdinalEncoder with unknown handling.
    """
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    print("\nCategorical columns found:", categorical_cols)

    if not use_categorical_features:
        if categorical_cols:
            print("Dropping categorical columns for XGBoost.")
            X_train = X_train.drop(columns=categorical_cols)
            X_test = X_test.drop(columns=categorical_cols, errors="ignore")
        return X_train, X_test, None

    if not categorical_cols:
        return X_train, X_test, None

    print("Encoding categorical columns for XGBoost with OrdinalEncoder.")

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[categorical_cols] = X_train[categorical_cols].fillna("unknown").astype(str)
    X_test[categorical_cols] = X_test[categorical_cols].fillna("unknown").astype(str)

    X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

    return X_train, X_test, encoder


def prepare_features(
    file_path: str,
    target_col: str = "purchased",
    use_event_features: bool = False,
    event_feature_path: str = r"F:\CIS5450\event_feature_table_v3.csv",
    use_categorical_features: bool = False,
):
    """
    Prepare train/test features for XGBoost.
    Supports:
    - base features
    - +event numeric features
    - +categorical/object features
    """
    X_train, X_test, y_train, y_test, df = load_and_prepare_data(
        file_path=file_path,
        target_col=target_col,
        drop_cols=["total_purchases", "purchase_rate", "total_events", "cart_rate"],
        test_size=0.2,
        random_state=42,
        stratify=True,
    )

    if use_event_features:
        print("\n" + "=" * 80)
        print("ADDING EVENT-LEVEL FEATURES")
        print("=" * 80)

        event_df = pd.read_csv(event_feature_path)
        assert "user_id" in event_df.columns, "event_feature_table must contain user_id"

        X_train = X_train.merge(event_df, on="user_id", how="left")
        X_test = X_test.merge(event_df, on="user_id", how="left")

        # remove clearly leakage features
        leakage_cols = [
            "purchase",
            "purchase_count",
            "purchase_per_event",
            "cart_to_purchase_rate",
            "time_to_first_purchase",
            "fast_purchase",
            # category conversion features from full-data statistics: remove for now
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
            "view",               # already have total_views
            "cart",               # already have total_carts
            "num_products_y",
            "num_categories_y",
            "avg_price_y",
            "max_price_y",
            "min_price_y",
            "total_events",       # redundant aggregate
        ]
        X_train = X_train.drop(columns=[c for c in duplicate_cols if c in X_train.columns])
        X_test = X_test.drop(columns=[c for c in duplicate_cols if c in X_test.columns])

        print(f"Merged train shape: {X_train.shape}")
        print(f"Merged test shape : {X_test.shape}")

    # drop user_id after merge
    X_train = X_train.drop(columns=["user_id"], errors="ignore")
    X_test = X_test.drop(columns=["user_id"], errors="ignore")

    # encode or drop object columns
    X_train, X_test, encoder = encode_categorical_features(
        X_train, X_test, use_categorical_features=use_categorical_features
    )

    print_split_summary(X_train, X_test, y_train, y_test)
    print("\nFeature columns:")
    print(f"Total features: {len(X_train.columns)}")

    return X_train, X_test, y_train, y_test, encoder


def get_or_train_model(
    X_train,
    y_train,
    model_path: str,
):
    """
    Load existing model if available; otherwise train and save.
    """
    y_train_np = np.array(y_train)

    num_pos = (y_train_np == 1).sum()
    num_neg = (y_train_np == 0).sum()
    scale_pos_weight = num_neg / max(num_pos, 1)

    print("\n" + "=" * 80)
    print("XGBOOST MODEL SETUP")
    print("=" * 80)
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")
    print(f"model_path: {model_path}")

    model = build_xgboost(scale_pos_weight=scale_pos_weight)

    if os.path.exists(model_path) and not force_retrain:
        print("\nFound existing model. Loading from disk...")
        model.load_model(model_path)
        print("Model loaded successfully.")
    else:
        print("\nTraining a new model...")
        model.fit(X_train, y_train_np)
        model.save_model(model_path)
        print(f"Model trained and saved to {model_path}")

    return model


def evaluate_model(model, X_test, y_test, threshold: float = 0.5):
    """
    Evaluate model performance.
    """
    y_test_np = np.array(y_test)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_test_np, y_pred)
    auc = roc_auc_score(y_test_np, y_prob)
    precision = precision_score(y_test_np, y_pred, zero_division=0)
    recall = recall_score(y_test_np, y_pred, zero_division=0)
    f1 = f1_score(y_test_np, y_pred, zero_division=0)

    print("\n" + "=" * 80)
    print("XGBOOST RESULTS")
    print("=" * 80)
    print(f"Threshold : {threshold:.2f}")
    print(f"Accuracy  : {acc:.6f}")
    print(f"AUC       : {auc:.6f}")
    print(f"Precision : {precision:.6f}")
    print(f"Recall    : {recall:.6f}")
    print(f"F1 Score  : {f1:.6f}")

    return {
        "accuracy": acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def main():
    file_path = r"F:\CIS5450\compressed_data.csv"
    target_col = "purchased"
    event_feature_path = r"F:\CIS5450\event_feature_table_v3.csv"

    # ====== choose stage here ======
    use_event_features = True
    use_categorical_features = True
    # stages:
    # 1) basic: use_event_features=False, use_categorical_features=False
    # 2) +event: use_event_features=True,  use_categorical_features=False
    # 3) +object: use_event_features=True,  use_categorical_features=True

    model_suffix = []
    model_suffix.append("event" if use_event_features else "basic")
    model_suffix.append("cat" if use_categorical_features else "nocat")

    model_path = f"xgb_model_{'_'.join(model_suffix)}.json"

    X_train, X_test, y_train, y_test, encoder = prepare_features(
        file_path=file_path,
        target_col=target_col,
        use_event_features=use_event_features,
        event_feature_path=event_feature_path,
        use_categorical_features=use_categorical_features,
    )

    model = get_or_train_model(
        X_train=X_train,
        y_train=y_train,
        model_path=model_path,
    )

    metrics = evaluate_model(model, X_test, y_test, threshold=0.5)

    # save predictions
    pred_df = pd.DataFrame({
        "actual": np.array(y_test),
        "pred_score": metrics["y_prob"],
        "pred_label": metrics["y_pred"],
    })
    pred_path = f"xgb_predictions_{'_'.join(model_suffix)}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to: {pred_path}")


if __name__ == "__main__":
    main()


# We observed that certain behavioral features, such as the number of cart actions (total_carts) and whether a user has ever added items to the cart (has_carted), exhibit strong predictive power for purchase behavior. In general, users who have engaged in cart-related actions are significantly more likely to make a purchase, making these features highly informative for the model.

# However, this relationship is not deterministic. According to our data analysis, approximately 41.6% of users who made a purchase did not have any recorded cart activity within the observation window (i.e., has_carted = 0 or total_carts = 0). This indicates that while cart-related behaviors are strong signals, they are not necessary conditions for a purchase to occur.

# Several factors may explain this phenomenon:

# Users may complete purchases through a “direct buy” flow without adding items to the cart
# Event logs may be incomplete or partially missing
# The aggregation window may not fully capture the entire user behavior sequence

# Therefore, these features should be interpreted as strongly correlated but not causal or exhaustive indicators of purchase intent. Models such as XGBoost can effectively leverage these signals through nonlinear feature interactions, but they do not rely on any single feature in isolation.

# In summary, while these behavioral features significantly improve model performance, their predictive strength stems from statistical correlation rather than strict causality, which also explains the presence of false positives despite high recall.