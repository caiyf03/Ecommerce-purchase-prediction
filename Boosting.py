import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from data_split import load_and_prepare_data, print_split_summary
force_retrain = False

def build_xgboost(scale_pos_weight: float) -> XGBClassifier:
    """
    创建一个 XGBoost 分类器实例
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


def get_or_train_model(X_train, y_train, model_path: str = "xgb_model.json"):
    """
    如果本地已有模型文件，则直接加载；
    否则训练一个新模型并保存。
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
        print("\nNo existing model found. Training a new model...")
        model.fit(X_train, y_train_np)
        model.save_model(model_path)
        print(f"Model trained and saved to {model_path}")

    return model


def evaluate_model(model, X_test, y_test, threshold: float = 0.5):
    """
    评估模型表现
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
    }


def main():
    file_path = "data/user_level_data.csv"
    target_col = "purchased"
    model_path = "xgb_model.json"

    X_train, X_test, y_train, y_test, df = load_and_prepare_data(
        file_path=file_path,
        target_col=target_col,
        drop_cols=["user_id", "total_purchases", "purchase_rate", "cart_rate"],
        test_size=0.2,
        random_state=42,
        stratify=True,
    )

    print_split_summary(X_train, X_test, y_train, y_test)

    print("\nFeature columns:")
    print(list(X_train.columns))

    model = get_or_train_model(
        X_train=X_train,
        y_train=y_train,
        model_path=model_path,
    )

    evaluate_model(model, X_test, y_test, threshold=0.5)


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