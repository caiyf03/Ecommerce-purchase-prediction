import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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


def run_linear_regression_baseline(
    file_path: str,
    target_col: str = "purchased",
    threshold: float = 0.5,
):
    # 1. 读取并拆分数据
    X_train, X_test, y_train, y_test, df = load_and_prepare_data(
        file_path=file_path,
        target_col=target_col,
        drop_cols=["user_id", "total_purchases", "purchase_rate", "total_events", "cart_rate"],
        test_size=0.2,
        random_state=42,
        stratify=True,
    )

    print_split_summary(X_train, X_test, y_train, y_test)

    print("\nFeature columns:")
    print(X_train.columns.tolist())

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )

    #  train
    print("\n" + "=" * 80)
    print("TRAINING LINEAR REGRESSION BASELINE")
    print("=" * 80)
    model.fit(X_train, y_train)

    #  predict
    y_pred_continuous = model.predict(X_test)

    # 线性回归输出不是概率，手动截断到 [0, 1] 便于解释
    y_pred_score = np.clip(y_pred_continuous, 0, 1)

    # 按阈值转成分类结果
    y_pred_label = (y_pred_score >= threshold).astype(int)

    # 5. 回归指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_score))
    mae = mean_absolute_error(y_test, y_pred_score)
    r2 = r2_score(y_test, y_pred_score)

    # 6. 分类指标
    acc = accuracy_score(y_test, y_pred_label)
    prec = precision_score(y_test, y_pred_label, zero_division=0)
    rec = recall_score(y_test, y_pred_label, zero_division=0)
    f1 = f1_score(y_test, y_pred_label, zero_division=0)

    # 虽然不是严格概率，但截断后的 score 仍可粗略看 ROC-AUC
    auc = roc_auc_score(y_test, y_pred_score)

    print("\n" + "=" * 80)
    print("LINEAR REGRESSION BASELINE RESULTS")
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

    # 7. 提取系数
    coefficients = model.named_steps["regressor"].coef_
    coef_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values(by="abs_coefficient", ascending=False)

    print("\n" + "=" * 80)
    print("TOP FEATURES BY ABSOLUTE COEFFICIENT")
    print("=" * 80)
    print(coef_df.head(15))

    # 8. 保存预测结果
    pred_df = X_test.copy()
    pred_df["actual"] = y_test.values
    pred_df["pred_score"] = y_pred_score
    pred_df["pred_label"] = y_pred_label

    pred_df.to_csv("linear_regression_predictions.csv", index=False)
    coef_df.to_csv("linear_regression_coefficients.csv", index=False)

    print("\nSaved files:")
    print("- linear_regression_predictions.csv")
    print("- linear_regression_coefficients.csv")

    # 9. 画图
    # 图1：预测分数分布
    plt.figure(figsize=(8, 5))
    plt.hist(y_pred_score, bins=50)
    plt.title("Distribution of Linear Regression Predicted Scores")
    plt.xlabel("Predicted score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # 图2：真实标签 vs 预测分数
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
    run_linear_regression_baseline(FILE_PATH, target_col="purchased", threshold=0.5)


# We observed that total_events is a linear combination of total_views, total_carts, and total_purchases. 
# Since total_purchases introduces target leakage and is removed, total_events becomes highly redundant with total_views and total_carts. 
# To improve interpretability and avoid multicollinearity, we keep total_views and total_carts and drop total_events.