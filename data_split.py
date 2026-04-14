import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_prepare_data(
    file_path: str,
    target_col: str = "purchased",
    drop_cols: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
):
    """
    读取数据、做基础清理、拆分训练集和测试集。

    Parameters
    ----------
    file_path : str
        处理好的 user-level CSV 路径
    target_col : str
        目标列名
    drop_cols : list[str] | None
        不参与建模的列
    test_size : float
        测试集比例
    random_state : int
        随机种子
    stratify : bool
        是否按目标变量分层抽样

    Returns
    -------
    X_train, X_test, y_train, y_test, df
    """
    df = pd.read_csv(file_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    # 数据泄露或无关信息
    if drop_cols is None:
        drop_cols = ["user_id", "total_purchases", "purchase_rate"]

    # 只删除确实存在的列，避免报错
    actual_drop_cols = [col for col in drop_cols if col in df.columns]

    # 基础数据修正
    if "cart_rate" in df.columns:
        print("[ERROR] error in cart rate!")
        df["cart_rate"] = df["cart_rate"].clip(0, 1)

    X = df.drop(columns=[target_col] + actual_drop_cols)
    y = df[target_col]

    # 保证 train 和 test 里类别比例一样
    stratify_arg = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    return X_train, X_test, y_train, y_test, df


def print_split_summary(X_train, X_test, y_train, y_test):
    print("split was done as follow:")
    print("=" * 80)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape : {y_test.shape}")
    print()
    print(f"Train positive rate: {y_train.mean():.6f}")
    print(f"Test  positive rate: {y_test.mean():.6f}")
    print("\nFeature columns:")
    print(X_train.columns.tolist())
    