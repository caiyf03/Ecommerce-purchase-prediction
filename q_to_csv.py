import pandas as pd
import numpy as np


def safe_divide(a, b, eps=1e-6):
    return a / (b + eps)


def build_event_features_v3(parquet_path: str, output_path: str):
    print("=" * 80)
    print("LOAD DATA")
    print("=" * 80)

    df = pd.read_parquet(parquet_path)
    df = df.sort_values(["user_id", "event_time"]).copy()

    print(f"Input shape: {df.shape}")
    print("Columns:", df.columns.tolist())

    # =========================================================
    # 0. Basic cleanup
    # =========================================================
    df = df.drop_duplicates()
    df["category_level1"] = df["category_level1"].fillna("unknown")
    df["category_level2"] = df["category_level2"].fillna("unknown")
    df["category_level3"] = df["category_level3"].fillna("unknown")
    df["brand"] = df["brand"].fillna("unknown")
    df["user_session"] = df["user_session"].fillna("unknown_session")

    # =========================================================
    # 1. Base user-level features
    # =========================================================
    print("\n[1] BASE FEATURES")

    user_group = df.groupby("user_id")

    feat = user_group.agg(
        total_events=("event_time", "count"),
        num_products=("product_id", "nunique"),
        num_categories=("category_id", "nunique"),
        num_sessions=("user_session", "nunique"),
    ).reset_index()

    # =========================================================
    # 2. Time features
    # =========================================================
    print("\n[2] TIME FEATURES")

    first_event = user_group["event_time"].min()
    last_event = user_group["event_time"].max()

    time_base = pd.DataFrame({
        "user_id": first_event.index,
        "first_event_time": first_event.values,
        "last_event_time": last_event.values,
    })

    time_base["active_duration"] = (
        time_base["last_event_time"] - time_base["first_event_time"]
    ).dt.total_seconds()

    feat = feat.merge(
        time_base[["user_id", "active_duration"]],
        on="user_id",
        how="left",
    )

    df["prev_time"] = df.groupby("user_id")["event_time"].shift(1)
    df["delta_time"] = (df["event_time"] - df["prev_time"]).dt.total_seconds()

    delta_stats = df.groupby("user_id")["delta_time"].agg(
        mean_delta_time="mean",
        min_delta_time="min",
        max_delta_time="max",
        std_delta_time="std",
        median_delta_time="median",
    ).reset_index()

    feat = feat.merge(delta_stats, on="user_id", how="left")

    # =========================================================
    # 3. Funnel behavior features
    # =========================================================
    print("\n[3] FUNNEL FEATURES")

    pivot = df.pivot_table(
        index="user_id",
        columns="event_type",
        values="product_id",
        aggfunc="count",
        fill_value=0,
    ).reset_index()

    feat = feat.merge(pivot, on="user_id", how="left")

    for col in ["view", "cart", "purchase", "remove_from_cart"]:
        if col not in feat.columns:
            feat[col] = 0

    feat["view_to_cart_rate"] = safe_divide(feat["cart"], feat["view"])
    feat["cart_to_purchase_rate"] = safe_divide(feat["purchase"], feat["cart"])
    feat["remove_to_view_rate"] = safe_divide(feat["remove_from_cart"], feat["view"])
    feat["purchase_per_event"] = safe_divide(feat["purchase"], feat["total_events"])

    # =========================================================
    # 4. Session features
    # =========================================================
    print("\n[4] SESSION FEATURES")

    session_size = (
        df.groupby(["user_id", "user_session"])
        .size()
        .reset_index(name="session_len")
    )

    session_stats = session_size.groupby("user_id")["session_len"].agg(
        avg_session_len="mean",
        max_session_len="max",
        std_session_len="std",
        median_session_len="median",
    ).reset_index()

    feat = feat.merge(session_stats, on="user_id", how="left")

    # =========================================================
    # 5. Purchase timing features
    # =========================================================
    print("\n[5] PURCHASE TIMING FEATURES")

    purchase_df = df[df["event_type"] == "purchase"].copy()
    first_purchase = purchase_df.groupby("user_id")["event_time"].min()

    purchase_time = pd.DataFrame({
        "user_id": first_purchase.index,
        "first_purchase_time": first_purchase.values,
    })

    purchase_time = purchase_time.merge(
        time_base[["user_id", "first_event_time"]],
        on="user_id",
        how="left",
    )

    purchase_time["time_to_first_purchase"] = (
        purchase_time["first_purchase_time"] - purchase_time["first_event_time"]
    ).dt.total_seconds()

    purchase_time["fast_purchase"] = (
        purchase_time["time_to_first_purchase"] < 1800
    ).astype(int)

    feat = feat.merge(
        purchase_time[["user_id", "time_to_first_purchase", "fast_purchase"]],
        on="user_id",
        how="left",
    )

    # =========================================================
    # 6. Category distribution features
    # =========================================================
    print("\n[6] CATEGORY DISTRIBUTION FEATURES")

    top_cat = (
        df.groupby(["user_id", "category_level1"])
        .size()
        .reset_index(name="cnt")
        .sort_values(["user_id", "cnt"], ascending=[True, False])
        .drop_duplicates("user_id")
        .rename(columns={"category_level1": "top_category"})
    )

    feat = feat.merge(
        top_cat[["user_id", "top_category"]],
        on="user_id",
        how="left",
    )

    cat_cnt = (
        df.groupby(["user_id", "category_level1"])
        .size()
        .reset_index(name="cnt")
    )

    cat_total = cat_cnt.groupby("user_id")["cnt"].sum().reset_index(name="total")
    cat_cnt = cat_cnt.merge(cat_total, on="user_id", how="left")
    cat_cnt["p"] = cat_cnt["cnt"] / cat_cnt["total"]
    cat_cnt["entropy_component"] = -cat_cnt["p"] * np.log(cat_cnt["p"] + 1e-9)

    entropy = cat_cnt.groupby("user_id")["entropy_component"].sum().reset_index()
    entropy = entropy.rename(columns={"entropy_component": "entropy"})

    feat = feat.merge(entropy, on="user_id", how="left")

    # =========================================================
    # 7. Repeated interest features
    # =========================================================
    print("\n[7] INTEREST FEATURES")

    repeat_view = (
        df[df["event_type"] == "view"]
        .groupby(["user_id", "product_id"])
        .size()
        .reset_index(name="cnt")
    )

    repeat_stat = repeat_view.groupby("user_id")["cnt"].agg(
        avg_repeat_view="mean",
        max_repeat_view="max",
        median_repeat_view="median",
    ).reset_index()

    feat = feat.merge(repeat_stat, on="user_id", how="left")

    # =========================================================
    # 8. Price features
    # =========================================================
    print("\n[8] PRICE FEATURES")

    price_stats = df.groupby("user_id")["price"].agg(
        avg_price="mean",
        max_price="max",
        min_price="min",
        std_price="std",
        median_price="median",
    ).reset_index()

    feat = feat.merge(price_stats, on="user_id", how="left")

    # =========================================================
    # 9. Category-aware conversion features
    # =========================================================
    print("\n[9] CATEGORY-AWARE CONVERSION FEATURES")

    # ---- category-level conversion based on level1
    cat_view = (
        df[df["event_type"] == "view"]
        .groupby("category_level1")
        .size()
        .reset_index(name="cat_view_count")
    )

    cat_purchase = (
        df[df["event_type"] == "purchase"]
        .groupby("category_level1")
        .size()
        .reset_index(name="cat_purchase_count")
    )

    cat_conv = cat_view.merge(cat_purchase, on="category_level1", how="left")
    cat_conv["cat_purchase_count"] = cat_conv["cat_purchase_count"].fillna(0)
    cat_conv["cat_conversion_rate"] = safe_divide(
        cat_conv["cat_purchase_count"],
        cat_conv["cat_view_count"],
    )

    # 可选：平滑，避免小类别过度波动
    global_conv = safe_divide(
        len(df[df["event_type"] == "purchase"]),
        len(df[df["event_type"] == "view"]),
    )

    alpha = 20.0  # 平滑强度，可调
    cat_conv["cat_conversion_rate_smooth"] = (
        cat_conv["cat_purchase_count"] + alpha * global_conv
    ) / (cat_conv["cat_view_count"] + alpha)

    # 把 category conversion merge 回每条 event
    df = df.merge(
        cat_conv[[
            "category_level1",
            "cat_view_count",
            "cat_purchase_count",
            "cat_conversion_rate",
            "cat_conversion_rate_smooth",
        ]],
        on="category_level1",
        how="left",
    )

    # 仅使用 view 事件来衡量“看的类别是不是好买”
    view_df = df[df["event_type"] == "view"].copy()

    # 用户层面的 conversion-aware 聚合
    user_cat_conv = view_df.groupby("user_id").agg(
        user_avg_category_conversion=("cat_conversion_rate_smooth", "mean"),
        user_max_category_conversion=("cat_conversion_rate_smooth", "max"),
        user_min_category_conversion=("cat_conversion_rate_smooth", "min"),
        user_std_category_conversion=("cat_conversion_rate_smooth", "std"),
        user_avg_category_view_volume=("cat_view_count", "mean"),
    ).reset_index()

    feat = feat.merge(user_cat_conv, on="user_id", how="left")

    # top_category 对应的 conversion
    top_cat_with_conv = top_cat.merge(
        cat_conv[["category_level1", "cat_conversion_rate_smooth"]],
        left_on="top_category",
        right_on="category_level1",
        how="left",
    )[
        ["user_id", "cat_conversion_rate_smooth"]
    ].rename(columns={"cat_conversion_rate_smooth": "top_category_conversion"})

    feat = feat.merge(top_cat_with_conv, on="user_id", how="left")

    # 高转化类别占比：用户看的 event 中，有多少落在“高转化类别”
    high_conv_threshold = cat_conv["cat_conversion_rate_smooth"].quantile(0.75)
    cat_conv["is_high_conversion_category"] = (
        cat_conv["cat_conversion_rate_smooth"] >= high_conv_threshold
    ).astype(int)

    view_df = view_df.merge(
        cat_conv[["category_level1", "is_high_conversion_category"]],
        on="category_level1",
        how="left",
    )

    high_conv_ratio = view_df.groupby("user_id")["is_high_conversion_category"].mean().reset_index()
    high_conv_ratio = high_conv_ratio.rename(
        columns={"is_high_conversion_category": "high_conversion_category_ratio"}
    )

    feat = feat.merge(high_conv_ratio, on="user_id", how="left")

    # 类别级别 2 的覆盖深度（辅助）
    user_cat_l2 = (
        df.groupby(["user_id", "category_level2"])
        .size()
        .reset_index(name="cnt")
    )
    user_cat_l2_stats = user_cat_l2.groupby("user_id")["cnt"].agg(
        avg_l2_category_count="mean",
        max_l2_category_count="max",
    ).reset_index()

    feat = feat.merge(user_cat_l2_stats, on="user_id", how="left")

    # =========================================================
    # 10. Extra behavioral pattern features
    # =========================================================
    print("\n[10] EXTRA PATTERN FEATURES")

    # 零价格占比
    df["is_zero_price"] = (df["price"] == 0).astype(int)
    zero_price_ratio = df.groupby("user_id")["is_zero_price"].mean().reset_index()
    zero_price_ratio = zero_price_ratio.rename(columns={"is_zero_price": "zero_price_ratio"})
    feat = feat.merge(zero_price_ratio, on="user_id", how="left")

    # unknown 类别占比
    df["is_unknown_category"] = (df["category_level1"] == "unknown").astype(int)
    unknown_ratio = df.groupby("user_id")["is_unknown_category"].mean().reset_index()
    unknown_ratio = unknown_ratio.rename(columns={"is_unknown_category": "unknown_category_ratio"})
    feat = feat.merge(unknown_ratio, on="user_id", how="left")

    # =========================================================
    # 11. Final cleanup
    # =========================================================
    print("\n[11] FINAL CLEANUP")

    # top_category 保留字符串，后面可 one-hot；其余缺失用0
    if "top_category" in feat.columns:
        feat["top_category"] = feat["top_category"].fillna("unknown")

    numeric_cols = feat.select_dtypes(include=[np.number]).columns.tolist()
    feat[numeric_cols] = feat[numeric_cols].fillna(0)

    print("Final shape:", feat.shape)
    print("\nSample rows:")
    print(feat.head())

    feat.to_csv(output_path, index=False)
    print(f"\nSaved feature table to: {output_path}")


if __name__ == "__main__":
    build_event_features_v3(
        parquet_path=r"F:\CIS5450\event_level.parquet",
        output_path=r"F:\CIS5450\event_feature_table_v3.csv",
    )