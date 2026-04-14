# basic information of the F:\CIS5450\event_level.parquet
import pandas as pd
import numpy as np


def check_event_level_parquet(file_path: str):
    print("=" * 100)
    print("EVENT-LEVEL DATA CHECK")
    print("=" * 100)

    # =========================================================
    # 1. Read file
    # =========================================================
    print("\n[1] READING FILE")
    try:
        df = pd.read_parquet(file_path)
        print(f"Successfully loaded: {file_path}")
    except Exception as e:
        print(f"[FAIL] Cannot read parquet file: {e}")
        return

    print(f"Shape: {df.shape}")

    # =========================================================
    # 2. Basic info
    # =========================================================
    print("\n[2] BASIC INFO")
    print("Columns:")
    print(list(df.columns))

    print("\nHead:")
    print(df.head())

    print("\nDtypes:")
    print(df.dtypes)

    # =========================================================
    # 3. Required columns check
    # =========================================================
    print("\n[3] REQUIRED COLUMNS CHECK")
    required_cols = [
        "user_id",
        "event_time",
        "event_type",
        "product_id",
        "category_id",
        "category_code",
        "brand",
        "price",
        "user_session",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in required_cols]

    if len(missing_cols) == 0:
        print("[PASS] All required columns are present.")
    else:
        print("[FAIL] Missing required columns:")
        print(missing_cols)

    if len(extra_cols) > 0:
        print("\nExtra columns found:")
        print(extra_cols)

    # =========================================================
    # 4. Null check
    # =========================================================
    print("\n[4] NULL CHECK")
    null_counts = df.isnull().sum().sort_values(ascending=False)
    print(null_counts)

    # =========================================================
    # 5. event_time check
    # =========================================================
    print("\n[5] EVENT TIME CHECK")
    if "event_time" in df.columns:
        original_null_time = df["event_time"].isnull().sum()

        parsed_time = pd.to_datetime(df["event_time"], errors="coerce")
        parsed_null_time = parsed_time.isnull().sum()

        print(f"Original null event_time count: {original_null_time}")
        print(f"Null after pd.to_datetime: {parsed_null_time}")

        if parsed_null_time == 0:
            print("[PASS] event_time is fully parseable.")
        else:
            print("[WARN] Some event_time values could not be parsed.")

        df["event_time"] = parsed_time

        if parsed_null_time < len(df):
            print(f"Min event_time: {df['event_time'].min()}")
            print(f"Max event_time: {df['event_time'].max()}")
    else:
        print("[FAIL] event_time column missing.")

    # =========================================================
    # 6. event_type check
    # =========================================================
    print("\n[6] EVENT TYPE CHECK")
    allowed_event_types = {"view", "cart", "remove_from_cart", "purchase"}

    if "event_type" in df.columns:
        event_counts = df["event_type"].value_counts(dropna=False)
        print(event_counts)

        actual_event_types = set(df["event_type"].dropna().unique())
        invalid_event_types = actual_event_types - allowed_event_types

        if len(invalid_event_types) == 0:
            print("[PASS] All event_type values are valid.")
        else:
            print("[WARN] Invalid event_type values found:")
            print(invalid_event_types)
    else:
        print("[FAIL] event_type column missing.")

    # =========================================================
    # 7. price check
    # =========================================================
    print("\n[7] PRICE CHECK")
    if "price" in df.columns:
        print(df["price"].describe())

        negative_price_count = (df["price"] < 0).sum() if pd.api.types.is_numeric_dtype(df["price"]) else "N/A"
        zero_price_count = (df["price"] == 0).sum() if pd.api.types.is_numeric_dtype(df["price"]) else "N/A"

        print(f"Negative price count: {negative_price_count}")
        print(f"Zero price count: {zero_price_count}")

        if pd.api.types.is_numeric_dtype(df["price"]):
            if negative_price_count == 0:
                print("[PASS] No negative prices.")
            else:
                print("[WARN] Negative prices found.")
        else:
            print("[WARN] price is not numeric.")
    else:
        print("[FAIL] price column missing.")

    # =========================================================
    # 8. category check
    # =========================================================
    print("\n[8] CATEGORY CHECK")
    if "category_code" in df.columns:
        print(f"Null category_code count: {df['category_code'].isnull().sum()}")
        print(f"Unique category_code count: {df['category_code'].nunique(dropna=True)}")

        non_null_category = df["category_code"].dropna().astype(str)

        if len(non_null_category) > 0:
            print("\nSample category_code values:")
            print(non_null_category.head(10).tolist())

            hierarchical_ratio = non_null_category.str.contains(r"\.").mean()
            print(f"\nRatio containing '.' (hierarchical style): {hierarchical_ratio:.4f}")

            unknown_count = (non_null_category.str.lower() == "unknown").sum()
            print(f"Count of 'unknown': {unknown_count}")
        else:
            print("[WARN] category_code has no non-null values.")
    else:
        print("[FAIL] category_code column missing.")

    # Optional level columns check
    category_level_cols = ["category_level1", "category_level2", "category_level3"]
    existing_level_cols = [col for col in category_level_cols if col in df.columns]
    if len(existing_level_cols) > 0:
        print("\nCategory level columns found:")
        for col in existing_level_cols:
            print(f"{col}: {df[col].nunique(dropna=True)} unique values")
    else:
        print("\nNo explicit category_level1/2/3 columns found.")

    # =========================================================
    # 9. brand check
    # =========================================================
    print("\n[9] BRAND CHECK")
    if "brand" in df.columns:
        print(f"Null brand count: {df['brand'].isnull().sum()}")
        print(f"Unique brand count: {df['brand'].nunique(dropna=True)}")

        non_null_brand = df["brand"].dropna().astype(str)
        if len(non_null_brand) > 0:
            print("Sample brand values:")
            print(non_null_brand.head(10).tolist())
    else:
        print("[FAIL] brand column missing.")

    # =========================================================
    # 10. user/session scale check
    # =========================================================
    print("\n[10] USER / SESSION SCALE CHECK")
    if "user_id" in df.columns:
        num_users = df["user_id"].nunique()
        num_events = len(df)
        avg_events_per_user = num_events / num_users if num_users > 0 else np.nan

        print(f"Number of users: {num_users}")
        print(f"Number of events: {num_events}")
        print(f"Average events per user: {avg_events_per_user:.4f}")

    if "user_session" in df.columns:
        num_sessions = df["user_session"].nunique(dropna=True)
        avg_events_per_session = len(df) / num_sessions if num_sessions > 0 else np.nan
        print(f"Number of sessions: {num_sessions}")
        print(f"Average events per session: {avg_events_per_session:.4f}")

    # =========================================================
    # 11. Sorting check
    # =========================================================
    print("\n[11] SORTING CHECK")
    sorting_pass = None

    if "user_id" in df.columns and "event_time" in df.columns and df["event_time"].notnull().all():
        # Global sort check
        globally_sorted = df.sort_values(["user_id", "event_time"]).index.equals(df.index)
        print(f"Globally sorted by user_id + event_time: {globally_sorted}")

        # Sample user-level monotonic check
        user_counts = df["user_id"].value_counts()
        check_users = user_counts.head(10).index.tolist()

        bad_users = []
        for u in check_users:
            sub = df[df["user_id"] == u]
            if not sub["event_time"].is_monotonic_increasing:
                bad_users.append(u)

        if len(bad_users) == 0:
            print("[PASS] Sampled users are internally time-sorted.")
        else:
            print("[WARN] Some sampled users are not time-sorted:")
            print(bad_users)

        sorting_pass = globally_sorted and (len(bad_users) == 0)
    else:
        print("[WARN] Cannot perform sorting check due to missing user_id/event_time or invalid event_time.")
        sorting_pass = False

    # =========================================================
    # 12. Duplicate check
    # =========================================================
    print("\n[12] DUPLICATE CHECK")
    duplicate_rows = df.duplicated().sum()
    print(f"Exact duplicate row count: {duplicate_rows}")

    key_cols = [col for col in ["user_id", "event_time", "event_type", "product_id", "user_session"] if col in df.columns]
    if len(key_cols) > 0:
        duplicate_keys = df.duplicated(subset=key_cols).sum()
        print(f"Duplicate count on key columns {key_cols}: {duplicate_keys}")

    # =========================================================
    # 13. Quick purchase sanity check
    # =========================================================
    print("\n[13] PURCHASE SANITY CHECK")
    if "event_type" in df.columns:
        purchase_rows = (df["event_type"] == "purchase").sum()
        print(f"Number of purchase events: {purchase_rows}")

        if "user_id" in df.columns:
            purchase_users = df.loc[df["event_type"] == "purchase", "user_id"].nunique()
            print(f"Number of users with at least one purchase event: {purchase_users}")

    # =========================================================
    # 14. Final summary
    # =========================================================
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    all_required_present = len(missing_cols) == 0
    valid_time = "event_time" in df.columns and df["event_time"].isnull().sum() == 0
    valid_event_types = (
        "event_type" in df.columns and
        len(set(df["event_type"].dropna().unique()) - allowed_event_types) == 0
    )
    valid_price = (
        "price" in df.columns and
        pd.api.types.is_numeric_dtype(df["price"]) and
        (df["price"] < 0).sum() == 0
    )

    print(f"Required columns present : {all_required_present}")
    print(f"event_time parseable     : {valid_time}")
    print(f"event_type valid         : {valid_event_types}")
    print(f"price non-negative       : {valid_price}")
    print(f"sorted by user+time      : {sorting_pass}")

    if all([all_required_present, valid_time, valid_event_types, valid_price, sorting_pass]):
        print("\n[PASS] This event-level dataset looks structurally valid and ready for feature engineering.")
    else:
        print("\n[WARN] This dataset is partially valid, but some checks failed or need manual review.")


if __name__ == "__main__":
    file_path = "event_level.parquet"  # 改成你的文件路径
    check_event_level_parquet(file_path)