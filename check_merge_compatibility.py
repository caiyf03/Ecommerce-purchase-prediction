import pandas as pd


def check_merge_compatibility(
    user_level_path: str,
    event_feature_path: str,
    user_key: str = "user_id",
):
    print("=" * 100)
    print("MERGE COMPATIBILITY CHECK")
    print("=" * 100)

    # 1. Read files
    print("\n[1] READING FILES")
    user_df = pd.read_csv(user_level_path)
    if event_feature_path.endswith(".parquet"):
        event_df = pd.read_parquet(event_feature_path)
    else:
        event_df = pd.read_csv(event_feature_path)

    print(f"user_df shape : {user_df.shape}")
    print(f"event_df shape: {event_df.shape}")

    # 2. Column existence
    print("\n[2] KEY COLUMN CHECK")
    if user_key not in user_df.columns:
        print(f"[FAIL] {user_key} not found in user_df")
        return
    if user_key not in event_df.columns:
        print(f"[FAIL] {user_key} not found in event_df")
        return
    print(f"[PASS] Both datasets contain key column: {user_key}")

    # 3. Dtype check
    print("\n[3] KEY DTYPE CHECK")
    print(f"user_df[{user_key}] dtype : {user_df[user_key].dtype}")
    print(f"event_df[{user_key}] dtype: {event_df[user_key].dtype}")

    # 强制统一类型，避免 int/string 混用
    user_df[user_key] = user_df[user_key].astype(str)
    event_df[user_key] = event_df[user_key].astype(str)

    # 4. Null key check
    print("\n[4] NULL KEY CHECK")
    user_null = user_df[user_key].isnull().sum()
    event_null = event_df[user_key].isnull().sum()
    print(f"user_df null {user_key}: {user_null}")
    print(f"event_df null {user_key}: {event_null}")

    # 5. Uniqueness check
    print("\n[5] UNIQUENESS CHECK")
    user_dup = user_df.duplicated(subset=[user_key]).sum()
    event_dup = event_df.duplicated(subset=[user_key]).sum()
    print(f"user_df duplicate {user_key}: {user_dup}")
    print(f"event_df duplicate {user_key}: {event_dup}")

    if user_dup == 0:
        print("[PASS] user_df is one-row-per-user")
    else:
        print("[WARN] user_df is NOT one-row-per-user")

    if event_dup == 0:
        print("[PASS] event_df is one-row-per-user")
    else:
        print("[WARN] event_df is NOT one-row-per-user")

    # 6. User set overlap
    print("\n[6] USER SET OVERLAP CHECK")
    user_ids = set(user_df[user_key])
    event_ids = set(event_df[user_key])

    only_in_user = user_ids - event_ids
    only_in_event = event_ids - user_ids
    overlap = user_ids & event_ids

    print(f"user_df unique users : {len(user_ids)}")
    print(f"event_df unique users: {len(event_ids)}")
    print(f"overlap users        : {len(overlap)}")
    print(f"only in user_df      : {len(only_in_user)}")
    print(f"only in event_df     : {len(only_in_event)}")

    if len(user_ids) > 0:
        print(f"coverage of user_df by event_df: {len(overlap) / len(user_ids):.6f}")
    if len(event_ids) > 0:
        print(f"coverage of event_df by user_df: {len(overlap) / len(event_ids):.6f}")

    # 7. Sample unmatched IDs
    print("\n[7] SAMPLE UNMATCHED IDS")
    if len(only_in_user) > 0:
        print("Sample only_in_user:", list(sorted(only_in_user))[:10])
    else:
        print("No unmatched users found on user_df side.")

    if len(only_in_event) > 0:
        print("Sample only_in_event:", list(sorted(only_in_event))[:10])
    else:
        print("No unmatched users found on event_df side.")

    # 8. Safe left merge test
    print("\n[8] LEFT MERGE TEST")
    merged = user_df.merge(event_df, on=user_key, how="left", suffixes=("", "_event"))
    print(f"merged shape: {merged.shape}")

    if len(merged) == len(user_df):
        print("[PASS] Left merge keeps row count unchanged.")
    else:
        print("[FAIL] Left merge changed row count. Likely duplicate keys in event_df.")

    # 9. Missing feature ratio after merge
    print("\n[9] MISSING AFTER MERGE")
    new_cols = [c for c in event_df.columns if c != user_key]
    if len(new_cols) == 0:
        print("[WARN] event_df has no feature columns other than user_id.")
    else:
        missing_summary = merged[new_cols].isnull().mean().sort_values(ascending=False)
        print("Top missing ratios in merged new features:")
        print(missing_summary.head(10))

    # 10. Final judgment
    print("\n" + "=" * 100)
    print("FINAL JUDGMENT")
    print("=" * 100)

    can_perfect_merge = (
        user_dup == 0 and
        event_dup == 0 and
        len(only_in_user) == 0 and
        len(only_in_event) == 0 and
        len(merged) == len(user_df)
    )

    if can_perfect_merge:
        print("[PASS] The two datasets can be perfectly merged one-to-one on user_id.")
    else:
        print("[WARN] The two datasets cannot be perfectly merged yet.")
        print("You should check duplicates, unmatched users, or whether event_df is still event-level instead of user-level feature-level.")


if __name__ == "__main__":
    USER_LEVEL_PATH = r"compressed_data.csv"
    EVENT_FEATURE_PATH = r"event_feature_table.csv"  # 改成你的增强特征表
    check_merge_compatibility(USER_LEVEL_PATH, EVENT_FEATURE_PATH)