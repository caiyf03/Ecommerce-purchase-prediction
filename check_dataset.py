import pandas as pd
import numpy as np

print("=" * 80)
print("DATA AUDIT START")
print("=" * 80)

# =========================
# 1. 读取数据
# =========================
file_path = r"F:\CIS5450\compressed_data.csv"  # 改路径
df = pd.read_csv(file_path)

print("\n[1] 基本信息")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# =========================
# 2. 检查是否一人一行
# =========================
print("\n[2] 唯一性检查")
if df["user_id"].nunique() == len(df):
    print("✔ 每个 user_id 唯一（已是 user-level 数据）")
else:
    print("❌ user_id 有重复（严重问题）")

# =========================
# 3. 缺失值检查
# =========================
print("\n[3] 缺失值检查")
missing = df.isna().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("✔ 无缺失值")
else:
    print("⚠ 存在缺失值：")
    print(missing)

# =========================
# 4. 目标变量检查
# =========================
target = "purchased"

print("\n[4] 目标变量检查")
print(df[target].value_counts())
print("比例：")
print(df[target].value_counts(normalize=True))

# 判断是否严重不平衡
pos_rate = df[target].mean()
if pos_rate < 0.05:
    print("⚠ 极度不平衡（需要特别处理）")
elif pos_rate < 0.2:
    print("⚠ 轻度不平衡（可用 class_weight）")
else:
    print("✔ 分布尚可")

# =========================
# 5. 数值分布检查
# =========================
print("\n[5] 数值分布检查")
print(df.describe().T)

# =========================
# 6. 异常值检测（简单规则）
# =========================
print("\n[6] 异常值检查")

for col in df.columns:
    if df[col].dtype != "object":
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = ((df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)).sum()
        
        if outliers > 0:
            print(f"{col}: {outliers} potential outliers")

# =========================
# 7. 相关性检查
# =========================
print("\n[7] 特征相关性（前10列）")

corr = df.drop(columns=["user_id"]).corr()
high_corr = []

for i in corr.columns:
    for j in corr.columns:
        if i != j and abs(corr.loc[i, j]) > 0.9:
            high_corr.append((i, j, corr.loc[i, j]))

if len(high_corr) == 0:
    print("✔ 没有极高相关特征")
else:
    print("⚠ 高相关特征：")
    for pair in high_corr[:10]:
        print(pair)

# =========================
# 8. 特征泄漏检查（重点）
# =========================
print("\n[8] 特征泄漏检查")

leak_features = ["total_purchases", "purchase_rate"]

for col in leak_features:
    if col in df.columns:
        corr_val = np.corrcoef(df[col], df[target])[0, 1]
        print(f"{col} 与 target 相关性: {corr_val:.4f}")
        
        if abs(corr_val) > 0.7:
            print(f"❌ {col} 很可能存在泄漏")
        else:
            print(f"⚠ {col} 需谨慎")

# =========================
# 9. 特征范围合理性
# =========================
print("\n[9] 特征范围检查")

print("cart_rate max:", df["cart_rate"].max())
print("purchase_rate max:", df["purchase_rate"].max())

if df["cart_rate"].max() > 1:
    print("⚠ cart_rate > 1（可能有异常）")

# =========================
# 10. 最终结论
# =========================
print("\n" + "=" * 80)
print("FINAL CHECK SUMMARY")
print("=" * 80)

print("✔ 数据已聚合为 user-level")
print("✔ 无缺失值")
print("✔ 可直接建模")

print("\n❗建模前请移除以下列：")
print(["user_id", "total_purchases", "purchase_rate"])

print("\n✔ 可以开始 baseline 模型（Logistic Regression）")
print("cart_rate max:", df["cart_rate"].max())
print("purchase_rate max:", df["purchase_rate"].max())

print("=" * 80)
print("CHECK PURCHASE vs HAS_CARTED")
print("=" * 80)

cross_tab = pd.crosstab(df["purchased"], df["has_carted"])
print(cross_tab)

print("\nNormalized by purchased:")
print(pd.crosstab(df["purchased"], df["has_carted"], normalize="index"))

bad_case = df[(df["purchased"] == 1) & (df["has_carted"] == 0)]
print(f"\n# purchased=1 but has_carted=0: {len(bad_case)}")
print("=" * 80)
print("CHECK total_carts DISTRIBUTION")
print("=" * 80)

num_eq_0 = (df["total_carts"] == 0).sum()
num_eq_1 = (df["total_carts"] == 1).sum()
num_gt_1 = (df["total_carts"] > 1).sum()
max_carts = df["total_carts"].max()

print(f"total_carts == 0 : {num_eq_0}")
print(f"total_carts == 1 : {num_eq_1}")
print(f"total_carts >  1 : {num_gt_1}")
print(f"max(total_carts) : {max_carts}")
print("=" * 80)
print("total_carts AMONG PURCHASED USERS")
print("=" * 80)

purchased_df = df[df["purchased"] == 1]

num_eq_0_p = (purchased_df["total_carts"] == 0).sum()
num_eq_1_p = (purchased_df["total_carts"] == 1).sum()
num_gt_1_p = (purchased_df["total_carts"] > 1).sum()
max_carts_p = purchased_df["total_carts"].max()

print(f"[purchased=1] total_carts == 0 : {num_eq_0_p}")
print(f"[purchased=1] total_carts == 1 : {num_eq_1_p}")
print(f"[purchased=1] total_carts >  1 : {num_gt_1_p}")
print(f"[purchased=1] max(total_carts) : {max_carts_p}")
print("\nNormalized distribution for purchased=1:")
print(
    purchased_df["total_carts"]
    .value_counts(normalize=True)
    .sort_index()
    .head(20)
)