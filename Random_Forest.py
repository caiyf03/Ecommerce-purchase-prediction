import pandas as pd
import numpy as np

# =========================
# 1. 读取数据
# =========================
file_path = r"F:\CIS5450\2020-Jan.csv"

# 如果文件很大，可以先只读前几万行试试：
df = pd.read_csv(file_path, nrows=50000)

#df = pd.read_csv(file_path)

print("=" * 80)
print("1. 数据读取成功")
print("=" * 80)

# =========================
# 2. 基本信息
# =========================
print("\n[数据规模]")
print("Shape:", df.shape)

print("\n[列名]")
print(df.columns.tolist())

print("\n[前5行]")
print(df.head())

print("\n[后5行]")
print(df.tail())

# =========================
# 3. 数据类型
# =========================
print("\n" + "=" * 80)
print("2. 数据类型")
print("=" * 80)
print(df.dtypes)

# 分类列和数值列
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
object_cols = df.select_dtypes(include=["object"]).columns.tolist()
datetime_like_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]

print("\n[数值列]")
print(numeric_cols)

print("\n[类别/字符串列]")
print(object_cols)

print("\n[疑似时间列]")
print(datetime_like_cols)

# =========================
# 4. 缺失值检查
# =========================
print("\n" + "=" * 80)
print("3. 缺失值检查")
print("=" * 80)

missing_count = df.isna().sum()
missing_ratio = (df.isna().sum() / len(df)).sort_values(ascending=False)

missing_df = pd.DataFrame({
    "missing_count": missing_count,
    "missing_ratio": missing_count / len(df)
}).sort_values(by="missing_ratio", ascending=False)

print("\n[缺失值前20列]")
print(missing_df.head(20))

# =========================
# 5. 数值列统计
# =========================
print("\n" + "=" * 80)
print("4. 数值列描述统计")
print("=" * 80)

if len(numeric_cols) > 0:
    print(df[numeric_cols].describe().T)
else:
    print("没有检测到数值列。")

# =========================
# 6. 类别列简单检查
# =========================
print("\n" + "=" * 80)
print("5. 类别列概览")
print("=" * 80)

for col in object_cols[:10]:  # 先只看前10个类别列，避免输出太长
    print(f"\n列名: {col}")
    print("唯一值数量:", df[col].nunique(dropna=False))
    print("前10个取值:")
    print(df[col].value_counts(dropna=False).head(10))

# =========================
# 7. 判断是否存在常见ID列
# =========================
print("\n" + "=" * 80)
print("6. 疑似ID列检查")
print("=" * 80)

id_keywords = ["id", "user", "customer", "household", "session", "order"]
possible_id_cols = []

for col in df.columns:
    lower_col = col.lower()
    if any(keyword in lower_col for keyword in id_keywords):
        possible_id_cols.append(col)

print("疑似ID列:", possible_id_cols)

for col in possible_id_cols:
    print(f"\n[{col}]")
    print("唯一值数量:", df[col].nunique(dropna=False))
    print("样例值:")
    print(df[col].head(10).tolist())

# =========================
# 8. 判断每行可能代表什么
# =========================
print("\n" + "=" * 80)
print("7. 粒度初步判断")
print("=" * 80)

# 粗略规则：
# 如果有 user/customer/household id，并且这些id重复很多次，
# 那每行可能不是“一个用户/家庭”，而是 transaction / event 级别
for col in possible_id_cols:
    nunique = df[col].nunique(dropna=False)
    total_rows = len(df)
    ratio = nunique / total_rows
    print(f"{col}: unique={nunique}, total_rows={total_rows}, unique_ratio={ratio:.4f}")

    if ratio < 0.9:
        print(f"-> {col} 很可能重复出现，说明每行可能不是 '{col}' 的唯一一条记录。")
    else:
        print(f"-> {col} 可能接近一行一个实体。")

# =========================
# 9. 时间列尝试转换
# =========================
print("\n" + "=" * 80)
print("8. 时间列检查")
print("=" * 80)

for col in datetime_like_cols:
    try:
        parsed = pd.to_datetime(df[col], errors="coerce")
        print(f"\n列 {col} 可解析为时间的比例: {(parsed.notna().mean() * 100):.2f}%")
        print(parsed.head())
    except Exception as e:
        print(f"\n列 {col} 时间解析失败: {e}")

# =========================
# 10. 输出简要诊断结论
# =========================
print("\n" + "=" * 80)
print("9. 简要诊断结论")
print("=" * 80)

print(f"总行数: {df.shape[0]}")
print(f"总列数: {df.shape[1]}")
print(f"数值列数量: {len(numeric_cols)}")
print(f"类别列数量: {len(object_cols)}")
print(f"疑似ID列: {possible_id_cols}")
print(f"疑似时间列: {datetime_like_cols}")

high_missing_cols = missing_df[missing_df["missing_ratio"] > 0.3].index.tolist()
print(f"缺失率 > 30% 的列: {high_missing_cols}")

print("\n建议你重点回答下面这些问题：")
print("1. 每一行到底代表什么？")
print("2. 有没有 household/user/customer 级别的唯一标识？")
print("3. target 列可能是哪一列？")
print("4. 是不是需要先按用户/家庭聚合后才能建模？")