import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from data_split import load_and_prepare_data, print_split_summary


# =========================================================
# Global config
# =========================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


@dataclass
class ExperimentConfig:
    file_path: str = r"F:\CIS5450\compressed_data.csv"
    event_feature_path: str = r"F:\CIS5450\event_feature_table_v3.csv"
    target_col: str = "purchased"

    # ===== Three stages =====
    # Stage 1: use_event_features=False, use_embedding_features=False
    # Stage 2: use_event_features=True,  use_embedding_features=False
    # Stage 3: use_event_features=True,  use_embedding_features=True
    use_event_features: bool = True
    use_embedding_features: bool = True

    # training
    batch_size: int = 4096
    epochs: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    threshold: float = 0.5

    load_model: bool = False
    force_retrain: bool = False


# =========================================================
# Dataset
# =========================================================
class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long) if X_cat is not None else None
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.X_cat is None:
            return self.X_num[idx], None, self.y[idx]
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


# =========================================================
# Model
# =========================================================
class EmbeddingMLP(nn.Module):
    def __init__(self, num_numeric_features, categorical_cardinalities=None):
        super().__init__()

        self.has_cat = categorical_cardinalities is not None and len(categorical_cardinalities) > 0

        if self.has_cat:
            self.embeddings = nn.ModuleList()
            emb_dims = []
            for card in categorical_cardinalities:
                # 常见经验公式
                emb_dim = min(50, (card + 1) // 2)
                emb_dim = max(4, emb_dim)
                self.embeddings.append(nn.Embedding(card, emb_dim))
                emb_dims.append(emb_dim)
            self.total_emb_dim = sum(emb_dims)
        else:
            self.embeddings = None
            self.total_emb_dim = 0

        input_dim = num_numeric_features + self.total_emb_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def forward(self, x_num, x_cat=None):
        if self.has_cat and x_cat is not None:
            embedded = []
            for i, emb in enumerate(self.embeddings):
                embedded.append(emb(x_cat[:, i]))
            x_emb = torch.cat(embedded, dim=1)
            x = torch.cat([x_num, x_emb], dim=1)
        else:
            x = x_num

        logits = self.mlp(x).squeeze(1)
        return logits


# =========================================================
# Feature preparation helpers
# =========================================================
def build_stage_name(use_event_features: bool, use_embedding_features: bool) -> str:
    if not use_event_features and not use_embedding_features:
        return "basic"
    if use_event_features and not use_embedding_features:
        return "event_numeric"
    if use_event_features and use_embedding_features:
        return "event_embedding"
    return "custom"


def prepare_base_data(cfg: ExperimentConfig):
    X_train, X_test, y_train, y_test, df = load_and_prepare_data(
        file_path=cfg.file_path,
        target_col=cfg.target_col,
        drop_cols=["total_purchases", "purchase_rate", "total_events", "cart_rate"],
        test_size=0.2,
        random_state=42,
        stratify=True,
    )
    return X_train, X_test, y_train, y_test, df


def merge_event_features(X_train, X_test, event_feature_path):
    event_df = pd.read_csv(event_feature_path)
    assert "user_id" in event_df.columns, "event_feature_table must contain user_id"

    X_train = X_train.merge(event_df, on="user_id", how="left")
    X_test = X_test.merge(event_df, on="user_id", how="left")

    # 删除明显泄露特征
    leakage_cols = [
        "purchase",
        "purchase_count",
        "purchase_per_event",
        "cart_to_purchase_rate",
        "time_to_first_purchase",
        "fast_purchase",
        # 下面这组如果是全量数据计算的 conversion，先删掉
        "user_avg_category_conversion",
        "user_max_category_conversion",
        "user_min_category_conversion",
        "user_std_category_conversion",
        "top_category_conversion",
        "high_conversion_category_ratio",
    ]
    X_train = X_train.drop(columns=[c for c in leakage_cols if c in X_train.columns])
    X_test = X_test.drop(columns=[c for c in leakage_cols if c in X_test.columns])

    # 删除重复/高度冗余特征
    duplicate_cols = [
        "view",               # 已有 total_views
        "cart",               # 已有 total_carts
        "num_products_y",
        "num_categories_y",
        "avg_price_y",
        "max_price_y",
        "min_price_y",
        "total_events",
    ]
    X_train = X_train.drop(columns=[c for c in duplicate_cols if c in X_train.columns])
    X_test = X_test.drop(columns=[c for c in duplicate_cols if c in X_test.columns])

    return X_train, X_test


def split_numeric_and_categorical(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    use_embedding_features: bool,
):
    # user_id 只用于 merge
    X_train = X_train.drop(columns=["user_id"], errors="ignore").copy()
    X_test = X_test.drop(columns=["user_id"], errors="ignore").copy()

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    if not use_embedding_features:
        # 不使用 embedding 时，直接丢弃 object 列
        X_train_num = X_train[numeric_cols].copy()
        X_test_num = X_test[numeric_cols].copy()
        X_train_cat = None
        X_test_cat = None
        cat_cardinalities = None
        return X_train_num, X_test_num, X_train_cat, X_test_cat, numeric_cols, [], cat_cardinalities

    # 使用 embedding 时，保留 categorical
    X_train_num = X_train[numeric_cols].copy()
    X_test_num = X_test[numeric_cols].copy()

    X_train_cat = X_train[categorical_cols].copy()
    X_test_cat = X_test[categorical_cols].copy()

    # 缺失填 unknown
    for col in categorical_cols:
        X_train_cat[col] = X_train_cat[col].fillna("unknown").astype(str)
        X_test_cat[col] = X_test_cat[col].fillna("unknown").astype(str)

    # 训练集建 vocab，测试集未知值映射为 0
    cat_cardinalities = []
    X_train_cat_encoded = pd.DataFrame(index=X_train_cat.index)
    X_test_cat_encoded = pd.DataFrame(index=X_test_cat.index)

    for col in categorical_cols:
        train_values = X_train_cat[col].unique().tolist()
        vocab = {v: i + 1 for i, v in enumerate(train_values)}  # 0 留给 unknown/unseen

        X_train_cat_encoded[col] = X_train_cat[col].map(vocab).fillna(0).astype(int)
        X_test_cat_encoded[col] = X_test_cat[col].map(vocab).fillna(0).astype(int)

        cat_cardinalities.append(len(vocab) + 1)

    return (
        X_train_num,
        X_test_num,
        X_train_cat_encoded,
        X_test_cat_encoded,
        numeric_cols,
        categorical_cols,
        cat_cardinalities,
    )


def standardize_numeric(X_train_num: pd.DataFrame, X_test_num: pd.DataFrame):
    train_mean = X_train_num.mean()
    train_std = X_train_num.std().replace(0, 1)

    X_train_scaled = (X_train_num - train_mean) / train_std
    X_test_scaled = (X_test_num - train_mean) / train_std

    X_train_scaled = X_train_scaled.fillna(0)
    X_test_scaled = X_test_scaled.fillna(0)

    return X_train_scaled.values, X_test_scaled.values


# =========================================================
# Training / Evaluation
# =========================================================
def build_dataloaders(X_train_num, X_test_num, X_train_cat, X_test_cat, y_train, y_test, batch_size):
    train_ds = TabularDataset(
        X_train_num,
        X_train_cat.values if X_train_cat is not None else None,
        np.array(y_train),
    )
    test_ds = TabularDataset(
        X_test_num,
        X_test_cat.values if X_test_cat is not None else None,
        np.array(y_test),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def train_model(model, train_loader, y_train, epochs, lr, weight_decay):
    model = model.to(DEVICE)

    y_train_np = np.array(y_train)
    num_pos = (y_train_np == 1).sum()
    num_neg = (y_train_np == 0).sum()
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32, device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for x_num, x_cat, y in train_loader:
            x_num = x_num.to(DEVICE)
            y = y.to(DEVICE)

            if x_cat is not None:
                x_cat = x_cat.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x_num, x_cat)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(y)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.6f}")

    return model


def evaluate_model(model, test_loader, threshold=0.5):
    model.eval()

    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_num, x_cat, y in test_loader:
            x_num = x_num.to(DEVICE)
            y = y.to(DEVICE)

            if x_cat is not None:
                x_cat = x_cat.to(DEVICE)

            logits = model(x_num, x_cat)
            probs = torch.sigmoid(logits)

            preds = (probs >= threshold).long()

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    y_prob = np.concatenate(all_probs)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "=" * 80)
    print("EMBEDDING + MLP RESULTS")
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
        "y_true": y_true,
    }


# =========================================================
# Main experiment
# =========================================================
def run_experiment(cfg: ExperimentConfig):
    stage_name = build_stage_name(cfg.use_event_features, cfg.use_embedding_features)
    print("\n" + "=" * 80)
    print(f"RUNNING STAGE: {stage_name}")
    print("=" * 80)

    X_train, X_test, y_train, y_test, df = prepare_base_data(cfg)

    if cfg.use_event_features:
        print("\n" + "=" * 80)
        print("ADDING EVENT-LEVEL FEATURES")
        print("=" * 80)
        X_train, X_test = merge_event_features(
            X_train, X_test, cfg.event_feature_path
        )
        print(f"Merged train shape: {X_train.shape}")
        print(f"Merged test shape : {X_test.shape}")

    (
        X_train_num_df,
        X_test_num_df,
        X_train_cat_df,
        X_test_cat_df,
        numeric_cols,
        categorical_cols,
        cat_cardinalities,
    ) = split_numeric_and_categorical(
        X_train, X_test, cfg.use_embedding_features
    )

    print_split_summary(X_train_num_df, X_test_num_df, y_train, y_test)

    print("\nNumeric feature count:", len(numeric_cols))
    print("Categorical feature count:", len(categorical_cols))
    if categorical_cols:
        print("Categorical columns:", categorical_cols)

    X_train_num, X_test_num = standardize_numeric(X_train_num_df, X_test_num_df)

    train_loader, test_loader = build_dataloaders(
        X_train_num,
        X_test_num,
        X_train_cat_df,
        X_test_cat_df,
        y_train,
        y_test,
        cfg.batch_size,
    )

    model_path = f"embedding_mlp_{stage_name}.pt"

    if cfg.load_model and os.path.exists(model_path) and not cfg.force_retrain:
        print(f"\nLoading existing model from {model_path}")
        model = EmbeddingMLP(
            num_numeric_features=X_train_num.shape[1],
            categorical_cardinalities=cat_cardinalities,
        )
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model = model.to(DEVICE)
    else:
        model = EmbeddingMLP(
            num_numeric_features=X_train_num.shape[1],
            categorical_cardinalities=cat_cardinalities,
        )
        model = train_model(
            model,
            train_loader,
            y_train,
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        torch.save(model.state_dict(), model_path)
        print(f"\nModel saved to {model_path}")

    metrics = evaluate_model(model, test_loader, threshold=cfg.threshold)

    pred_df = pd.DataFrame({
        "actual": metrics["y_true"],
        "pred_score": metrics["y_prob"],
        "pred_label": metrics["y_pred"],
    })
    pred_path = f"embedding_mlp_predictions_{stage_name}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")

    return metrics


if __name__ == "__main__":
    cfg = ExperimentConfig(
        file_path=r"F:\CIS5450\compressed_data.csv",
        event_feature_path=r"F:\CIS5450\event_feature_table_v3.csv",

        # ===== choose one stage =====
        use_event_features=True,
        use_embedding_features=True,

        batch_size=4096,
        epochs=8,
        lr=1e-3,
        weight_decay=1e-5,
        threshold=0.5,

        load_model=False,
        force_retrain=False,
    )

    run_experiment(cfg)