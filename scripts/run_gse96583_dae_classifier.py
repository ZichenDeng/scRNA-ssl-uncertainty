#!/usr/bin/env python3
"""Train DAE variants on GSE96583 and compare them against PCA."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "GSE96583_combined_shared_qc_singlets.h5ad"
LITE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "GSE96583_combined_shared_qc_singlets_lite.h5ad"
RESULTS_ROOT = PROJECT_ROOT / "results"
SLIDE_ROOT = PROJECT_ROOT / "deliverables" / "ms3_autoencoder"
FIGURE_ROOT = SLIDE_ROOT / "figures"

RANDOM_STATE = 1090
PCA_FEATURE_KEY = "X_pca"
PCA_N_COMPONENTS = 50


@dataclass
class DAEConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: tuple[int, int] = (512, 128)
    noise_type: str = "dropout"
    noise_level: float = 0.2
    reconstruction_loss: str = "mse"
    classifier: str = "logistic_regression"
    supervised_loss_weight: float = 1.0
    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 1e-3
    hidden_dropout: float = 0.1
    weight_decay: float = 1e-5
    patience: int = 8


class SparseRowDataset(Dataset):
    """Yield dense float32 tensors from a sparse matrix one row at a time."""

    def __init__(self, matrix) -> None:
        self.matrix = matrix.tocsr() if sparse.issparse(matrix) else np.asarray(matrix, dtype=np.float32)

    def __len__(self) -> int:
        return self.matrix.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.matrix[idx]
        if sparse.issparse(row):
            row = row.toarray().ravel()
        else:
            row = np.asarray(row).ravel()
        return torch.from_numpy(row.astype(np.float32, copy=False))


class LabeledSparseRowDataset(SparseRowDataset):
    """Yield dense float32 tensors together with encoded labels."""

    def __init__(self, matrix, labels: np.ndarray) -> None:
        super().__init__(matrix)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return super().__getitem__(idx), torch.tensor(self.labels[idx], dtype=torch.long)


class DenoisingAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: tuple[int, int],
        hidden_dropout: float,
        n_classes: int | None = None,
    ) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(h2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
        )
        self.classifier_head = nn.Linear(latent_dim, n_classes) if n_classes is not None else None

    def forward(
        self,
        x: torch.Tensor,
        noisy_x: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        inputs = noisy_x if noisy_x is not None else x
        z = self.encoder(inputs)
        recon = self.decoder(z)
        logits = self.classifier_head(z) if self.classifier_head is not None else None
        return recon, z, logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--noise-level", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--classifier",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "supervised_head"],
    )
    parser.add_argument("--supervised-loss-weight", type=float, default=1.0)
    parser.add_argument("--output-prefix", type=str, default="gse96583_dae")
    parser.add_argument("--representation-name", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_inputs() -> None:
    if DATA_PATH.exists() or LITE_DATA_PATH.exists():
        return


def build_lightweight_dataset() -> ad.AnnData:
    import prepare_gse96583_ms2 as prep

    prep.ensure_gse96583_raw_data()

    batch1 = prep.annotate_batch1(prep.load_subset(prep.BATCH1_FILES, "GSE96583_batch1.genes.tsv.gz", "batch1"))
    batch2 = prep.annotate_batch2(prep.load_subset(prep.BATCH2_FILES, "GSE96583_batch2.genes.tsv.gz", "batch2"))

    filtered_batches = []
    for adata in [batch1, batch2]:
        prep.add_qc_metrics(adata)
        sc.pp.filter_cells(adata, min_genes=prep.MIN_GENES)
        sc.pp.filter_genes(adata, min_cells=prep.MIN_CELLS)
        adata = adata[adata.obs["multiplets"] == "singlet"].copy()
        prep.add_qc_metrics(adata)
        adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
        adata.obs["batch_label"] = adata.obs["batch_label"].astype("category")
        adata.obs["condition"] = adata.obs["condition"].astype("category")
        filtered_batches.append(adata)

    batch1, batch2 = filtered_batches
    shared_genes = batch1.var_names.intersection(batch2.var_names)
    batch1 = batch1[:, shared_genes].copy()
    batch2 = batch2[:, shared_genes].copy()

    for adata in [batch1, batch2]:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    combined = ad.concat([batch1, batch2], join="inner", merge="same")
    combined.uns["lite_pipeline"] = {
        "builder": "run_gse96583_dae_classifier.py",
        "note": "Fallback lightweight dataset: basic QC + metadata singlets + shared genes, without Scrublet/Harmony.",
        "shared_genes": int(len(shared_genes)),
    }

    pca_input = combined.copy()
    sc.pp.highly_variable_genes(pca_input, n_top_genes=2000, flavor="seurat")
    if int(pca_input.var["highly_variable"].sum()) > 0:
        pca_input = pca_input[:, pca_input.var["highly_variable"]].copy()
    sc.pp.scale(pca_input, max_value=10)
    sc.tl.pca(pca_input, svd_solver="arpack", n_comps=min(PCA_N_COMPONENTS, pca_input.n_vars - 1))
    combined.obsm["X_pca"] = pca_input.obsm["X_pca"][:, :PCA_N_COMPONENTS]

    LITE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(LITE_DATA_PATH)
    return combined


def load_modeling_dataset() -> tuple[ad.AnnData, str, Path]:
    if DATA_PATH.exists():
        return ad.read_h5ad(DATA_PATH), "full_ms2_processed", DATA_PATH
    if LITE_DATA_PATH.exists():
        return ad.read_h5ad(LITE_DATA_PATH), "lite_fallback_cached", LITE_DATA_PATH
    combined = build_lightweight_dataset()
    return combined, "lite_fallback_built_from_raw", LITE_DATA_PATH


def make_fixed_splits(
    adata: ad.AnnData,
    label_col: str = "cell_type",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData, ad.AnnData]:
    if not abs(train_size + val_size + test_size - 1.0) < 1e-8:
        raise ValueError("train_size + val_size + test_size must sum to 1.")
    if label_col not in adata.obs.columns:
        raise KeyError(f"{label_col} not found in adata.obs")

    split_adata = adata.copy()
    labels = split_adata.obs[label_col].astype(str)
    indices = split_adata.obs_names.to_numpy()

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1.0 - train_size),
        random_state=random_state,
        stratify=labels.loc[indices],
    )

    temp_labels = labels.loc[temp_idx]
    val_fraction_of_temp = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=random_state,
        stratify=temp_labels,
    )

    split_adata.obs["split"] = "unassigned"
    split_adata.obs.loc[train_idx, "split"] = "train"
    split_adata.obs.loc[val_idx, "split"] = "val"
    split_adata.obs.loc[test_idx, "split"] = "test"
    split_adata.obs["split"] = split_adata.obs["split"].astype("category")
    split_adata.uns["split_config"] = {
        "label_col": label_col,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "random_state": random_state,
    }

    train_adata = split_adata[split_adata.obs["split"] == "train"].copy()
    val_adata = split_adata[split_adata.obs["split"] == "val"].copy()
    test_adata = split_adata[split_adata.obs["split"] == "test"].copy()
    return split_adata, train_adata, val_adata, test_adata


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loader(matrix, batch_size: int, shuffle: bool, labels: np.ndarray | None = None) -> DataLoader:
    dataset: Dataset
    if labels is None:
        dataset = SparseRowDataset(matrix)
    else:
        dataset = LabeledSparseRowDataset(matrix, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def corrupt_inputs(x: torch.Tensor, noise_type: str, noise_level: float) -> torch.Tensor:
    if noise_type == "dropout":
        return F.dropout(x, p=noise_level, training=True)
    raise ValueError(f"Unsupported noise_type: {noise_type}")


def reconstruction_loss_fn(name: str) -> nn.Module:
    if name.lower() == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported reconstruction_loss: {name}")


def evaluate_reconstruction(
    model: DenoisingAutoencoder,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    ce_loss_fn: nn.Module | None = None,
    supervised_weight: float = 1.0,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_class_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for payload in loader:
            if isinstance(payload, (list, tuple)):
                batch, labels = payload
                labels = labels.to(device)
            else:
                batch = payload
                labels = None
            batch = batch.to(device)
            recon, _, logits = model(batch)
            recon_loss = loss_fn(recon, batch)
            class_loss = torch.tensor(0.0, device=device)
            if labels is not None and ce_loss_fn is not None and logits is not None:
                class_loss = ce_loss_fn(logits, labels)
            loss = recon_loss + supervised_weight * class_loss
            total_loss += float(loss.item()) * batch.size(0)
            total_recon_loss += float(recon_loss.item()) * batch.size(0)
            total_class_loss += float(class_loss.item()) * batch.size(0)
            total_examples += batch.size(0)
    denom = max(total_examples, 1)
    return {
        "loss": total_loss / denom,
        "recon_loss": total_recon_loss / denom,
        "class_loss": total_class_loss / denom,
    }


def train_autoencoder(
    train_matrix,
    val_matrix,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    config: DAEConfig,
    device: torch.device,
) -> tuple[DenoisingAutoencoder, pd.DataFrame, LabelEncoder]:
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    val_labels_encoded = label_encoder.transform(val_labels)
    n_classes = len(label_encoder.classes_) if config.classifier == "supervised_head" else None

    model = DenoisingAutoencoder(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        hidden_dims=config.hidden_dims,
        hidden_dropout=config.hidden_dropout,
        n_classes=n_classes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = reconstruction_loss_fn(config.reconstruction_loss)
    ce_loss_fn = nn.CrossEntropyLoss() if config.classifier == "supervised_head" else None
    train_loader = build_loader(
        train_matrix,
        batch_size=config.batch_size,
        shuffle=True,
        labels=train_labels_encoded if config.classifier == "supervised_head" else None,
    )
    val_loader = build_loader(
        val_matrix,
        batch_size=config.batch_size,
        shuffle=False,
        labels=val_labels_encoded if config.classifier == "supervised_head" else None,
    )

    best_state = None
    best_val = float("inf")
    patience_left = config.patience
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_recon_loss = 0.0
        total_class_loss = 0.0
        total_examples = 0

        for payload in train_loader:
            if isinstance(payload, (list, tuple)):
                batch, labels = payload
                labels = labels.to(device)
            else:
                batch = payload
                labels = None

            batch = batch.to(device)
            noisy_batch = corrupt_inputs(batch, config.noise_type, config.noise_level)
            recon, _, logits = model(batch, noisy_x=noisy_batch)
            recon_loss = loss_fn(recon, batch)
            class_loss = torch.tensor(0.0, device=device)
            if labels is not None and ce_loss_fn is not None and logits is not None:
                class_loss = ce_loss_fn(logits, labels)
            loss = recon_loss + config.supervised_loss_weight * class_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += float(loss.item()) * batch.size(0)
            total_recon_loss += float(recon_loss.item()) * batch.size(0)
            total_class_loss += float(class_loss.item()) * batch.size(0)
            total_examples += batch.size(0)

        val_metrics = evaluate_reconstruction(
            model,
            val_loader,
            device,
            loss_fn,
            ce_loss_fn=ce_loss_fn,
            supervised_weight=config.supervised_loss_weight,
        )
        denom = max(total_examples, 1)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": total_train_loss / denom,
                "train_recon_loss": total_recon_loss / denom,
                "train_class_loss": total_class_loss / denom,
                "val_loss": val_metrics["loss"],
                "val_recon_loss": val_metrics["recon_loss"],
                "val_class_loss": val_metrics["class_loss"],
            }
        )

        if val_metrics["loss"] < best_val - 1e-6:
            best_val = val_metrics["loss"]
            patience_left = config.patience
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError("Autoencoder training finished without a best checkpoint.")

    model.load_state_dict(best_state)
    history = pd.DataFrame(history_rows)
    return model, history, label_encoder


def encode_matrix(
    model: DenoisingAutoencoder,
    matrix,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = build_loader(matrix, batch_size=batch_size, shuffle=False)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model.encode(batch)
            outputs.append(z.cpu().numpy())
    return np.vstack(outputs)


def predict_with_supervised_head(
    model: DenoisingAutoencoder,
    matrix,
    batch_size: int,
    device: torch.device,
    label_encoder: LabelEncoder,
) -> np.ndarray:
    if model.classifier_head is None:
        raise RuntimeError("Model does not have a supervised classifier head.")
    loader = build_loader(matrix, batch_size=batch_size, shuffle=False)
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, _, logits = model(batch)
            pred_idx = logits.argmax(dim=1).cpu().numpy()
            predictions.append(pred_idx)
    pred_encoded = np.concatenate(predictions)
    return label_encoder.inverse_transform(pred_encoded)


def summarize_predictions(
    pred: np.ndarray,
    split_y: np.ndarray,
    representation: str,
    classifier_name: str,
    split_name: str,
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame]:
    metrics_row = {
        "representation": representation,
        "classifier": classifier_name,
        "split": split_name,
        "n_cells": int(len(split_y)),
        "accuracy": float(accuracy_score(split_y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(split_y, pred)),
        "macro_f1": float(f1_score(split_y, pred, average="macro")),
        "weighted_f1": float(f1_score(split_y, pred, average="weighted")),
    }

    report = classification_report(split_y, pred, output_dict=True, zero_division=0)
    per_class = (
        pd.DataFrame(report)
        .T
        .rename(columns={"f1-score": "f1_score"})
        .drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
        .reset_index()
        .rename(columns={"index": "cell_type"})
    )
    per_class["representation"] = representation
    per_class["split"] = split_name

    label_order = sorted(np.unique(np.concatenate([split_y, pred])))
    cm = pd.DataFrame(
        confusion_matrix(split_y, pred, labels=label_order),
        index=label_order,
        columns=label_order,
    )
    cm_long = cm.reset_index().melt(id_vars="index", var_name="predicted_label", value_name="count")
    cm_long = cm_long.rename(columns={"index": "true_label"})
    cm_long["representation"] = representation
    cm_long["split"] = split_name
    return metrics_row, per_class, cm_long


def evaluate_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    split_x: np.ndarray,
    split_y: np.ndarray,
    split_name: str,
    representation: str,
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame]:
    clf = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    clf.fit(train_x, train_y)
    pred = clf.predict(split_x)
    return summarize_predictions(
        pred=pred,
        split_y=split_y,
        representation=representation,
        classifier_name="Logistic Regression",
        split_name=split_name,
    )


def run_pca_reference(split_adata: ad.AnnData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if PCA_FEATURE_KEY not in split_adata.obsm:
        raise KeyError(f"{PCA_FEATURE_KEY} is missing from split_adata.obsm")

    features = np.asarray(split_adata.obsm[PCA_FEATURE_KEY])[:, :PCA_N_COMPONENTS]
    labels = split_adata.obs["cell_type"].astype(str).to_numpy()
    splits = split_adata.obs["split"].astype(str).to_numpy()

    train_mask = splits == "train"
    train_x = features[train_mask]
    train_y = labels[train_mask]

    rows = []
    per_class_parts = []
    confusion_parts = []
    for split_name in ["val", "test"]:
        mask = splits == split_name
        row, part, cm_long = evaluate_classifier(
            train_x=train_x,
            train_y=train_y,
            split_x=features[mask],
            split_y=labels[mask],
            split_name=split_name,
            representation="PCA-50",
        )
        rows.append(row)
        per_class_parts.append(part)
        confusion_parts.append(cm_long)

    return pd.DataFrame(rows), pd.concat(per_class_parts, ignore_index=True), pd.concat(confusion_parts, ignore_index=True)


def run_dae_reference(
    train_adata: ad.AnnData,
    val_adata: ad.AnnData,
    test_adata: ad.AnnData,
    config: DAEConfig,
    device: torch.device,
    representation_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_y = train_adata.obs["cell_type"].astype(str).to_numpy()
    val_y = val_adata.obs["cell_type"].astype(str).to_numpy()
    test_y = test_adata.obs["cell_type"].astype(str).to_numpy()

    model, history, label_encoder = train_autoencoder(
        train_adata.X,
        val_adata.X,
        train_labels=train_y,
        val_labels=val_y,
        config=config,
        device=device,
    )

    rows = []
    per_class_parts = []
    confusion_parts = []

    if config.classifier == "supervised_head":
        eval_splits = [("val", val_adata.X, val_y), ("test", test_adata.X, test_y)]
        for split_name, split_matrix, split_y in eval_splits:
            pred = predict_with_supervised_head(
                model,
                split_matrix,
                batch_size=config.batch_size,
                device=device,
                label_encoder=label_encoder,
            )
            row, part, cm_long = summarize_predictions(
                pred=pred,
                split_y=split_y,
                representation=representation_name,
                classifier_name="Supervised Head",
                split_name=split_name,
            )
            rows.append(row)
            per_class_parts.append(part)
            confusion_parts.append(cm_long)
    else:
        train_z = encode_matrix(model, train_adata.X, batch_size=config.batch_size, device=device)
        val_z = encode_matrix(model, val_adata.X, batch_size=config.batch_size, device=device)
        test_z = encode_matrix(model, test_adata.X, batch_size=config.batch_size, device=device)
        for split_name, split_x, split_y in [("val", val_z, val_y), ("test", test_z, test_y)]:
            row, part, cm_long = evaluate_classifier(
                train_x=train_z,
                train_y=train_y,
                split_x=split_x,
                split_y=split_y,
                split_name=split_name,
                representation=representation_name,
            )
            rows.append(row)
            per_class_parts.append(part)
            confusion_parts.append(cm_long)

    return (
        pd.DataFrame(rows),
        pd.concat(per_class_parts, ignore_index=True),
        pd.concat(confusion_parts, ignore_index=True),
        history,
    )


def build_summary_tables(
    metrics: pd.DataFrame,
    per_class: pd.DataFrame,
    target_representation: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_metrics = metrics[metrics["split"] == "test"].copy()
    comparison = (
        test_metrics[["representation", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]]
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )

    per_class_test = per_class[per_class["split"] == "test"].copy()
    pivot = per_class_test.pivot(index="cell_type", columns="representation", values="f1_score")
    if {"PCA-50", target_representation}.issubset(pivot.columns):
        pivot["delta_dae_minus_pca"] = pivot[target_representation] - pivot["PCA-50"]
    return comparison, pivot.sort_values("delta_dae_minus_pca", ascending=False)


def save_training_curve(history: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["val_loss"], label="val")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_per_class_delta_figure(per_class_delta: pd.DataFrame, out_path: Path, representation_name: str) -> None:
    delta = per_class_delta.reset_index().sort_values("delta_dae_minus_pca")
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    colors = ["#2E7D6F" if value >= 0 else "#C44E52" for value in delta["delta_dae_minus_pca"]]
    ax.barh(delta["cell_type"], delta["delta_dae_minus_pca"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f"{representation_name} vs PCA Test F1 Delta by Cell Type")
    ax.set_xlabel(f"{representation_name} minus PCA-50 test F1")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_slide_markdown(
    config: DAEConfig,
    split_adata: ad.AnnData,
    metrics: pd.DataFrame,
    history: pd.DataFrame,
    per_class_delta: pd.DataFrame,
    dataset_source: str,
    dataset_path: Path,
    representation_name: str,
    out_path: Path,
) -> None:
    split_counts = split_adata.obs["split"].value_counts().sort_index()
    test_metrics = metrics.query("split == 'test'").set_index("representation")
    pca_test = test_metrics.loc["PCA-50"]
    dae_test = test_metrics.loc[representation_name]
    macro_delta = float(dae_test["macro_f1"] - pca_test["macro_f1"])
    acc_delta = float(dae_test["accuracy"] - pca_test["accuracy"])

    val_gap = (
        metrics.pivot(index="representation", columns="split", values="macro_f1")
        .assign(val_test_gap=lambda df: (df["val"] - df["test"]).abs())
        .sort_values("val_test_gap")
    )
    val_gap_table = "| representation | val | test | val_test_gap |\n| --- | ---: | ---: | ---: |\n"
    for representation, row in val_gap.iterrows():
        val_gap_table += (
            f"| {representation} | {row['val']:.3f} | {row['test']:.3f} | {row['val_test_gap']:.3f} |\n"
        )

    top_gains = per_class_delta.head(3).reset_index()
    top_drops = per_class_delta.tail(3).reset_index().sort_values("delta_dae_minus_pca")

    if macro_delta > 0:
        headline = f"{representation_name} improves test macro-F1 over PCA-50 by {macro_delta:+.3f}."
    elif macro_delta < 0:
        headline = f"{representation_name} does not beat PCA-50 yet; test macro-F1 changes by {macro_delta:+.3f}."
    else:
        headline = f"{representation_name} matches PCA-50 on test macro-F1 to three decimals."

    positive_gains = top_gains[top_gains["delta_dae_minus_pca"] > 0]
    if not positive_gains.empty:
        class_story_positive = "Biggest gains: " + "; ".join(
            f"{row['cell_type']} ({row['delta_dae_minus_pca']:+.3f})" for _, row in positive_gains.iterrows()
        )
    else:
        class_story_positive = "No cell type improved over PCA in this run. Smallest losses: " + "; ".join(
            f"{row['cell_type']} ({row['delta_dae_minus_pca']:+.3f})" for _, row in top_gains.iterrows()
        )
    negative_drops = top_drops[top_drops["delta_dae_minus_pca"] < 0]
    if not negative_drops.empty:
        class_story_negative = "Biggest drops: " + "; ".join(
            f"{row['cell_type']} ({row['delta_dae_minus_pca']:+.3f})" for _, row in negative_drops.iterrows()
        )
    else:
        class_story_negative = "No cell type dropped below PCA; smallest gains: " + "; ".join(
            f"{row['cell_type']} ({row['delta_dae_minus_pca']:+.3f})" for _, row in top_drops.iterrows()
        )

    slides = f"""# {representation_name} Slides

## Slide 1: Why this run exists

- Siheng's latest notebook push finishes data wrangling, PCA baselines, and the fixed `train / val / test` split.
- This run extends that handoff by testing `{representation_name}` against the same `PCA-50` baseline.
- The goal is not to over-claim. The goal is to learn whether a stronger DAE setup is moving in the right direction.

## Slide 2: Setup

- Input dataset source: `{dataset_source}`
- Input dataset path: `{dataset_path}`
- Input dimension: `{config.input_dim}` genes
- Split sizes: train `{int(split_counts['train'])}`, val `{int(split_counts['val'])}`, test `{int(split_counts['test'])}`
- DAE hidden dimensions: `{list(config.hidden_dims)}`
- Latent dimension: `{config.latent_dim}`
- Noise type / level: `{config.noise_type}` / `{config.noise_level}`
- Reconstruction loss: `{config.reconstruction_loss.upper()}`
- Classifier mode: `{config.classifier}`
- Supervised loss weight: `{config.supervised_loss_weight}`
- Training epochs actually used: `{int(history['epoch'].iloc[-1])}`

## Slide 3: Main result

- {headline}
- Test accuracy delta: `{acc_delta:+.3f}`
- PCA-50 test metrics: accuracy `{pca_test['accuracy']:.3f}`, macro-F1 `{pca_test['macro_f1']:.3f}`, balanced accuracy `{pca_test['balanced_accuracy']:.3f}`
- {representation_name} test metrics: accuracy `{dae_test['accuracy']:.3f}`, macro-F1 `{dae_test['macro_f1']:.3f}`, balanced accuracy `{dae_test['balanced_accuracy']:.3f}`
- Validation/test macro-F1 stability:
{val_gap_table}

## Slide 4: Class-level story

- {class_story_positive}
- {class_story_negative}
- Use the matching per-class delta figure for this run.

## Slide 5: Training story

- Best validation total loss: `{history['val_loss'].min():.4f}`
- Final restored checkpoint validation total loss: `{history['val_loss'].iloc[-1]:.4f}`
- Use the matching training-curve figure for this run.

## Slide 6: Caveat / next step

- This still uses the notebook's fixed split, so it is a representation-learning comparison before the stricter cross-batch transfer test.
- If the dataset source is a `lite_fallback_*` variant, this run uses the raw GEO matrices plus metadata singlets, not the full heavy MS2 object.
- If this variant helps, the next step is to retest under the original cross-batch and cross-condition story.
- If it still loses to PCA, the next tuning knobs are latent width, corruption strength, classifier loss weight, and stronger supervision.
"""
    out_path.write_text(slides, encoding="utf-8")


def sanitize_suffix(text: str) -> str:
    chars: list[str] = []
    for char in text:
        if char.isalnum():
            chars.append(char.lower())
        elif char in {"-", "_"}:
            chars.append(char)
        else:
            chars.append("_")
    return "".join(chars).strip("_")


def main() -> None:
    args = parse_args()
    ensure_inputs()
    set_seed(RANDOM_STATE)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    adata, dataset_source, dataset_path = load_modeling_dataset()
    split_adata, train_adata, val_adata, test_adata = make_fixed_splits(adata)

    config = DAEConfig(
        input_dim=int(train_adata.n_vars),
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        noise_level=args.noise_level,
        learning_rate=args.learning_rate,
        patience=args.patience,
        classifier=args.classifier,
        supervised_loss_weight=args.supervised_loss_weight,
    )

    if args.representation_name is None:
        if config.classifier == "supervised_head":
            representation_name = f"SupDAE-{config.latent_dim}"
        else:
            representation_name = f"DAE-{config.latent_dim}"
    else:
        representation_name = args.representation_name

    prefix = sanitize_suffix(args.output_prefix)
    metrics_path = RESULTS_ROOT / f"{prefix}_metrics.csv"
    per_class_path = RESULTS_ROOT / f"{prefix}_per_class.csv"
    confusions_path = RESULTS_ROOT / f"{prefix}_confusions.csv"
    history_path = RESULTS_ROOT / f"{prefix}_training_history.csv"
    split_manifest_path = RESULTS_ROOT / f"{prefix}_split_manifest.csv"
    run_summary_path = RESULTS_ROOT / f"{prefix}_run_summary.json"
    figure_training_path = FIGURE_ROOT / f"{prefix}_training_curve.png"
    figure_delta_path = FIGURE_ROOT / f"{prefix}_per_class_delta.png"
    slide_path = SLIDE_ROOT / f"{prefix}_slides.md"

    pca_metrics, pca_per_class, pca_confusions = run_pca_reference(split_adata)
    dae_metrics, dae_per_class, dae_confusions, history = run_dae_reference(
        train_adata=train_adata,
        val_adata=val_adata,
        test_adata=test_adata,
        config=config,
        device=device,
        representation_name=representation_name,
    )

    metrics = pd.concat([pca_metrics, dae_metrics], ignore_index=True)
    per_class = pd.concat([pca_per_class, dae_per_class], ignore_index=True)
    confusions = pd.concat([pca_confusions, dae_confusions], ignore_index=True)
    comparison, per_class_delta = build_summary_tables(metrics, per_class, target_representation=representation_name)

    metrics.to_csv(metrics_path, index=False)
    per_class.to_csv(per_class_path, index=False)
    confusions.to_csv(confusions_path, index=False)
    history.to_csv(history_path, index=False)
    split_adata.obs[["split", "cell_type", "batch_label", "condition"]].to_csv(split_manifest_path)

    save_training_curve(history, figure_training_path, title=f"{representation_name} Training Loss")
    save_per_class_delta_figure(per_class_delta, figure_delta_path, representation_name=representation_name)
    build_slide_markdown(
        config=config,
        split_adata=split_adata,
        metrics=metrics,
        history=history,
        per_class_delta=per_class_delta,
        dataset_source=dataset_source,
        dataset_path=dataset_path,
        representation_name=representation_name,
        out_path=slide_path,
    )

    run_summary = {
        "data_path": str(dataset_path),
        "dataset_source": dataset_source,
        "device": str(device),
        "representation_name": representation_name,
        "output_prefix": prefix,
        "config": asdict(config),
        "best_val_loss": float(history["val_loss"].min()),
        "metrics_file": str(metrics_path),
        "per_class_file": str(per_class_path),
        "confusion_file": str(confusions_path),
        "history_file": str(history_path),
        "slides_file": str(slide_path),
    }
    run_summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("Test comparison:")
    print(comparison.round(4).to_string(index=False))
    print("\nTop per-class experiment minus PCA deltas:")
    print(per_class_delta.round(4).head(8).to_string())
    print("\nSaved artifacts:")
    for path in [
        metrics_path,
        per_class_path,
        confusions_path,
        history_path,
        split_manifest_path,
        figure_training_path,
        figure_delta_path,
        slide_path,
        run_summary_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()
