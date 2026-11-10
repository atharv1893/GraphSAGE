"""Train GraphSAGE on recorded Bio-DECAGON pairs; absent labels are treated as negatives."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class GraphSAGEDataset:
    node_features: torch.Tensor
    feature_mapping: list[tuple[str, str]]
    edge_index: torch.Tensor
    ppi_edge_index: torch.Tensor
    drug_pairs: torch.Tensor
    label_indices: np.ndarray
    label_indptr: np.ndarray
    drug_ids: list[str]
    side_effect_ids: list[str]
    side_effect_names: dict[str, str]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_pairs(
    pair_count: int, seed: int, val_ratio: float, test_ratio: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pair_count < 3:
        raise ValueError(f"At least 3 unique drug pairs are required; found {pair_count}.")
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Validation and test ratios must be positive and sum to less than 1.")

    order = np.random.default_rng(seed).permutation(pair_count)
    test_count = max(1, int(round(pair_count * test_ratio)))
    val_count = max(1, int(round(pair_count * val_ratio)))
    if test_count + val_count >= pair_count:
        raise ValueError("Validation and test ratios leave no training pairs.")
    return (
        order[test_count + val_count :],
        order[test_count : test_count + val_count],
        order[:test_count],
    )


def load_data(
    data_dir: Path,
    chunk_size: int,
) -> GraphSAGEDataset:
    combo_path = data_dir / "bio-decagon-combo.csv"
    targets_path = data_dir / "bio-decagon-targets-all.csv"
    mono_path = data_dir / "bio-decagon-mono.csv"
    ppi_path = data_dir / "bio-decagon-ppi.csv"
    for path in (combo_path, targets_path, mono_path, ppi_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset file not found: {path}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    drug_set: set[str] = set()
    effect_set: set[str] = set()
    effect_names: dict[str, str] = {}
    combo_columns = [
        "STITCH 1",
        "STITCH 2",
        "Polypharmacy Side Effect",
        "Side Effect Name",
    ]

    for chunk in pd.read_csv(
        combo_path,
        usecols=combo_columns,
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    ):
        for column in combo_columns:
            chunk[column] = chunk[column].str.strip()
            if (chunk[column] == "").any():
                raise ValueError(f"{combo_path.name} contains empty values in {column!r}.")
        drug_set.update(chunk["STITCH 1"].unique())
        drug_set.update(chunk["STITCH 2"].unique())
        effect_set.update(chunk["Polypharmacy Side Effect"].unique())
        for effect_id, name in chunk[
            ["Polypharmacy Side Effect", "Side Effect Name"]
        ].drop_duplicates().itertuples(index=False, name=None):
            previous_name = effect_names.get(effect_id)
            if previous_name is not None and previous_name.lower() != name.lower():
                raise ValueError(f"Conflicting side-effect names for {effect_id!r}.")
            effect_names[effect_id] = name

    drug_ids = sorted(drug_set)
    side_effect_ids = sorted(effect_set)
    drug_to_idx = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}
    effect_to_idx = {effect_id: idx for idx, effect_id in enumerate(side_effect_ids)}
    n_drugs = len(drug_ids)

    pair_key_chunks: list[np.ndarray] = []
    effect_index_chunks: list[np.ndarray] = []
    for chunk in pd.read_csv(
        combo_path,
        usecols=["STITCH 1", "STITCH 2", "Polypharmacy Side Effect"],
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    ):
        drug_a = chunk["STITCH 1"].map(drug_to_idx).to_numpy(dtype=np.int64)
        drug_b = chunk["STITCH 2"].map(drug_to_idx).to_numpy(dtype=np.int64)
        pair_key_chunks.append(
            np.minimum(drug_a, drug_b) * np.int64(n_drugs)
            + np.maximum(drug_a, drug_b)
        )
        effect_index_chunks.append(
            chunk["Polypharmacy Side Effect"].map(effect_to_idx).to_numpy(dtype=np.int32)
        )

    if not pair_key_chunks:
        raise ValueError(f"No side-effect records found in {combo_path}.")
    pair_keys = np.concatenate(pair_key_chunks)
    effect_indices = np.concatenate(effect_index_chunks)
    sort_order = np.lexsort((effect_indices, pair_keys))
    pair_keys = pair_keys[sort_order]
    effect_indices = effect_indices[sort_order]

    unique_records = np.ones(len(pair_keys), dtype=bool)
    unique_records[1:] = (pair_keys[1:] != pair_keys[:-1]) | (
        effect_indices[1:] != effect_indices[:-1]
    )
    pair_keys = pair_keys[unique_records]
    effect_indices = effect_indices[unique_records]
    unique_pair_keys, pair_starts = np.unique(pair_keys, return_index=True)
    label_indptr = np.append(pair_starts, len(effect_indices)).astype(np.int64)
    drug_pairs = torch.from_numpy(
        np.column_stack(
            (unique_pair_keys // n_drugs, unique_pair_keys % n_drugs)
        ).astype(np.int64)
    )

    targets = pd.read_csv(
        targets_path,
        usecols=["STITCH", "Gene"],
        dtype=str,
        keep_default_na=False,
    )
    targets["STITCH"] = targets["STITCH"].str.strip()
    targets["Gene"] = targets["Gene"].str.strip()
    targets = targets[
        targets["STITCH"].isin(drug_to_idx) & (targets["Gene"] != "")
    ].drop_duplicates(["STITCH", "Gene"])
    gene_ids = sorted(targets["Gene"].unique())
    if not gene_ids:
        raise ValueError("No drug-target features overlap the drugs in the combo dataset.")

    mono = pd.read_csv(
        mono_path,
        usecols=["STITCH", "Individual Side Effect"],
        dtype=str,
        keep_default_na=False,
    )
    mono["STITCH"] = mono["STITCH"].str.strip()
    mono["Individual Side Effect"] = mono["Individual Side Effect"].str.strip()
    mono = mono[
        mono["STITCH"].isin(drug_to_idx)
        & (mono["Individual Side Effect"] != "")
        & ~mono["Individual Side Effect"].isin(effect_to_idx)
    ].drop_duplicates(["STITCH", "Individual Side Effect"])
    mono_effect_ids = sorted(mono["Individual Side Effect"].unique())

    gene_to_idx = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}
    mono_effect_to_idx = {
        effect_id: idx + len(gene_ids) for idx, effect_id in enumerate(mono_effect_ids)
    }
    feature_mapping = [
        ("protein_target", gene_id) for gene_id in gene_ids
    ] + [
        ("monotherapy_side_effect", effect_id) for effect_id in mono_effect_ids
    ]
    drug_features = np.zeros((n_drugs, len(feature_mapping)), dtype=np.float32)
    target_drug_indices = targets["STITCH"].map(drug_to_idx).to_numpy(dtype=np.int64)
    target_gene_indices = targets["Gene"].map(gene_to_idx).to_numpy(dtype=np.int64)
    drug_features[target_drug_indices, target_gene_indices] = 1.0
    mono_drug_indices = mono["STITCH"].map(drug_to_idx).to_numpy(dtype=np.int64)
    mono_feature_indices = mono["Individual Side Effect"].map(
        mono_effect_to_idx
    ).to_numpy(dtype=np.int64)
    drug_features[mono_drug_indices, mono_feature_indices] = 1.0

    drugs_by_gene: dict[str, set[int]] = {}
    targets_by_gene = targets.assign(
        drug_index=targets["STITCH"].map(drug_to_idx)
    ).groupby("Gene")["drug_index"]
    for drug_indices in targets_by_gene:
        drugs_by_gene[drug_indices[0]] = set(drug_indices[1].astype(int).tolist())

    shared_target_edges: set[tuple[int, int]] = set()
    for members in drugs_by_gene.values():
        shared_target_edges.update(combinations(sorted(members), 2))

    if shared_target_edges:
        undirected_edges = np.asarray(sorted(shared_target_edges), dtype=np.int64)
        edge_index = torch.from_numpy(
            np.concatenate(
                (undirected_edges.T, undirected_edges[:, ::-1].T), axis=1
            ).copy()
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    ppi = pd.read_csv(
        ppi_path,
        usecols=["Gene 1", "Gene 2"],
        dtype=str,
        keep_default_na=False,
    )
    ppi["Gene 1"] = ppi["Gene 1"].str.strip()
    ppi["Gene 2"] = ppi["Gene 2"].str.strip()
    ppi = ppi[
        (ppi["Gene 1"] != "")
        & (ppi["Gene 2"] != "")
        & (ppi["Gene 1"] != ppi["Gene 2"])
    ].drop_duplicates()
    target_interaction_edges: set[tuple[int, int]] = set()
    for gene_a, gene_b in ppi.itertuples(index=False, name=None):
        for drug_a in drugs_by_gene.get(gene_a, ()):
            for drug_b in drugs_by_gene.get(gene_b, ()):
                if drug_a != drug_b:
                    target_interaction_edges.add(
                        (min(drug_a, drug_b), max(drug_a, drug_b))
                    )
    if target_interaction_edges:
        ppi_undirected_edges = np.asarray(
            sorted(target_interaction_edges), dtype=np.int64
        )
        ppi_edge_index = torch.from_numpy(
            np.concatenate(
                (ppi_undirected_edges.T, ppi_undirected_edges[:, ::-1].T), axis=1
            ).copy()
        )
    else:
        ppi_edge_index = torch.empty((2, 0), dtype=torch.long)

    return GraphSAGEDataset(
        node_features=torch.from_numpy(drug_features),
        feature_mapping=feature_mapping,
        edge_index=edge_index,
        ppi_edge_index=ppi_edge_index,
        drug_pairs=drug_pairs,
        label_indices=effect_indices,
        label_indptr=label_indptr,
        drug_ids=drug_ids,
        side_effect_ids=side_effect_ids,
        side_effect_names=effect_names,
    )


class GraphSAGEEncoder(nn.Module):
    def __init__(
        self, in_channels: int, hidden_channels: int = 128, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class SideEffectDecoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_side_effects: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 3, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_side_effects),
        )

    def forward(
        self, z: torch.Tensor, drug_pairs: torch.Tensor
    ) -> torch.Tensor:
        drug_u = z[drug_pairs[:, 0]]
        drug_v = z[drug_pairs[:, 1]]
        pair_features = torch.cat(
            (
                drug_u + drug_v,
                torch.abs(drug_u - drug_v),
                drug_u * drug_v,
            ),
            dim=1,
        )
        return self.mlp(pair_features)


class DrugSideEffectGraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_side_effects: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_channels, dropout)
        self.decoder = SideEffectDecoder(hidden_channels, num_side_effects, dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        drug_pairs: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encoder(x, edge_index)
        return self.decoder(z, drug_pairs)


def make_batch_labels(
    pair_indices: np.ndarray,
    label_indices: np.ndarray,
    label_indptr: np.ndarray,
    num_side_effects: int,
    device: torch.device,
) -> torch.Tensor:
    labels = torch.zeros(
        (len(pair_indices), num_side_effects), dtype=torch.float32, device=device
    )
    for row, pair_index in enumerate(pair_indices):
        start, end = label_indptr[pair_index : pair_index + 2]
        labels[row, label_indices[start:end]] = 1.0
    return labels


def get_split_indices(
    data: GraphSAGEDataset,
    split_mode: str,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    if split_mode == "random_pair":
        train_idx, val_idx, test_idx = split_pairs(
            len(data.drug_pairs), seed, val_ratio, test_ratio
        )
        return train_idx, val_idx, test_idx, {
            "excluded_cross_partition_pairs": 0,
            "candidate_drugs": len(data.drug_ids),
        }

    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Validation and test ratios must be positive and sum to less than 1.")
    rng = np.random.default_rng(seed)
    drug_order = rng.permutation(len(data.drug_ids))
    test_drug_count = max(1, int(round(len(drug_order) * test_ratio)))
    val_drug_count = max(1, int(round(len(drug_order) * val_ratio)))
    if test_drug_count + val_drug_count >= len(drug_order):
        raise ValueError("Validation and test ratios leave no training drugs.")

    groups = np.zeros(len(drug_order), dtype=np.int8)
    groups[drug_order[:test_drug_count]] = 2
    groups[drug_order[test_drug_count : test_drug_count + val_drug_count]] = 1
    pair_array = data.drug_pairs.numpy()
    pair_groups = groups[pair_array]
    train_mask = np.all(pair_groups == 0, axis=1)
    val_mask = np.all(pair_groups == 1, axis=1)
    test_mask = np.all(pair_groups == 2, axis=1)
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)
    test_idx = np.flatnonzero(test_mask)
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError(
            "Drug-disjoint split produced an empty pair partition; adjust split ratios or seed."
        )
    return train_idx, val_idx, test_idx, {
        "excluded_cross_partition_pairs": int(
            len(data.drug_pairs) - len(train_idx) - len(val_idx) - len(test_idx)
        ),
        "candidate_drugs": len(data.drug_ids),
        "train_drugs": int(np.count_nonzero(groups == 0)),
        "validation_drugs": int(np.count_nonzero(groups == 1)),
        "test_drugs": int(np.count_nonzero(groups == 2)),
    }


def select_features(
    data: GraphSAGEDataset, feature_set: str
) -> tuple[torch.Tensor, int]:
    if feature_set == "all":
        return data.node_features, data.node_features.size(1)
    selected = [
        idx
        for idx, (feature_type, _) in enumerate(data.feature_mapping)
        if (feature_set == "targets" and feature_type == "protein_target")
        or (
            feature_set == "monotherapy"
            and feature_type == "monotherapy_side_effect"
        )
    ]
    if not selected:
        raise ValueError(f"No node features are available for feature set {feature_set!r}.")
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    return data.node_features.index_select(1, selected_tensor), len(selected)


def count_positive_labels(
    data: GraphSAGEDataset, pair_indices: np.ndarray
) -> np.ndarray:
    counts = np.zeros(len(data.side_effect_ids), dtype=np.int64)
    for pair_index in pair_indices:
        start, end = data.label_indptr[pair_index : pair_index + 2]
        counts[data.label_indices[start:end]] += 1
    return counts


def predict_split(
    model: DrugSideEffectGraphSAGE,
    data: GraphSAGEDataset,
    pair_indices: np.ndarray,
    drug_pairs: torch.Tensor,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    total_bce = 0.0
    with torch.no_grad():
        embeddings = model.encoder(node_features, edge_index)
        for start in range(0, len(pair_indices), batch_size):
            batch_ids = pair_indices[start : start + batch_size]
            batch_pair_ids = torch.as_tensor(batch_ids, dtype=torch.long, device=device)
            labels = make_batch_labels(
                batch_ids,
                data.label_indices,
                data.label_indptr,
                len(data.side_effect_ids),
                device,
            )
            logits = model.decoder(embeddings, drug_pairs[batch_pair_ids])
            total_bce += F.binary_cross_entropy_with_logits(
                logits, labels, reduction="sum"
            ).item()
            all_labels.append(labels.cpu().numpy())
            all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return (
        np.concatenate(all_labels),
        np.concatenate(all_probabilities),
        total_bce / (len(pair_indices) * len(data.side_effect_ids)),
    )


def ranking_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    learnable_labels: np.ndarray | None = None,
    include_micro: bool = True,
) -> dict[str, float | int | None]:
    if learnable_labels is None:
        learnable_labels = np.ones(labels.shape[1], dtype=bool)
    if learnable_labels.shape != (labels.shape[1],):
        raise ValueError("learnable_labels must have one boolean per output label.")

    positive_counts = labels.sum(axis=0)
    has_both_classes = (
        learnable_labels
        & (positive_counts > 0)
        & (positive_counts < len(labels))
    )
    macro_auprc = (
        float(
            average_precision_score(
                labels[:, has_both_classes],
                probabilities[:, has_both_classes],
                average="macro",
            )
        )
        if np.any(has_both_classes)
        else None
    )
    macro_auroc = (
        float(
            roc_auc_score(
                labels[:, has_both_classes],
                probabilities[:, has_both_classes],
                average="macro",
            )
        )
        if np.any(has_both_classes)
        else None
    )
    result: dict[str, float | int | None] = {
        "macro_auprc": macro_auprc,
        "macro_auroc": macro_auroc,
        "labels_in_macro_auprc": int(np.count_nonzero(has_both_classes)),
        "labels_in_macro_auroc": int(np.count_nonzero(has_both_classes)),
        "learnable_label_count": int(np.count_nonzero(learnable_labels)),
    }
    if include_micro:
        micro_labels = labels[:, learnable_labels].ravel()
        micro_probabilities = probabilities[:, learnable_labels].ravel()
        if micro_labels.size and np.unique(micro_labels).size == 2:
            result["micro_auprc"] = float(
                average_precision_score(micro_labels, micro_probabilities)
            )
            result["micro_auroc"] = float(
                roc_auc_score(micro_labels, micro_probabilities)
            )
        else:
            result["micro_auprc"] = None
            result["micro_auroc"] = None
    return result


def select_validation_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    learnable_labels: np.ndarray | None = None,
) -> float:
    if learnable_labels is None:
        learnable_labels = np.ones(labels.shape[1], dtype=bool)
    positive_counts = labels.sum(axis=0)
    supported_labels = (
        learnable_labels
        & (positive_counts > 0)
        & (positive_counts < len(labels))
    )
    if not np.any(supported_labels):
        return 0.5

    best_threshold = 0.5
    best_macro_f1 = -1.0
    for threshold in np.arange(0.05, 1.0, 0.05):
        predictions = probabilities[:, supported_labels] >= threshold
        truth = labels[:, supported_labels].astype(bool)
        true_positive = np.logical_and(truth, predictions).sum(axis=0)
        false_positive = np.logical_and(~truth, predictions).sum(axis=0)
        false_negative = np.logical_and(truth, ~predictions).sum(axis=0)
        denominator = 2 * true_positive + false_positive + false_negative
        class_f1 = np.divide(
            2 * true_positive,
            denominator,
            out=np.zeros_like(true_positive, dtype=np.float64),
            where=denominator != 0,
        )
        macro_f1 = float(class_f1.mean())
        if (macro_f1, -abs(float(threshold) - 0.5)) > (
            best_macro_f1,
            -abs(best_threshold - 0.5),
        ):
            best_threshold = float(threshold)
            best_macro_f1 = macro_f1
    return best_threshold


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    learnable_labels: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    if learnable_labels is None:
        learnable_labels = np.ones(labels.shape[1], dtype=bool)
    result = ranking_metrics(labels, probabilities, learnable_labels)
    if not np.any(learnable_labels):
        result.update(
            {
                "threshold": threshold,
                "micro_f1": None,
                "macro_f1": None,
                "micro_precision": None,
                "micro_recall": None,
                "macro_precision": None,
                "macro_recall": None,
            }
        )
        return result
    eligible_labels = labels[:, learnable_labels]
    predictions = probabilities[:, learnable_labels] >= threshold
    result.update(
        {
            "threshold": threshold,
            "micro_f1": float(
                f1_score(eligible_labels, predictions, average="micro", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(eligible_labels, predictions, average="macro", zero_division=0)
            ),
            "micro_precision": float(
                precision_score(
                    eligible_labels, predictions, average="micro", zero_division=0
                )
            ),
            "micro_recall": float(
                recall_score(
                    eligible_labels, predictions, average="micro", zero_division=0
                )
            ),
            "macro_precision": float(
                precision_score(
                    eligible_labels, predictions, average="macro", zero_division=0
                )
            ),
            "macro_recall": float(
                recall_score(
                    eligible_labels, predictions, average="macro", zero_division=0
                )
            ),
        }
    )
    return result


def per_class_metrics(
    data: GraphSAGEDataset,
    train_positive_counts: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    test_probabilities: np.ndarray,
    threshold: float,
    threshold_eligible_labels: np.ndarray,
) -> list[dict[str, str | int | float | None]]:
    rows = []
    for label_idx, effect_id in enumerate(data.side_effect_ids):
        truth = test_labels[:, label_idx].astype(bool)
        scores = test_probabilities[:, label_idx]
        predictions = scores >= threshold
        positive_count = int(truth.sum())
        learnable = train_positive_counts[label_idx] > 0
        eligible = learnable and 0 < positive_count < len(truth)
        rows.append(
            {
                "side_effect_id": effect_id,
                "side_effect_name": data.side_effect_names[effect_id],
                "training_positive_pairs": int(train_positive_counts[label_idx]),
                "learnable_from_training": bool(learnable),
                "validation_positive_pairs": int(val_labels[:, label_idx].sum()),
                "threshold_tuning_eligible": bool(
                    threshold_eligible_labels[label_idx]
                ),
                "test_positive_pairs": positive_count,
                "test_auroc": (
                    float(roc_auc_score(truth, scores))
                    if eligible
                    else None
                ),
                "test_auprc": (
                    float(average_precision_score(truth, scores))
                    if eligible
                    else None
                ),
                "test_precision": float(
                    precision_score(truth, predictions, zero_division=0)
                ),
                "test_recall": float(recall_score(truth, predictions, zero_division=0)),
                "test_f1": float(f1_score(truth, predictions, zero_division=0)),
                "predicted_positive_pairs": int(predictions.sum()),
            }
        )
    return rows


def save_top_predictions(
    path: Path,
    data: GraphSAGEDataset,
    pair_indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    train_positive_counts: np.ndarray,
    top_k: int,
) -> None:
    rows = []
    k = min(top_k, len(data.side_effect_ids))
    for row_idx, pair_idx in enumerate(pair_indices):
        top_indices = np.argpartition(probabilities[row_idx], -k)[-k:]
        top_indices = top_indices[
            np.argsort(probabilities[row_idx, top_indices])[::-1]
        ]
        drug_u, drug_v = data.drug_pairs[pair_idx].tolist()
        for rank, label_idx in enumerate(top_indices, start=1):
            effect_id = data.side_effect_ids[int(label_idx)]
            rows.append(
                {
                    "drug_1": data.drug_ids[drug_u],
                    "drug_2": data.drug_ids[drug_v],
                    "rank": rank,
                    "side_effect_id": effect_id,
                    "side_effect_name": data.side_effect_names[effect_id],
                    "score": float(probabilities[row_idx, label_idx]),
                    "is_true": bool(labels[row_idx, label_idx]),
                    "training_positive_pairs": int(
                        train_positive_counts[label_idx]
                    ),
                    "learnable_from_training": bool(
                        train_positive_counts[label_idx] > 0
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def run_experiment(
    data: GraphSAGEDataset,
    args: argparse.Namespace,
    seed: int,
    split_mode: str,
    feature_set: str,
    graph_mode: str,
) -> dict[str, str | int | float | None]:
    seed_everything(seed)
    train_indices, val_indices, test_indices, split_details = get_split_indices(
        data, split_mode, seed, args.val_ratio, args.test_ratio
    )
    positive_counts = count_positive_labels(data, train_indices)
    zero_positive_indices = np.flatnonzero(positive_counts == 0)
    negative_counts = len(train_indices) - positive_counts
    val_positive_counts = count_positive_labels(data, val_indices)
    positive_weights = np.ones(len(data.side_effect_ids), dtype=np.float32)
    weighted_labels = (positive_counts > 0) & (negative_counts > 0)
    uncapped_weights = (
        negative_counts[weighted_labels] / positive_counts[weighted_labels]
    )
    if args.pos_weight_cap:
        positive_weights[weighted_labels] = np.clip(
            uncapped_weights, 1.0, args.pos_weight_cap
        )
    else:
        positive_weights[weighted_labels] = np.maximum(uncapped_weights, 1.0)
    learnable_labels = positive_counts > 0
    if not np.any(learnable_labels):
        raise ValueError("No side-effect labels have positive training examples.")
    threshold_eligible_labels = (
        learnable_labels
        & (val_positive_counts > 0)
        & (val_positive_counts < len(val_indices))
    )

    node_features_cpu, feature_count = select_features(data, feature_set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_features = node_features_cpu.to(device)
    if graph_mode == "shared_target":
        edge_index_cpu = data.edge_index
    elif graph_mode == "no_edge":
        edge_index_cpu = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index_cpu = data.ppi_edge_index
    edge_index = edge_index_cpu.to(device)
    drug_pairs = data.drug_pairs.to(device)
    positive_weights_tensor = torch.from_numpy(positive_weights).to(device)
    active_label_mask = torch.from_numpy(learnable_labels).to(device)
    model = DrugSideEffectGraphSAGE(
        in_channels=feature_count,
        hidden_channels=args.hidden_channels,
        num_side_effects=len(data.side_effect_ids),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(2, args.patience // 3),
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_weights_tensor, reduction="none"
    )

    run_name = (
        f"{split_mode}_seed{seed}_features-{feature_set}_graph-{graph_mode}"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_model.pt"
    best_val_auprc = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    print(
        f"\nRun {run_name} | device={device} | features={feature_count} | "
        f"pairs train/val/test={len(train_indices):,}/{len(val_indices):,}/"
        f"{len(test_indices):,} | zero-positive training labels="
        f"{len(zero_positive_indices)} | pos_weight_cap={args.pos_weight_cap or 'none'}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled_indices = np.random.permutation(train_indices)
        train_loss = 0.0
        for start in range(0, len(shuffled_indices), args.batch_size):
            batch_ids = shuffled_indices[start : start + args.batch_size]
            batch_pair_ids = torch.as_tensor(batch_ids, dtype=torch.long, device=device)
            labels = make_batch_labels(
                batch_ids,
                data.label_indices,
                data.label_indptr,
                len(data.side_effect_ids),
                device,
            )
            optimizer.zero_grad()
            logits = model(node_features, edge_index, drug_pairs[batch_pair_ids])
            loss_values = criterion(logits, labels)
            loss = loss_values[:, active_label_mask].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += loss.item() * len(batch_ids)
        train_loss /= len(shuffled_indices)

        val_labels, val_probabilities, val_bce = predict_split(
            model,
            data,
            val_indices,
            drug_pairs,
            node_features,
            edge_index,
            args.batch_size,
            device,
        )
        val_ranking = ranking_metrics(
            val_labels,
            val_probabilities,
            learnable_labels,
            include_micro=False,
        )
        val_auprc = val_ranking["macro_auprc"]
        if val_auprc is None or not np.isfinite(val_auprc):
            raise ValueError(
                f"Macro AUPRC is undefined for the {split_mode} validation split."
            )
        scheduler.step(val_auprc)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | train_weighted_bce={train_loss:.5f} | "
            f"val_bce={val_bce:.5f} | val_macro_auprc={val_auprc:.5f} | "
            f"val_macro_auroc={val_ranking['macro_auroc']} | lr={current_lr:.2g}"
        )
        if val_auprc > best_val_auprc + args.min_delta:
            best_val_auprc = float(val_auprc)
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    val_labels, val_probabilities, val_bce = predict_split(
        model,
        data,
        val_indices,
        drug_pairs,
        node_features,
        edge_index,
        args.batch_size,
        device,
    )
    test_labels, test_probabilities, test_bce = predict_split(
        model,
        data,
        test_indices,
        drug_pairs,
        node_features,
        edge_index,
        args.batch_size,
        device,
    )
    threshold = select_validation_threshold(
        val_labels, val_probabilities, learnable_labels
    )
    validation_metrics = classification_metrics(
        val_labels, val_probabilities, threshold, learnable_labels
    )
    validation_metrics["bce"] = val_bce
    test_metrics = classification_metrics(
        test_labels, test_probabilities, threshold, learnable_labels
    )
    test_metrics["bce"] = test_bce

    class_rows = per_class_metrics(
        data,
        positive_counts,
        val_labels,
        test_labels,
        test_probabilities,
        threshold,
        threshold_eligible_labels,
    )
    pd.DataFrame(class_rows).to_csv(run_dir / "per_class_metrics.csv", index=False)
    save_top_predictions(
        run_dir / "test_top10_predictions.csv",
        data,
        test_indices,
        test_labels,
        test_probabilities,
        positive_counts,
        args.top_k,
    )
    metrics = {
        "task": "known_pair_side_effect_prediction",
        "label_semantics": (
            "Only pairs present in bio-decagon-combo.csv are evaluated. "
            "Within those pairs, an unlisted effect is an unobserved association "
            "treated as negative by this benchmark, not a verified absence. "
            "Unrecorded drug pairs are excluded."
        ),
        "graph_construction": (
            "No-edge ablations use an empty edge list. Shared-target runs connect "
            "drugs sharing at least one protein target. PPI-target runs connect "
            "drugs whose protein targets interact in the PPI dataset. Biological "
            "edges are unweighted and bidirectional; combo labels are not graph edges."
        ),
        "drug_disjoint_protocol": (
            "This is transductive drug-disjoint evaluation, not strict inductive "
            "cold start. Each validation/test pair has both drugs assigned to its "
            "held-out drug partition. Cross-partition pairs are excluded; held-out "
            "drug features and target-similarity edges remain available to the encoder."
        ),
        "feature_ablation_scope": (
            "Feature-set runs select node-feature columns; graph topology is fixed "
            "within each graph-mode experiment."
        ),
        "split_mode": split_mode,
        "feature_set": feature_set,
        "graph_mode": graph_mode,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_macro_auprc": best_val_auprc,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "selected_threshold": threshold,
        "threshold_selection": (
            "Validation macro F1 over learnable labels with both positive and "
            "negative validation examples."
        ),
        "num_drugs": len(data.drug_ids),
        "num_side_effects": len(data.side_effect_ids),
        "total_side_effect_classes": len(data.side_effect_ids),
        "num_pairs": len(data.drug_pairs),
        "train_pairs": len(train_indices),
        "validation_pairs": len(val_indices),
        "test_pairs": len(test_indices),
        "graph_edges_directed": int(edge_index_cpu.size(1)),
        "graph_edges_undirected": int(edge_index_cpu.size(1) // 2),
        "graph_edge_index_bytes": int(
            edge_index_cpu.numel() * edge_index_cpu.element_size()
        ),
        "graph_max_degree": (
            int(
                torch.bincount(
                    edge_index_cpu[1], minlength=len(data.drug_ids)
                ).max()
            )
            if edge_index_cpu.size(1)
            else 0
        ),
        "split_details": split_details,
        "zero_positive_training_label_count": int(len(zero_positive_indices)),
        "zero_positive_training_labels": [
            data.side_effect_ids[idx] for idx in zero_positive_indices
        ],
        "evaluation_eligible_label_count": int(
            test_metrics["labels_in_macro_auprc"]
        ),
        "threshold_tuning_eligible_label_count": int(
            np.count_nonzero(threshold_eligible_labels)
        ),
        "loss_excluded_zero_positive_labels": True,
        "pos_weight_cap": args.pos_weight_cap or None,
        "uncapped_class_weight_max": (
            float(uncapped_weights.max()) if len(uncapped_weights) else 1.0
        ),
        "class_weight_max": float(positive_weights.max()),
        "device": str(device),
        "config": {
            "hidden_channels": args.hidden_channels,
            "dropout": args.dropout,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "pos_weight_cap": args.pos_weight_cap or None,
            "batch_size": args.batch_size,
            "gradient_clip_norm": args.grad_clip,
            "validation_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "seed": seed,
        },
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        f"Selected validation threshold={threshold:.2f} | "
        f"test macro AUPRC={test_metrics['macro_auprc']} | "
        f"test micro AUPRC={test_metrics['micro_auprc']} | "
        f"test macro AUROC={test_metrics['macro_auroc']} | "
        f"test micro AUROC={test_metrics['micro_auroc']} | "
        f"test macro F1={test_metrics['macro_f1']:.5f}"
    )
    return {
        "run": run_name,
        "split_mode": split_mode,
        "feature_set": feature_set,
        "graph_mode": graph_mode,
        "seed": seed,
        "best_epoch": best_epoch,
        "threshold": threshold,
        "validation_macro_auprc": validation_metrics["macro_auprc"],
        "test_macro_auprc": test_metrics["macro_auprc"],
        "test_micro_auprc": test_metrics["micro_auprc"],
        "test_macro_auroc": test_metrics["macro_auroc"],
        "test_micro_auroc": test_metrics["micro_auroc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_micro_f1": test_metrics["micro_f1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate GraphSAGE for Bio-DECAGON side effects."
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=script_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for repeated runs; overrides --seed.",
    )
    parser.add_argument(
        "--split-modes",
        nargs="+",
        choices=("random_pair", "drug_disjoint"),
        default=("random_pair", "drug_disjoint"),
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=("all", "targets", "monotherapy"),
        default=("all",),
        help="Run all, target-only, and/or monotherapy-only node-feature ablations.",
    )
    parser.add_argument(
        "--graph-modes",
        nargs="+",
        choices=("shared_target", "target_ppi", "no_edge"),
        default=("shared_target",),
        help="Compare shared-target, PPI-derived, and/or no-edge graphs.",
    )
    parser.add_argument(
        "--pos-weight-cap",
        type=float,
        default=100.0,
        help="Maximum positive BCE weight; use 0 for uncapped weights.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("--epochs, --batch-size, and --patience must be positive.")
    if args.hidden_channels < 1 or not 0 <= args.dropout < 1:
        parser.error("--hidden-channels must be positive and --dropout must be in [0, 1).")
    if args.lr <= 0 or args.weight_decay < 0:
        parser.error("--lr must be positive and --weight-decay cannot be negative.")
    if args.pos_weight_cap < 0 or 0 < args.pos_weight_cap < 1:
        parser.error("--pos-weight-cap must be 0 (uncapped) or at least 1.")
    if args.grad_clip <= 0 or args.min_delta < 0 or args.top_k < 1:
        parser.error("--grad-clip and --top-k must be positive; --min-delta cannot be negative.")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive.")

    if args.seeds:
        try:
            seeds = [int(value.strip()) for value in args.seeds.split(",")]
        except ValueError as exc:
            parser.error(f"--seeds must be comma-separated integers: {exc}")
        if not seeds or len(set(seeds)) != len(seeds):
            parser.error("--seeds must contain at least one unique integer.")
    else:
        seeds = [args.seed]
    if len(set(args.split_modes)) != len(args.split_modes):
        parser.error("--split-modes cannot contain duplicates.")
    if len(set(args.feature_sets)) != len(args.feature_sets):
        parser.error("--feature-sets cannot contain duplicates.")
    if len(set(args.graph_modes)) != len(args.graph_modes):
        parser.error("--graph-modes cannot contain duplicates.")

    data = load_data(args.data_dir, args.chunk_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"drug_index": range(len(data.drug_ids)), "drug_id": data.drug_ids}
    ).to_csv(args.output_dir / "drug_mapping.csv", index=False)
    pd.DataFrame(
        {
            "relation_index": range(len(data.side_effect_ids)),
            "side_effect_id": data.side_effect_ids,
            "side_effect_name": [
                data.side_effect_names[effect_id]
                for effect_id in data.side_effect_ids
            ],
        }
    ).to_csv(args.output_dir / "side_effect_mapping.csv", index=False)
    pd.DataFrame(
        {
            "feature_index": range(len(data.feature_mapping)),
            "feature_type": [feature_type for feature_type, _ in data.feature_mapping],
            "feature_id": [feature_id for _, feature_id in data.feature_mapping],
        }
    ).to_csv(args.output_dir / "node_feature_mapping.csv", index=False)

    summary = []
    for seed in seeds:
        for split_mode in args.split_modes:
            for feature_set in args.feature_sets:
                for graph_mode in args.graph_modes:
                    summary.append(
                        run_experiment(
                            data, args, seed, split_mode, feature_set, graph_mode
                        )
                    )
    pd.DataFrame(summary).to_csv(args.output_dir / "experiment_summary.csv", index=False)
    print(f"Saved experiment summary to {args.output_dir / 'experiment_summary.csv'}")


if __name__ == "__main__":
    main()