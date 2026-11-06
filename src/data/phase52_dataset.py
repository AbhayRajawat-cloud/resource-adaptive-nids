"""
Stage 5: Phase 5.2 Dataset Loader & Safe Batching Module.

Loads Gate-passed datasets extracted in Phase 5.1 from:
  data/processed/gate_passed/{quantile}/client_{k:02d}_passed.npz

Features:
- Preserves 38-D feature matrices, ground-truth labels, and true original row indices.
- Preserves Tier 1, Tier 2, and Tier 3 lineage masks for test evaluation.
- Implements SafeBatchSampler: preserves 100% of samples while guaranteeing that no batch
  has size == 1 (merging single remainder with the penultimate batch), preventing BatchNorm crashes.
- Computes client-specific positive class weighting w_pos = min(10.0, N_benign / N_attack)
  with explicit safeguards for zero-attack or zero-benign cases.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Iterator
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader


def compute_positive_class_weight(
    n_benign: int,
    n_attack: int,
    max_weight: float = 10.0,
) -> Tuple[float, bool]:
    """
    Compute positive class loss weight according to specification:
      w_pos = min(10.0, N_benign_train / N_attack_train)
    
    Safeguards:
      If n_attack == 0 or n_benign == 0, returns fallback weight 1.0 and sets fallback_used=True.
      
    Returns:
        (w_pos, fallback_used)
    """
    if n_attack <= 0 or n_benign <= 0:
        return 1.0, True
    raw_ratio = float(n_benign) / float(n_attack)
    w_pos = min(float(max_weight), raw_ratio)
    return w_pos, False


class SafeBatchSampler(Sampler[List[int]]):
    """
    BatchSampler that groups indices into batches of size `batch_size`.
    If the final batch contains only 1 sample and there is a preceding batch,
    merges the final sample into the preceding batch (giving size batch_size + 1),
    ensuring 100% sample preservation with zero size-1 batches.
    """

    def __init__(
        self,
        data_source_len: int,
        batch_size: int = 32,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.data_source_len = int(data_source_len)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            indices = self.rng.permutation(self.data_source_len).tolist()
        else:
            indices = list(range(self.data_source_len))

        batches: List[List[int]] = []
        for i in range(0, len(indices), self.batch_size):
            batches.append(indices[i : i + self.batch_size])

        # If the last batch has exactly 1 sample and there is a preceding batch,
        # merge it into the preceding batch to prevent BatchNorm failure while preserving all samples.
        if len(batches) > 1 and len(batches[-1]) == 1:
            last_sample = batches.pop()
            batches[-1].extend(last_sample)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        if self.data_source_len == 0:
            return 0
        num_batches = (self.data_source_len + self.batch_size - 1) // self.batch_size
        remainder = self.data_source_len % self.batch_size
        if remainder == 1 and num_batches > 1:
            return num_batches - 1
        return num_batches


class GatePassedPartitionDataset(Dataset):
    """
    PyTorch Dataset wrapping a single partition (train, val, or test) of Gate-passed flows.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        orig_idx: np.ndarray,
        tier2_mask: Optional[np.ndarray] = None,
        tier3_mask: Optional[np.ndarray] = None,
        as_sequence_w1: bool = False,
    ):
        self.x = torch.tensor(x, dtype=torch.float32)
        if as_sequence_w1:
            # W=1: shape (N, 1, 38)
            self.x = self.x.unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.orig_idx = torch.tensor(orig_idx, dtype=torch.int64)

        self.tier2_mask = torch.tensor(tier2_mask, dtype=torch.bool) if tier2_mask is not None else None
        self.tier3_mask = torch.tensor(tier3_mask, dtype=torch.bool) if tier3_mask is not None else None

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "x": self.x[idx],
            "y": self.y[idx],
            "orig_idx": self.orig_idx[idx],
        }
        if self.tier2_mask is not None:
            item["tier2_mask"] = self.tier2_mask[idx]
        if self.tier3_mask is not None:
            item["tier3_mask"] = self.tier3_mask[idx]
        return item


def load_client_gate_passed_data(
    client_id: int,
    quantile: str = "Q99",
    base_dir: Path = Path("data/processed/gate_passed"),
    as_sequence_w1: bool = False,
) -> Tuple[GatePassedPartitionDataset, GatePassedPartitionDataset, GatePassedPartitionDataset, Dict[str, Any]]:
    """
    Load Gate-passed train, val, and test datasets along with verified metadata for a client.
    
    Returns:
        (train_dataset, val_dataset, test_dataset, metadata_dict)
    """
    cid_str = f"client_{client_id:02d}"
    npz_path = base_dir / quantile / f"{cid_str}_passed.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing Gate-passed dataset: {npz_path}")

    data = np.load(npz_path)

    # 1. Train dataset
    train_ds = GatePassedPartitionDataset(
        x=data["x_train_passed"],
        y=data["y_train_passed"],
        orig_idx=data["orig_idx_train_passed"],
        as_sequence_w1=as_sequence_w1,
    )

    # 2. Val dataset
    val_ds = GatePassedPartitionDataset(
        x=data["x_val_passed"],
        y=data["y_val_passed"],
        orig_idx=data["orig_idx_val_passed"],
        as_sequence_w1=as_sequence_w1,
    )

    # 3. Test dataset (with Tier 2 and Tier 3 lineage masks)
    test_ds = GatePassedPartitionDataset(
        x=data["x_test_passed"],
        y=data["y_test_passed"],
        orig_idx=data["orig_idx_test_passed"],
        tier2_mask=data["tier2_mask_test"],
        tier3_mask=data["tier3_mask_test"],
        as_sequence_w1=as_sequence_w1,
    )

    # 4. Metadata and class weight calculation
    n_benign_train = int(data["benign_count_train_passed"])
    n_attack_train = int(data["attack_count_train_passed"])
    w_pos, fallback_used = compute_positive_class_weight(n_benign_train, n_attack_train, max_weight=10.0)

    metadata = {
        "client_id": int(data["client_id"]),
        "client_str": cid_str,
        "quantile_name": str(data["quantile_name"]),
        "tau_gate": float(data["tau_gate"]),
        "n_train_passed": len(train_ds),
        "n_val_passed": len(val_ds),
        "n_test_passed": len(test_ds),
        "benign_count_train_passed": n_benign_train,
        "attack_count_train_passed": n_attack_train,
        "w_pos": w_pos,
        "w_pos_fallback_used": fallback_used,
        "tier1_test_count": int(data["tier1_test_count"]),
        "tier2_test_count": int(data["tier2_test_count"]),
        "tier3_test_count": int(data["tier3_test_count"]),
    }

    return train_ds, val_ds, test_ds, metadata


def create_safe_dataloader(
    dataset: GatePassedPartitionDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader wrapping SafeBatchSampler to prevent single-sample batches."""
    sampler = SafeBatchSampler(
        data_source_len=len(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(dataset, batch_sampler=sampler)

