"""Dataset and collate function for denoising traces (Phase 3 distillation)."""

import os
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset, Sampler


class DenoisingTraceDataset(Dataset):
    """Dataset over saved denoising trace steps.

    Each item is a dict with:
        full_input_ids: [L] token ids (prefix + noisy block)
        mask_positions: [block_size] bool mask
        prefix_len: int
        sub_block_start: int
        sub_block_end: int
        target_ids: [block_size] target tokens
    """

    def __init__(self, trace_dir: str, block_size: int = 32,
                 block_filter: Optional[int] = None):
        self.block_size = block_size
        self.steps = []

        # Load all .pt trace files
        trace_files = sorted(
            f for f in os.listdir(trace_dir) if f.endswith('.pt')
        )
        for tf in trace_files:
            path = os.path.join(trace_dir, tf)
            data = torch.load(path, map_location='cpu', weights_only=True)
            if isinstance(data, list):
                self.steps.extend(data)
            elif isinstance(data, dict) and 'steps' in data:
                self.steps.extend(data['steps'])
            else:
                self.steps.append(data)

        print(f"[DenoisingTraceDataset] Loaded {len(self.steps)} steps "
              f"from {len(trace_files)} files in {trace_dir}")

        # Filter by block index if requested (for curriculum learning)
        if block_filter is not None:
            self.steps = [
                s for s in self.steps
                if (s['block_idx'].item() if isinstance(s['block_idx'], torch.Tensor)
                    else s['block_idx']) == block_filter
            ]
            print(f"[DenoisingTraceDataset] Filtered to block_idx={block_filter}: "
                  f"{len(self.steps)} steps")

        # Build length index for same-length batching
        self._length_to_indices = defaultdict(list)
        for i, step in enumerate(self.steps):
            ids = step.full_input_ids if hasattr(step, 'full_input_ids') else step['full_input_ids']
            self._length_to_indices[ids.shape[0]].append(i)

        lengths_summary = {k: len(v) for k, v in sorted(self._length_to_indices.items())}
        print(f"[DenoisingTraceDataset] Length distribution: {lengths_summary}")

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        step = self.steps[idx]

        if hasattr(step, 'full_input_ids'):
            # DenoisingStep dataclass
            d = {
                'full_input_ids': step.full_input_ids,
                'mask_positions': step.mask_positions,
                'prefix_len': torch.tensor(step.prefix_len, dtype=torch.long),
                'sub_block_start': torch.tensor(step.sub_block_start, dtype=torch.long),
                'sub_block_end': torch.tensor(step.sub_block_end, dtype=torch.long),
                'target_ids': step.target_ids,
                'unmask_this_step': step.unmask_this_step,
            }
        else:
            # Already a dict (from torch.save)
            d = step
        # Backward compat: old traces without unmask_this_step
        if 'unmask_this_step' not in d:
            d['unmask_this_step'] = d['mask_positions']
        return d

    @property
    def length_to_indices(self):
        return self._length_to_indices


class SameLengthBatchSampler(Sampler[List[int]]):
    """Yields batches where all samples have the same sequence length.

    Groups dataset indices by full_input_ids length, shuffles within each
    group, and interleaves batches across groups each epoch.
    """

    def __init__(self, dataset: DenoisingTraceDataset, batch_size: int,
                 rank: int = 0, world_size: int = 1):
        self.length_to_indices = dataset.length_to_indices
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[List[int]]:
        all_batches = []
        for length, indices in self.length_to_indices.items():
            shuffled = indices.copy()
            random.shuffle(shuffled)
            for i in range(0, len(shuffled), self.batch_size):
                all_batches.append(shuffled[i:i + self.batch_size])
        random.shuffle(all_batches)
        # Each rank takes every world_size-th batch.
        # Truncate to equal count so DDP AllReduce stays in sync.
        per_rank = len(all_batches) // self.world_size
        rank_batches = all_batches[self.rank::self.world_size][:per_rank]
        return iter(rank_batches)

    def __len__(self) -> int:
        total = 0
        for indices in self.length_to_indices.values():
            total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total // self.world_size


def same_length_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Collate same-length denoising trace steps by stacking directly.

    All samples must have the same full_input_ids length (guaranteed by
    SameLengthBatchSampler). No padding is applied.
    """
    return {
        'full_input_ids': torch.stack([item['full_input_ids'] for item in batch]),
        'mask_positions': torch.stack([item['mask_positions'] for item in batch]),
        'prefix_len': torch.stack([item['prefix_len'] for item in batch]),
        'sub_block_start': torch.stack([item['sub_block_start'] for item in batch]),
        'sub_block_end': torch.stack([item['sub_block_end'] for item in batch]),
        'target_ids': torch.stack([item['target_ids'] for item in batch]),
        'unmask_this_step': torch.stack([item['unmask_this_step'] for item in batch]),
    }
