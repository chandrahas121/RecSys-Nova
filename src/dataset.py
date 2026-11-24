"""
PyTorch Dataset for the NOVA-style sequential recommender.

Sequences are pre-computed in __init__ as contiguous numpy arrays.
Training negatives are re-drawn each epoch with a fast rejection loop.
Val/test negatives are drawn once in __init__ (so every evaluation pass scores
the same 100 candidates) and exclude every item the user interacted with.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class SeqRecDataset(Dataset):
    def __init__(self, splits: dict, mode: str,
                 num_items: int,
                 num_negatives: int = 4,
                 max_seq_len: int = 50,
                 seed: int = 42):
        assert mode in ('train', 'val', 'test')
        self.mode          = mode
        self.num_items     = num_items
        self.num_negatives = num_negatives
        self.rng           = np.random.default_rng(seed)

        # ── Collect raw triples ───────────────────────────────────────────────
        raw = []
        for uid, split in splits.items():
            history = split['train']
            if mode == 'train':
                for t in range(1, len(history)):
                    raw.append((uid, history[:t], history[t]))
            elif mode == 'val':
                tgt = split['val'][0] if split['val'] else None
                if tgt is not None:
                    raw.append((uid, split['train'], tgt))
            else:
                tgt = split['test'][0] if split['test'] else None
                if tgt is not None:
                    raw.append((uid, split['train'] + split['val'], tgt))

        n = len(raw)

        # ── Pre-allocate padded arrays ────────────────────────────────────────
        self.uids        = np.empty(n, dtype=np.int64)
        self.arr_iid     = np.zeros((n, max_seq_len), dtype=np.int64)
        self.arr_rat     = np.zeros((n, max_seq_len), dtype=np.int64)
        self.arr_tgap    = np.zeros((n, max_seq_len), dtype=np.int64)
        self.arr_unexp   = np.zeros((n, max_seq_len), dtype=np.float32)
        self.arr_mask    = np.zeros((n, max_seq_len), dtype=bool)
        self.target_iid  = np.empty(n, dtype=np.int64)

        for i, (uid, history, target) in enumerate(raw):
            self.uids[i] = uid
            T = min(len(history), max_seq_len)
            hist = history[-T:]                      # last T items
            pad  = max_seq_len - T

            self.arr_iid  [i, pad:] = [x[0] + 1 for x in hist]
            self.arr_rat  [i, pad:] = [int(round(x[1])) for x in hist]
            self.arr_tgap [i, pad:] = [x[3] for x in hist]
            self.arr_unexp[i, pad:] = [float(x[4]) for x in hist]
            self.arr_mask [i, pad:] = True
            self.target_iid[i]      = target[0] + 1

        # ── Fixed evaluation candidates ───────────────────────────────────────
        # Negatives must be items the user never interacted with in ANY split
        # (not just the last max_seq_len items), and must not change between
        # evaluation passes, or checkpoint selection on val becomes noisy.
        self.eval_negs = None
        if mode != 'train':
            self.eval_negs = np.empty((n, num_negatives), dtype=np.int64)
            for i, (uid, _, _) in enumerate(raw):
                seen = {x[0] + 1 for part in ('train', 'val', 'test')
                        for x in splits[uid][part]}
                self.eval_negs[i] = self._sample_unseen(seen, num_negatives)

    def _sample_unseen(self, seen: set, k: int) -> np.ndarray:
        """k distinct 1-indexed item ids that are not in `seen`."""
        if self.num_items - len(seen) < k:
            # Catalogue too small for k distinct negatives (toy data in
            # tests): fall back to sampling unseen items with replacement.
            pool = np.setdiff1d(np.arange(1, self.num_items + 1),
                                np.fromiter(seen, dtype=np.int64))
            return self.rng.choice(pool, size=k, replace=True)
        negs, picked = [], set()
        while len(negs) < k:
            for c in self.rng.integers(1, self.num_items + 1, size=k).tolist():
                if c not in seen and c not in picked:
                    picked.add(c)
                    negs.append(c)
                    if len(negs) == k:
                        break
        return np.asarray(negs, dtype=np.int64)

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        tgt_iid = int(self.target_iid[idx])

        if self.eval_negs is not None:
            negs = self.eval_negs[idx]
        else:
            # Bulk tolist() is faster than a generator comprehension for set build
            hist_iids = set(self.arr_iid[idx].tolist())
            hist_iids.discard(0)     # remove PAD token
            hist_iids.add(tgt_iid)

            # Rejection sampling — ~4.06 iterations on average for num_neg=4
            negs = []
            while len(negs) < self.num_negatives:
                n = int(self.rng.integers(1, self.num_items + 1))
                if n not in hist_iids:
                    negs.append(n)

        # np.ascontiguousarray is a no-op for already-contiguous slices
        return {
            'uid':        torch.tensor(int(self.uids[idx]), dtype=torch.long),
            'seq_iid':    torch.as_tensor(self.arr_iid  [idx]),
            'seq_rat':    torch.as_tensor(self.arr_rat  [idx]),
            'seq_tgap':   torch.as_tensor(self.arr_tgap [idx]),
            'seq_unexp':  torch.as_tensor(self.arr_unexp[idx]),
            'pad_mask':   torch.as_tensor(self.arr_mask [idx]),
            'target_iid': torch.tensor(tgt_iid,            dtype=torch.long),
            'neg_iids':   torch.tensor(negs,               dtype=torch.long),
        }
