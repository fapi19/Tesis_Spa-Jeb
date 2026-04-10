from __future__ import annotations

import random
from torch.utils.data import Sampler


class NoDuplicateBatchSampler(Sampler[list[int]]):
    """
    Batch sampler simple que evita duplicados exactos de shiwilu o español
    dentro del mismo batch.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        rng = random.Random(self.seed)

        if self.shuffle:
            rng.shuffle(indices)

        batch = []
        seen_shw = set()
        seen_es = set()

        for idx in indices:
            row = self.dataset.rows[idx]
            shw = row["shiwilu"]
            es = row["spanish"]

            if shw in seen_shw or es in seen_es:
                continue

            batch.append(idx)
            seen_shw.add(shw)
            seen_es.add(es)

            if len(batch) == self.batch_size:
                yield batch
                batch = []
                seen_shw.clear()
                seen_es.clear()

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return max(1, (len(self.dataset) + self.batch_size - 1) // self.batch_size)