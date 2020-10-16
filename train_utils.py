import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
class BucketBatchSampler(Sampler):
        # want inputs to be an array
        def __init__(self, length, shuffle=True, bins=[32, 64, 128, 196, 256], batch_sizes=[128, 64, 32, 20, 16, 8]):
            self.batch_sizes = batch_sizes
            self.bins = bins
            self.shuffle = shuffle
            self.batch_map = defaultdict(list)

            buckets = np.digitize(length, bins)
            for idx, bucket in enumerate(buckets):
                self.batch_map[bucket].append(idx)
            self.batch_list = self._generate_batch_map()
            self.num_batches = len(self.batch_list)

        def _generate_batch_map(self):
            # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
            resid_list = []
            batch_list = []
            for bucket_idx, indices in sorted(self.batch_map.items(), key = lambda x : x[0]):
                batch_size = self.batch_sizes[bucket_idx]
                np.random.shuffle(indices)
                for group in [indices[i:(i + batch_size)] for i in range(0, len(indices), batch_size)]:
                    if len(group) == batch_size:
                        batch_list.append(group)
                    else:
                        resid_list += group
            for group in [resid_list[i:(i + self.batch_sizes[-1])] for i in range(0, len(resid_list), self.batch_sizes[-1])]:
                batch_list.append(group)
            return batch_list

        def batch_count(self):
            return self.num_batches

        def __len__(self):
            return len(self.batch_list)

        def __iter__(self):
            self.batch_list = self._generate_batch_map()
            self.num_batches = len(self.batch_list)
            # shuffle all the batches so they arent ordered by bucket size
            if self.shuffle:
                np.random.shuffle(self.batch_list)
            for i in self.batch_list:
                yield i

