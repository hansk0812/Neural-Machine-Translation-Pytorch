import torch
from torch.utils.data import Dataset, Sampler

import numpy as np

def get_sentences_from_file(l1_path, l2_path):
    l1_sentences, l2_sentences = [], []

    with open(l1_path, 'r') as f:
        for line in f.readlines():
            l1_sentences.append(line)
    with open(l2_path, 'r') as f:
        for line in f.readlines():
            l2_sentences.append(line)
    
    return l1_sentences, l2_sentences

class BucketingBatchSamplerReplace(Sampler):
    def __init__(self, bucketing_indices, batch_size):
        self.bucketing_indices = bucketing_indices
        self.batch_size = batch_size

        length = sum([x[1]-x[0] for x in self.bucketing_indices])
        self.bucket_wt = [(x[1]-x[0])/float(length) for x in self.bucketing_indices]
        
    def __len__(self) -> int:
        return (self.bucketing_indices[-1][1] + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        bucket_idx = np.random.choice(len(self.bucketing_indices), p=self.bucket_wt)
        start, end = self.bucketing_indices[bucket_idx]
        yield start + np.random.choice(end-start, self.batch_size, replace=False)

#TODO: Batch sequence from lowest to biggest bucket
class BucketingBatchSampler(Sampler):
    def __init__(self, bucketing_indices, batch_size):
        self.bucketing_indices = bucketing_indices
        self.batch_size = batch_size

    def __len__(self) -> int:
        return (self.bucketing_indices[-1][1] + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for _ in range(len(self)):
            bucket_sample = torch.randint(low=0, high=len(self.bucketing_indices), size=(1,))
            start, end = self.bucketing_indices[bucket_sample]

            if end + 1 - start < self.batch_size:
                yield range(start, end+1)
            else:
                start_idx = torch.randint(low=start, high=end+1-self.batch_size, size=(1,))
                yield range(start_idx, start_idx+self.batch_size)

if __name__ == "__main__":

    batcher = BucketingBatchSamplerReplace(
                    bucketing_indices = [[0, 1000], [1001,1500], [1501,2000], [2001,5000], [5001,6000], [6001,7500], [7501,8000], [8001,9000], [9001,9500], [9501,10000]],
                    batch_size = 16)
    
    batcher = iter(batcher)
    while batcher:
        print (next(batcher))
