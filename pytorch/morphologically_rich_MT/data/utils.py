import torch
from torch.utils.data import Dataset, Sampler

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

        self.batches = []
        for b_idx in range(len(bucketing_indices)):
            st, en = bucketing_indices[b_idx]

            if en - st < batch_size:
                self.batches.append((st, en))
            else:
                bucket_batches = [(x, x+batch_size) for x in range(st, en-batch_size+1)]
                self.batches.extend(bucket_batches)
        
        self.sampler = torch.randperm(len(self.batches))

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        for sample_idx in self.sampler:
            yield range(self.batches[sample_idx][0], self.batches[sample_idx][1])

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
