import torch
from torch.utils.data import Dataset, Sampler

from .logger import Logger
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

class BucketingBatchSamplerReplace(Sampler, Logger):
    def __init__(self, bucketing_indices, batch_size, verbose):

        Logger.__init__(self, verbose)

        self.bucketing_indices = bucketing_indices
        self.batch_size = batch_size
        
        length = sum([x[1]-x[0] if x[1]-x[0] > 1 else 0 for x in self.bucketing_indices])
        self.bucket_wt = [1-((x[1]-x[0])/float(length)) for x in self.bucketing_indices]
        self.bucket_wt = np.array(self.bucket_wt)
        self.bucket_wt = np.exp(self.bucket_wt) / np.sum(np.exp(self.bucket_wt))
        self.print ('Sampling weights per bucket:') 
        self.print (self.bucket_wt)

    def __len__(self) -> int:
        return (self.bucketing_indices[-1][1] + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for _ in range(len(self)):
            bucket_idx = np.random.choice(len(self.bucketing_indices), p=self.bucket_wt)
            start, end = self.bucketing_indices[bucket_idx]
            replace = (end-start < self.batch_size)
            yd = start + np.random.choice(end-start, self.batch_size, replace=replace)
            yield yd

class BucketingBatchSamplerCurriculum(Sampler, Logger):
    def __init__(self, bucketing_indices, batch_size, verbose, curriculum_index = 0):

        # hyperparameter
        assert curriculum_index < 3, "Hardcoding number of training runs"
        
        Logger.__init__(self, verbose)

        self.bucketing_indices = bucketing_indices

        self.batch_size = batch_size
        self.curriculum_index = curriculum_index

    def __len__(self) -> int:
        # sort order assumed
        return (self.bucketing_indices[self.curriculum_index][1] + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for _ in range(len(self)):
            #bucket_idx = np.random.choice(len(self.bucketing_indices), p=self.bucket_wt)
            bucket_idx = self.curriculum_index
            start, end = self.bucketing_indices[bucket_idx]
            replace = (end-start < self.batch_size)
            yd = start + np.random.choice(end-start, self.batch_size, replace=replace)
            yield yd

"""
#TODO: Batch sequence from lowest to biggest bucket
class BucketingBatchSamplerValTest(Sampler):
    def __init__(self, bucketing_indices, batch_size):
        self.bucketing_indices = bucketing_indices
        self.batch_size = batch_size

        self.b_index, self.bucket_idx = 0, 0

    def __len__(self) -> int:
        return (self.bucketing_indices[-1][1] + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for idx in range(len(self)):

            bucket_sample = self.b_index
            start, end = self.bucketing_indices[bucket_sample]

            if end-start < self.batch_size:
                yield range(start, end+1)
                self.bucket_idx += 1
            else:
                for 
            self.total_idx = end+1

            if end + 1 - start < self.batch_size:
                yield range(start, end+1)
            else:
                start_idx = torch.randint(low=start, high=end+1-self.batch_size, size=(1,))
                yield range(start_idx, start_idx+self.batch_size)
"""

def remove_repetitions(sent):

    tokens = sent.split(' ')

    new_tokens = []
    # 1-word repetitions
    for idx in range(len(tokens)-1):
        if tokens[idx+1] == tokens[idx]:
            continue
        new_tokens.append(tokens[idx])
    new_tokens.append(tokens[len(tokens)-1])
    tokens = new_tokens
    
    print ("SENTENCE: ", " ".join(new_tokens))

    phrases = []
    # many-words repetitions
    for idx in range(len(tokens)):
        phrases.append(tokens[idx])
        
        if idx != 0:
            indices = [kdx for kdx, x in enumerate(phrases) if x.endswith(tokens[idx-1])]
            print ('indices', indices)
            for jdx in indices:
                phrases.append(phrases[jdx] + " " + tokens[idx])
                
        print (phrases)

    print (new_tokens)

if __name__ == "__main__":

    sent = "abc abc abc adsgf asf saf abc abc adg abc"
    
    print ("SENTENCE: ", sent)
    remove_repetitions(sent)
