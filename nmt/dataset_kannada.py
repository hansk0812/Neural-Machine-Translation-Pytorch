import random

from torch.utils.data import Dataset

class EnKnDataset(Dataset):

    en_data_dir = "dataset_kannada/train.en"
    kn_data_dir = "dataset_kannada/train.kn"
    
    reserved_tokens = ["UNK", "PAD", "START", "END", "NUM", "ENG"]
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, NUM_IDX, ENG_IDX = 0, 1, 2, 3, 4, 5

    def __init__(self, split="train"):

        assert split in ["train", "val", "test"]

        with open(self.en_data_dir, 'r') as f:
            en_sentences = [x.strip() for x in f.readlines()]

        with open(self.kn_data_dir, 'r') as f:
            kn_sentences = [x.strip() for x in f.readlines()]
        
        dataset = list(zip(en_sentences, kn_sentences))
        random.shuffle(dataset)

        splits = [0.75, 0.1, 0.15]
        train_idx, val_idx = int(len(dataset) * splits[0]), int(len(dataset) * (splits[0]+splits[1]))
        
        if split == "train":
            en_sentences = [x[0] for x in dataset][:train_idx]
            kn_sentences = [x[1] for x in dataset][:train_idx]
        elif split == "val":
            en_sentences = [x[0] for x in dataset][train_idx:val_idx]
            kn_sentences = [x[1] for x in dataset][train_idx:val_idx]
        else:
            en_sentences = [x[0] for x in dataset][val_idx:]
            kn_sentences = [x[1] for x in dataset][val_idx:]
        dataset = list(zip(en_sentences, kn_sentences))

        print ("%s set has %d sentences" % (split, len(en_sentences)))

        #en_sentences, kn_sentences = self.preprocess(en_sentences, kn_sentences)

if __name__ == "__main__":

    EnKnDataset("train")
    EnKnDataset("val")
    EnKnDataset("test")
