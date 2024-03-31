from logger import Logger

class Bucketing(Logger):

    def __init__(self, bilingual_pairs, buckets, sort_order, pad_token="PAD", verbose=False):

        super().__init__(verbose)

        self.bilingual_pairs = bilingual_pairs
        self.buckets = buckets

        self.bucket_complete = False
        self.bucketing_indices = self.return_bucketed_pairs(sort_order=sort_order, pad_token=pad_token)
        self.bucket_complete = True

    def return_bucketed_pairs(self, sort_order="l2", pad_token="PAD"):

        if self.bucket_complete:
            self.print("Abusing call stack! Call return_bucketed_pairs once only!")
            return self.bucketing_indices

        assert sort_order in ["l1", "l2"]
        sort_index = int(sort_order == "l2")

        for idx in range(len(self.bilingual_pairs)):
            l1, l2 = self.bilingual_pairs[idx]
            l1_tokens, l2_tokens = l1.split(' '), l2.split(' ')

            L1, L2 = len(l1_tokens), len(l2_tokens)

            # clip all tokens after buckets[-1] words
            if L1 > self.buckets[-1][0]:
                L1_tokens = L1_tokens[:self.buckets[-1][0]]
            if L2 > self.buckets[-1][1]:
                L2_tokens = L2_tokens[:self.buckets[-1][1]]
                
            for bucket_idx in range(len(self.buckets)):
                if (self.buckets[bucket_idx][1] < L2 and sort_index) or \
                   (self.buckets[bucket_idx][0] < L1 and not sort_index):
                    continue
                if (self.buckets[bucket_idx][0] >= L1 and sort_index) or \
                   (self.buckets[bucket_idx][1] >= L2 and not sort_index):
                    break

            l1_tokens = l1_tokens + [pad_token] * (self.buckets[bucket_idx][0] - L1)
            l2_tokens = l2_tokens + [pad_token] * (self.buckets[bucket_idx][1] - L2)

            self.bilingual_pairs[idx][0] = " ".join(l1_tokens)
            self.bilingual_pairs[idx][1] = " ".join(l2_tokens)
        
        self.bilingual_pairs = sorted(self.bilingual_pairs, key=lambda x: len(x[sort_index]))
        self.bilingual_pairs = [x for x in self.bilingual_pairs if x[0] != "" and x[1] != ""]
        
        bucketing_indices, b_idx, start_idx = [], 0, 0

        for idx in range(len(self.bilingual_pairs)):
            if self.buckets[b_idx][sort_index] == len(self.bilingual_pairs[idx][sort_index].split(' ')):
                continue
            else:
                b_idx += 1
                bucketing_indices.append((start_idx, idx-1))
                start_idx = idx
        bucketing_indices.append((start_idx, idx-1))

        return bucketing_indices

if __name__ == "__main__":

    bilingual_pairs = [
                        ["w1 w2 w3", "w1"],
                        ["w1 w2 w3", "w1 w2 w3 w4"],
                        ["w1 w2 w3 w4 w5 w6 w7 w8", "w1"],
                        ["w1", "w1 w2 w3 w4"]
            ]
    bucketer = Bucketing(bilingual_pairs, buckets=[[3,3],[5,5],[10,10]], sort_order="l2", verbose=True)
    print (bucketer.bucketing_indices, bucketer.bilingual_pairs)
