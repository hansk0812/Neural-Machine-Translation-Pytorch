import re
import glob
import os

from torch.utils.data import Dataset

import magic

import stanza
import quntoken

class EnHuDataset(Dataset):
    
    folders = ["classic.lit/bi/", "law/nonagg/1/bi/", "law/nonagg/2/", "modern.lit/bi/", "softwaredocs/bi/", "staging/", "subtitles/bi/"]

    def __init__(self, dataset_dir, max_vocab_size=150000):
        
        self.dataset_dir = dataset_dir
        self.max_vocab_size = max_vocab_size
        
        self.eng_tokenizer = stanza.Pipeline(lang='en', processors='tokenize', download_method=None)

        bilingual_files = []
        for folder in self.folders:
            bilingual_files.extend(glob.glob(os.path.join(dataset_dir, folder, '*.bi')))

        bilingual_pairs = []
        for fl in bilingual_files:
            blob = open(fl, 'rb').read()
            m = magic.Magic(mime_encoding=True)
            encoding = m.from_buffer(blob)
            
            if encoding == "binary" or "unknown" in encoding:
                rd = "rb"
                encoding=None
            else:
                rd = "r"
            with open(fl, rd, encoding=encoding) as f:
                data = [x.strip() for x in f.readlines()]
                if encoding is None:
                    bi = [str(x).split("\t") for x in data if len(str(x).split("\t"))==2]
                    bi = [(x[0].encode("iso-8859-1", errors="ignore"), x[1]) for x in bi]
                    if len(bi) == 0:
                        bi = [x.decode('iso-8859-1').split("\t") for x in data if len(str(x).split("\\t"))==2]
                else:
                    bi = [x.split('\t') for x in data if len(x.split('\t'))==2]
                for x in range(len(bi)):
                    matches = re.search(r"([\d\.]+)*", bi[x][0])
                    if matches.group(1) and not matches.group(1) in bi[x][1]:
                        bi[x][1] = matches.group(1) + " " + bi[x][1]
                    bi[x][0] = re.sub(r"^- ", r"", bi[x][0])
                    if isinstance(bi[x][1], bytes):
                        bi[x][1] = bi[x][1].decode('iso-8859-1', errors="ignore")
                bilingual_pairs.extend(bi)
        
        for hun, eng in bilingual_pairs:
            doc = self.eng_tokenizer(eng)
            assert len(doc.sentences) == 1
            sentence = doc.sentences[0]
            eng_tokens = [x.text for x in sentence.tokens]
            hun_tokens = quntoken.tokenize(hun)
            print ([x.split('\t')[0] for x in list(hun_tokens) if x!="\n"])
            print (hun)
            hun_tokens = hun.split(' ')
            
        # 3697273 pairs
        #print (len(bilingual_pairs))

if __name__ == "__main__":

    dataset = EnHuDataset(dataset_dir="dataset_hungarian")

