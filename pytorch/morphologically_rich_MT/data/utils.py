def get_sentences_from_file(l1_path, l2_path):
    l1_sentences, l2_sentences = [], []

    with open(l1_path, 'r') as f:
        for line in f.readlines():
            l1_sentences.append(line)
    with open(l2_path, 'r') as f:
        for line in f.readlines():
            l2_sentences.append(line)
    
    return l1_sentences, l2_sentences
