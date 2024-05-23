import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.gru import EncoderDecoder

from data.utils import get_sentences_from_file, BucketingBatchSamplerReplace as BucketingBatchSampler
from bilingual_sets.entam import EnTam, BucketingBatchSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

train_dataset = EnTam("dataset/corpus.bcn.train.en", "dataset/corpus.bcn.train.ta", bucketing_language_sort="l2", cache_id=0, morphemes=True)
vocabs = train_dataset.return_vocabularies()
word2vecs = train_dataset.return_word2vecs()
val_dataset = EnTam("dataset/corpus.bcn.dev.en", "dataset/corpus.bcn.dev.ta", bucketing_language_sort="l2", vocabularies=vocabs, word2vecs=word2vecs, cache_id=1, morphemes=True)

bucketing_batch_sampler = BucketingBatchSampler(train_dataset.bucketer.bucketing_indices, batch_size=batch_size, verbose=True)
train_dataloader = DataLoader(train_dataset, batch_sampler=bucketing_batch_sampler)

bucketing_batch_sampler = BucketingBatchSampler(val_dataset.bucketer.bucketing_indices, batch_size=batch_size, verbose=True)
val_dataloader = DataLoader(val_dataset, batch_sampler=bucketing_batch_sampler)

PAD_idx = train_dataset.pad_idx_2

hidden_size = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(train_dataloader, model, n_epochs, learning_rate=0.0003):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD_idx)

    for epoch in range(1, n_epochs + 1):
        loss = 0
        for iter, batch in enumerate(train_dataloader):
            # Batch tensors: [B, SeqLen]
            input_tensor  = batch[0].to(device)
            input_mask    = batch[1].to(device)
            target_tensor = batch[2].to(device)
            loss += train_step(input_tensor, input_mask, target_tensor,
                               model, optimizer, criterion)
        print('Epoch {} Loss {}'.format(epoch, loss / iter))


def train_step(input_tensor, input_mask, target_tensor, model,
               optimizer, criterion):
    optimizer.zero_grad()
    decoder_outputs, decoder_hidden = model(input_tensor, input_mask, max_len=target_tensor.shape[1])

    # Collapse [B, Seq] dimensions for NLL Loss
    loss = criterion(
        decoder_outputs.view(-1, decoder_outputs.size(-1)), # [B, Seq, OutVoc] -> [B*Seq, OutVoc]
        target_tensor.view(-1) # [B, Seq] -> [B*Seq]
    )

    loss.backward()
    optimizer.step()
    return loss.item()

def ids2words(lang, ids):
    return [lang.index2word[idx] for idx in ids]

def greedy_decode(model, training_data_obj, dataloader):
    with torch.no_grad():
        batch = next(iter(dataloader))
        input_tensor  = batch[0]
        input_mask    = batch[1]
        target_tensor = batch[2]

        decoder_outputs, decoder_hidden = model(input_tensor, input_mask, max_len=target_tensor.shape[1])
        topv, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        for idx in range(input_tensor.size(0)):
            input_sent = train_dataset.indices_to_words(input_tensor[idx].cpu().numpy(), language='en')
            output_sent = train_dataset.indices_to_words(decoded_ids[idx].cpu().numpy(), language='ta')
            target_sent = train_dataset.indices_to_words(target_tensor[idx].cpu().numpy(), language='ta')
            print('Input:  {}'.format(input_sent))
            print('Target: {}'.format(target_sent))
            print('Output: {}'.format(output_sent))


if __name__ == '__main__':
    hidden_size = 256
    input_wordc = len(train_dataset.l1_vocab.sorted_tokens)
    output_wordc = len(train_dataset.l2_vocab.sorted_tokens)
    
    model = EncoderDecoder(hidden_size, input_wordc, output_wordc).to(device)
    
    train(val_dataloader, model, n_epochs=200)
    greedy_decode(model, train_dataset, val_dataloader)
