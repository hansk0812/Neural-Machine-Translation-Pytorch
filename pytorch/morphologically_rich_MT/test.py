import os
import glob

from dataset import EnTamV2Dataset, BucketingBatchSampler
from models.lstm import EncoderRNNLSTM, AttnDecoderRNNLSTM

import torch
from torch.utils.data import DataLoader

def test(dataloader, model):

    encoder, decoder = model
    
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        encoder_outputs, encoder_hidden = encoder(src)
        decoder_outputs, decoder_hidden, attn = decoder(encoder_outputs, encoder_hidden, max_length=tgt.shape[1], target_tensor=None)

        outputs = [dataloader.dataset.get_sentence_given_preds(x.detach().cpu().numpy()) for x in decoder_outputs]
        inputs = [dataloader.dataset.get_sentence_given_src(x.detach().cpu().numpy()) for x in src]

        for io in zip(inputs, outputs):
            print (io)

if __name__ == "__main__":
    
    word2vec_vector_size = 100
    hidden_size = 128
    device = torch.device("cuda")

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("load_epoch", type=int, help="Epoch to load checkpoint from", default=-1)
    ap.add_argument("--nosymbols", help="Flag to disable symbols from datasets", action="store_true")
    ap.add_argument("--verbose", help="Flag to print logs and stats", action="store_true")
    ap.add_argument("--morphemes", help="Flag to enable morphological analysis", action="store_true")
    ap.add_argument("--batch_size", type=int, help="Num sentences per batch", default=64)
    args = ap.parse_args()

    test_dataset = EnTamV2Dataset("test", symbols=not args.nosymbols, verbose=args.verbose, morphemes=args.morphemes)
    bucketing_batch_sampler = BucketingBatchSampler(test_dataset.bucketing_indices, batch_size=args.batch_size)
    dataloader = DataLoader(test_dataset, batch_sampler=bucketing_batch_sampler)
 
    encoder = EncoderRNNLSTM(word2vec_vector_size, hidden_size).to(device)
    decoder = AttnDecoderRNNLSTM(hidden_size, word2vec_vector_size, device=device).to(device)

    if args.load_epoch != -1:
        load_file_encoder = glob.glob(os.path.join("trained_models", "encoder_epoch%d_valloss*.pt" % args.load_epoch))[0]
        load_file_decoder = load_file_encoder.replace("encoder", "decoder")
        encoder.load_state_dict(torch.load(load_file_encoder))
        decoder.load_state_dict(torch.load(load_file_decoder))
        print ("Loaded model from ", load_file_encoder, load_file_decoder)

    with torch.no_grad():
        test(dataloader, (encoder, decoder))
