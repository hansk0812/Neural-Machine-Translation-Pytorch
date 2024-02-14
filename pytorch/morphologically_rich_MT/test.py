import os
import glob

from dataset import EnTamV2Dataset, BucketingBatchSampler
#from models.lstm import EncoderRNNLSTM, AttnDecoderRNNLSTM
from models.lstm_classifier import EncoderRNN, AttnDecoderRNN

import torch
from torch.utils.data import DataLoader

def test(dataloader, model, train_dataset):

    encoder, decoder = model
    
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        #encoder_outputs, encoder_hidden = encoder(src)
        #decoder_outputs, decoder_hidden, attn = decoder(encoder_outputs, encoder_hidden, max_length=tgt.shape[1], target_tensor=None)
        
        encoder_outputs, encoder_hidden = encoder(src)
        # don't use teacher forcing
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_length=tgt.shape[1], target_tensor=None)
        
        outputs = [train_dataset.get_sentence_given_preds(x.detach().cpu().numpy()) for x in decoder_outputs]
        inputs = [train_dataset.get_sentence_given_src(x.detach().cpu().numpy()) for x in src]

        for io in zip(inputs, outputs):
            print (io)

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("load_epoch", type=int, help="Epoch to load checkpoint from", default=-1)
    ap.add_argument("--nosymbols", help="Flag to disable symbols from datasets", action="store_true")
    ap.add_argument("--verbose", help="Flag to print logs and stats", action="store_true")
    ap.add_argument("--morphemes", help="Flag to enable morphological analysis", action="store_true")
    ap.add_argument("--no_start_stop", help="Flag to remove START and STOP tokens", action="store_true")
    ap.add_argument("--batch_size", type=int, help="Num sentences per batch", default=64)
    args = ap.parse_args()

    train_dataset = EnTamV2Dataset("train", symbols=not args.nosymbols, verbose=args.verbose, 
                                   morphemes=args.morphemes, start_stop_tokens=not args.no_start_stop)
    eng_vocab, tam_vocab = train_dataset.return_vocabularies()
    test_dataset = EnTamV2Dataset("test", symbols=not args.nosymbols, verbose=args.verbose, 
                                  morphemes=args.morphemes, vocabularies=(eng_vocab, tam_vocab), 
                                  start_stop_tokens=not args.no_start_stop)
    
    bucketing_batch_sampler = BucketingBatchSampler(test_dataset.bucketing_indices, batch_size=args.batch_size)
    dataloader = DataLoader(test_dataset, batch_sampler=bucketing_batch_sampler)
    
    INPUT_SIZE = train_dataset.eng_embedding.shape[0]
    HIDDEN_DIM = train_dataset.eng_embedding.shape[1]
    OUTPUT_SIZE = train_dataset.tam_embedding.shape[0]
 
    #encoder = EncoderRNNLSTM(INPUT_SIZE, HIDDEN_DIM).to(device)
    #decoder = AttnDecoderRNNLSTM(HIDDEN_DIM, OUTPUT_SIZE, device=device).to(device)

    encoder = EncoderRNN(INPUT_SIZE, HIDDEN_DIM).to(device)
    decoder = AttnDecoderRNN(HIDDEN_DIM, OUTPUT_SIZE, device=device).to(device)
    
    if args.load_epoch != -1:
        load_file_encoder = glob.glob(os.path.join("trained_models", "encoder_epoch%d_valloss*.pt" % args.load_epoch))[0]
        load_file_decoder = load_file_encoder.replace("encoder", "decoder")
        
        if torch.cuda.is_available():
            encoder.load_state_dict(torch.load(load_file_encoder))
            decoder.load_state_dict(torch.load(load_file_decoder))
        else:
            encoder.load_state_dict(torch.load(load_file_encoder, map_location=device))
            decoder.load_state_dict(torch.load(load_file_decoder, map_location=device))
        print ("Loaded model from ", load_file_encoder, load_file_decoder)

    with torch.no_grad():
        test(dataloader, (encoder, decoder), train_dataset)
