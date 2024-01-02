import torch
import os 
import glob

from model import DEVICE
from model import generate_square_subsequent_mask

from train import NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, \
                  NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM

from dataset import vocab_transform, text_transform, SRC_LANGUAGE, TGT_LANGUAGE
from dataset import BOS_IDX, EOS_IDX 

from model import Seq2SeqTransformer, DEVICE
transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
saved_model_path = sorted(glob.glob('trained_models/*.pt'), key=lambda x: float(x.split('valloss')[-1].split('.')[0]))[0]
load_epoch, load_loss = int(saved_model_path.split('epoch')[-1].split('_')[0]), float(saved_model_path.split('valloss')[-1].split('.')[0])
best_val_loss = {"epoch": load_epoch+1, "loss": load_loss}
transformer.load_state_dict(torch.load(saved_model_path))

transformer.to(DEVICE)

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

print(translate(transformer, "Eine Gruppe von Menschen steht vor einem Iglu ."))
