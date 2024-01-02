from data_preprocessing import MAX_SEQUENCE_LENGTH
from data_preprocessing import eng_tokenizer, spa_tokenizer
from data_preprocessing import test_pairs

from model import transformer

import tensorflow as tf
import keras_nlp
import random

from nltk.translate.bleu_score import modified_precision

from train import MODEL_CHECKPOINT_DIR
from model import transformer
#transformer.load_weights(MODEL_CHECKPOINT_DIR)

import numpy as np

TEST_BATCH_SIZE = 1
"""
tensorflow.python.framework.errors_impl.InvalidArgumentError: Exception encountered when calling layer 'cross_attention' (type CachedMultiHeadAttention).

{{function_node __wrapped__Einsum_N_2_device_/job:localhost/replica:0/task:0/device:GPU:0}} Expected dimension 1 at axis 0 of the input shaped [64,40,8,32] but got dimension 64 [Op:Einsum] name: 

Call arguments received by layer 'cross_attention' (type CachedMultiHeadAttention):
  • query=tf.Tensor(shape=(64, 40, 256), dtype=float32)
  • value=tf.Tensor(shape=(1, 40, 256), dtype=float32)
  • key=None
  • attention_mask=None
  • cache=None
  • cache_update_index=None
"""

def decode_sequences(input_sentences):
    batch_size = TEST_BATCH_SIZE

    # Tokenize the encoder input.
    encoder_input_tokens = tf.convert_to_tensor(eng_tokenizer(input_sentences).to_tensor())
    if len(encoder_input_tokens[0]) < MAX_SEQUENCE_LENGTH:
        pads = tf.fill((1, MAX_SEQUENCE_LENGTH - len(encoder_input_tokens[0])), 0)
        encoder_input_tokens = tf.concat([encoder_input_tokens, pads], 1)
    
    # Cannot handle sequences with token length > 40
    if encoder_input_tokens.shape[-1] > 40:
        #print ('seg fault')
        encoder_input_tokens = encoder_input_tokens[:,:40]
    
    # Define a function that outputs the next token's probability given the
    # input sequence.
    def next_fn(prompt, cache, index):
        #print (encoder_input_tokens, prompt, index-1)
        #print ('seg fault:nextfn1')
        logits = transformer([encoder_input_tokens, prompt])
        #print ('seg fault:nextfn2')
        #print (logits.shape, index-1)
        logits = logits[:, index - 1, :]
        #print ('seg fault:nextfn3')
        # Ignore hidden states for now; only needed for contrastive search.
        hidden_states = None
        #print ('seg fault:nextfn4')
        print (logits.shape, [x.shape for x in cache], hidden_states)
        return logits, hidden_states, cache

    # Build a prompt of length 40 with a start token and padding tokens.
    length = 40
    start = tf.fill((batch_size, 1), spa_tokenizer.token_to_id("[START]"))
    pad = tf.fill((batch_size, length - 1), spa_tokenizer.token_to_id("[PAD]"))
    prompt = tf.concat((start, pad), axis=-1)

    generated_tokens = keras_nlp.samplers.GreedySampler()(
        next_fn,
        prompt,
        end_token_id=spa_tokenizer.token_to_id("[END]"),
        index=1,  # Start sampling after start token.
    )
    generated_sentences = spa_tokenizer.detokenize(generated_tokens)
    return generated_sentences


test_eng_texts = [pair[0] for pair in test_pairs]
test_spa_texts = [pair[1] for pair in test_pairs]

bleu_score = []
for i in range(len(test_pairs)):
    input_sentence = test_eng_texts[i]
    #print ('seg fault: input')
    translated = decode_sequences([input_sentence])
    #print ('seg fault: decoded')
    translated = translated.numpy()[0].decode("utf-8")
    translated = (
        translated.replace("[PAD]", "")
        .replace("[START]", "")
        .replace("[END]", "")
        .strip()
    )
    translated = translated.split(' ')
    bleu = modified_precision([test_spa_texts[i]], translated, n=4)
    bleu_score.append(bleu)

bleu_score_print = np.array(bleu_score)
print ("4-gram BLEU score: %f" % (bleu_score_print.mean()))
