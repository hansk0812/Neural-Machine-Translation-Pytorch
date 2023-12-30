from data_preprocessing import MAX_SEQUENCE_LENGTH
from data_preprocessing import eng_tokenizer, spa_tokenizer
from data_preprocessing import test_pairs

from model import transformer

import tensorflow as tf
import keras_nlp
import random

def decode_sequences(input_sentences):
    batch_size = 1

    # Tokenize the encoder input.
    encoder_input_tokens = tf.convert_to_tensor(eng_tokenizer(input_sentences).to_tensor())
    if len(encoder_input_tokens[0]) < MAX_SEQUENCE_LENGTH:
        pads = tf.fill((1, MAX_SEQUENCE_LENGTH - len(encoder_input_tokens[0])), 0)
        encoder_input_tokens = tf.concat([encoder_input_tokens, pads], 1)

    # Define a function that outputs the next token's probability given the
    # input sequence.
    def next_fn(prompt, cache, index):
        logits = transformer([encoder_input_tokens, prompt])[:, index - 1, :]
        # Ignore hidden states for now; only needed for contrastive search.
        hidden_states = None
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
for i in range(2):
    input_sentence = random.choice(test_eng_texts)
    translated = decode_sequences([input_sentence])
    translated = translated.numpy()[0].decode("utf-8")
    translated = (
        translated.replace("[PAD]", "")
        .replace("[START]", "")
        .replace("[END]", "")
        .strip()
    )
    print(f"** Example {i} **")
    print(input_sentence)
    print(translated)
    print()

