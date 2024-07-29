import os
import keras_nlp
import keras

import tensorflow.data as tf_data
import tensorflow.strings as tf_strings

#Data
BATCH_SIZE = 64
MIN_STRING_LEN = 512 # Strings shorter than this will be discarded
SEQ_LEN = 128 #Length of training sequences, in tokens

# Models
EMBED_DIM = 256
FEED_FORWARD_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 2
VOCAB_SIZE = 5000 #Limits parameters in model

#Training
EPOCHS = 5

#Inference
NUM_TOKENS_TO_GENERATE = 80

#Load data
keras.utils.get_file(
    origin="https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip",
    extract=True,
)
dir = os.path.expanduser("~/.keras/datasets/simplebooks")

#Load simplebooks-92 train set and filter out short lines
raw_train_ds = (tf_data.TextLineDataset(dir + "/simplebooks-92-raw/train.txt")
.filter(lambda x: tf_strings.length(x) > MIN_STRING_LEN)
.batch(BATCH_SIZE)
.shuffle(buffer_size=256)
)

#Load simplebooks-92 validation set and filter out short lines
raw_val_ds = (
    tf_data.TextLineDataset(dir + "/simplebooks-92-raw/valid.txt")
    .filter(lambda x: tf_strings.length(x) > MIN_STRING_LEN)
    .batch(BATCH_SIZE)
)

#Train tokenizer vocabulary
vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
    raw_train_ds,
    vocabulary_size=VOCAB_SIZE,
    lowercase=True,
    reserved_tokens=["[PAD]", "[UNK]", "[BOS]"]
)

#Load tokenizer
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    sequence_length=SEQ_LEN,
    lowercase=True,
)

#Tokenize data
#packer adds a start token
start_packer = keras_nlp.layers.StartEndPacker(
    sequence_length=SEQ_LEN,
    start_value=tokenizer.token_to_id("[BOS]"),
)

def preprocess(inputs):
    outputs = tokenizer(inputs)
    features = start_packer(outputs)
    labels = outputs
    return features, labels

#Tokenize and split into train and label sequences
train_ds = raw_train_ds.map(preprocess, num_parallel_calls=tf_data.AUTOTUNE).prefetch(tf_data.AUTOTUNE)
val_ds = raw_val_ds.map(preprocess, num_parallel_calls=tf_data.AUTOTUNE).prefetch(tf_data.AUTOTUNE)

inputs = keras.layers.Input(shape=(None,), dtype="int32")
#Embedding
embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=SEQ_LEN,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)
x = embedding_layer(inputs)
#Transform decoders
for _ in range(NUM_LAYERS):
    decoder_layer = keras_nlp.layers.TransformerDecoder(
    num_heads = NUM_HEADS,
    intermediate_dim = FEED_FORWARD_DIM,
    )
    x = decoder_layer(x) #giving only one argument skips cross-attention
#Output
outputs = keras.layers.Dense(VOCAB_SIZE)(x)
model = keras.Model(inputs=inputs, outputs=outputs)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)
model.compile(optimizer="adam", loss=loss_fn, metrics=[perplexity])

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

#The "packer" layers adds the [BOS] token for us
prompt_tokens = start_packer(tokenizer([""]))
prompt_tokens

def next(prompt, cache, index):
    logits = model(prompt)[:, index - 1, :]
    #Ignore hidden states for now
    hidden_states = None
    return logits, hidden_states, cache

sampler = keras_nlp.samplers.TopPSampler(p=0.5)
output_tokens = sampler(
    next = next,
    prompt=prompt_tokens,
    index=1, #start sampling data immediately after the [BOS] token
)
txt = tokenizer.detokenize(output_tokens)
print(f"Top P search generated text:\n{txt}\n")