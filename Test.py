import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?',
    'i really love my dog',
    'my dog loves my manatee'
]
training_size = 4
training_data = sentences[0:training_size]
test_data = sentences[training_size:]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(training_data)
word_index = tokenizer.word_index

train = tokenizer.texts_to_sequences(training_data)
test = tokenizer.texts_to_sequences(test_data)
padding = pad_sequences(train, padding='post')

print(test)
print(word_index)
print(train)
print(padding)
