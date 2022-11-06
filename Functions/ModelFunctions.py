import gdown
from keras.utils.np_utils import to_categorical


# file variables
train_feats_url         = 'https://drive.google.com/uc?id=1u485ehCP-ccAt3sSPwjCE0qCK-dOVsl5'
train_feats_path        = 'train_features.hdf5'

gdown.download(train_feats_url, 'train_features.hdf5', False)

# file variables

# file variables
val_feats_url         = 'https://drive.google.com/uc?id=1cdXPB49VuLr4ZbH-bwza9orlDddwDYOr'
val_feats_path        = 'val_features.hdf5'

gdown.download(val_feats_url, 'val_features.hdf5', False)



from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras import Input
from keras.layers.recurrent import LSTM
from keras.layers import concatenate

from keras.models import model_from_json, Model
from tensorflow.keras.utils import plot_model
from collections import defaultdict
import operator
from keras.utils import np_utils, generic_utils
from progressbar import Bar, ETA, Percentage, ProgressBar
from itertools import zip_longest
from keras.models import load_model

def image_model(hidden_size=512):
  image_model = Sequential()
  image_model.add(Reshape(input_shape = (4096,), target_shape=(4096,)))
  image_model.add(Dense(hidden_size, activation = 'tanh'))
  return image_model

def language_model(embedding, vocab_size, q_lengths = 20, 
                   num_layers = 2, hidden_size=512, 
                   embedding_size=100):

  language_model = Sequential()
  language_model.add(Embedding(vocab_size, embedding_size, 
                    weights=[embedding], input_length=q_lengths, 
                    trainable=False))
  language_model.add(LSTM(hidden_size, return_sequences=True, 
                          input_shape=(q_lengths, embedding_size)))
  for i in range(num_layers-2):
      language_model.add(LSTM(hidden_size, return_sequences=True))
  language_model.add(LSTM(hidden_size, return_sequences=False))

  return language_model

def vqa_model(embedding, vocab_size, q_lengths = 20, 
              num_lstm_layers = 2, hidden_size=512, 
              embedding_size=100, num_classes=30):
  image_model_ = image_model(hidden_size)

  language_model_ = language_model(embedding, vocab_size, 
                                   q_lengths = q_lengths, 
                                   num_layers = num_lstm_layers, 
                                   hidden_size=hidden_size, 
                                   embedding_size=embedding_size)
  combined = multiply([image_model_.output, language_model_.output])

  model = Dense(256, activation = 'tanh')(combined)
  model = Dropout(0.5)(model)

  model = Dense(256, activation = 'tanh')(model)
  model = Dropout(0.5)(model)

  model = Dense(128, activation = 'tanh')(model)
  model = Dropout(0.5)(model)

  model = Dense(num_classes)(model)
  model = Activation("softmax")(model)

  model = Model(inputs=[image_model_.input, language_model_.input], outputs=model)

  model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

  return model
