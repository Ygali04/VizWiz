from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout, LSTM, Flatten, Embedding, multiply
import h5py


#Taken from https://github.com/ranjaykrishna/iq with authorization from the author
#Read https://cs.stanford.edu/people/ranjaykrishna/iq/index.html if interested



import nltk
import re
import numpy as np
nltk.download('punkt')

class Vocabulary(object):

    def __init__(self):
        """Constructor for Vocabulary.
        """
        # Init mappings between words and ids
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')

    def add_word(self, word):
        if word not in self.word2idx:
          self.word2idx[word] = self.idx
          self.idx2word[self.idx] = word
          self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def save_vocab(self, location):
        with open(location, 'w') as f:
            json.dump({'word2idx': self.word2idx,
                       'idx2word': self.idx2word,
                       'idx': self.idx}, f)

    def load_vocab(self, location):
        with open(location, 'r') as f:
            data = json.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.idx = data['idx']

    def decode_sentence(self, tokens):
        words = []
        for token in tokens:
            word = self.idx2word[str(token)]
            if word == '<end>':
                break
            if word not in ['<pad>', '<start>', 
                            '<end>', '<unk>']:
                words.append(word)
        return ' '.join(words)

def load_vocab(vocab_path):
    vocab = Vocabulary()
    vocab.load_vocab(vocab_path)
    return vocab

def get_embeddings(path):
  matrix = h5py.File(path, 'r')
  return matrix['embedding_matrix'][()]

from keras.utils.np_utils import to_categorical

def prepare_data(data_path):
    data = h5py.File(data_path, 'r')

    questions = data['questions'][()]
    answers = to_categorical(data['answers'][()])
    return questions, answers

def encode_image(images_path, data_path):
  image_data = h5py.File(images_path, 'r')
  data = h5py.File(data_path, 'r')
  questions = data['questions'][()]
  image_features = image_data['feats'][()]
  answers = to_categorical(data['answers'][()])
  return image_features, questions, answers
