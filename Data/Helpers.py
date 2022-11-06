import json
import random

def get_annotations(annotations_path):
    with open(annotations_path, 'r') as f:
      metadata = json.load(f)
    return metadata

def get_random_annotation(annotations_path):
    with open(annotations_path, 'r') as f:
      metadata = json.load(f)
    i = random.choice(range(len(metadata)))
    return metadata[i]

  
 
#Obtain Data
annotations_path = './Annotations/val.json'
vocab_path = 'vocab.json'

def get_questions(annotations_path):
  
  with open(annotations_path, 'r') as f:
    annos = json.load(f)

  questions = set()
  for i in range(len(annos)):
    if annos[i]['answerable'] == 0:
      continue
    question = annos[i]['question']
    questions.add(question)

  return questions
  
from collections import Counter 
  
def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0] 

def get_answers(annotations_path):
  with open(annotations_path, 'r') as f:
    annos = json.load(f)
  answers = set()
  for i in range(len(annos)):
    if annos[i]['answerable'] == 0:
      continue
    ans = annos[i]['answers']
    answers_list = []
    for a in ans:
      answers_list += [a['answer']]
    answers.add(most_frequent(answers_list))
  return answers

def get_image_paths(annotations_path):
  with open(annotations_path, 'r') as f:
    annos = json.load(f)

  images = set()
  for i in range(len(annos)):
    if annos[i]['answerable'] == 0:
      continue
    image = annos[i]['image']
    images.add(image)

  return images

def parse_vocab(q, a):
  vocab = set()
  for qu in q:
    if '?' in qu:
      qu = qu.replace('?', '')

    vocab |= set(qu.split(' '))
  return vocab



#Plot Data
import h5py
import matplotlib.pyplot as plt

def get_data(path):
    data = h5py.File(path, 'r')
    images = data['images']
    image_idx = data['image_indices']
    questions = data['questions']
    answers = data['answers']
    return images, image_idx, questions, answers

def plot_image(img, questions=None, answers=None, indices=None):
    if indices is None:
      plt.imshow(img/255)
      plt.show()
    else:
      for i in indices:
        print('Question: ', questions[i])
        print('Answer: ', answers[i])
        plt.imshow(img[i]/255)
        plt.show()
