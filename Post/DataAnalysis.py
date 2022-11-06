import gdown
from collections import Counter

filter_url         = 'https://drive.google.com/uc?id=1Jc4Nff-5oPrjc3dwdB4eI8NZjiWT1VTJ'
answers_list_path        = 'filter.json'

gdown.download(filter_url, 'filter.json', True)

def load_answer(path):
  return json.load(open(path, 'r'))

def data_stats(answers_data, answers_list):
  a = [answers_data[i].argmax() for i in range(answers_data.shape[0])]
  c = Counter(a)
  for i, cnt in c.items():
    print(answer_list[i],' : ', cnt, '(label %d)'%i)

  print('\n \n')

  answers_x = range(len(answer_list))
  a = [answers_data[i].argmax() for i in range(answers_data.shape[0])]
  c = Counter(a)
  freq = [0]*len(answers_x)
  for i, cnt in c.items():
    freq[i] = cnt

  LABELS = answer_list

  plt.bar(answers_x, freq, align='center')
  plt.show()
  
answer_list = load_answer(answers_list_path)
print(answer_list)

data_stats(train_answers, answer_list)

data_stats(val_answers, answer_list)
