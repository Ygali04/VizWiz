def get_frequency(train_answers, X):
  a_index = np.argwhere(np.argmax(train_answers, axis=1) == X).reshape(-1)
  print("we have around %d answers"%len(a_index))


def get_stats(model, val_feats, val_questions, val_answers, X):
  a_index = np.argwhere(np.argmax(val_answers, axis=1) == X).reshape(-1)
  y_pred = np.argmax(model.predict([val_feats[a_index], val_questions[a_index]]), axis=1)
  accuracy = sum(np.argmax(val_answers, axis=1)[a_index] == y_pred) / len(a_index)
  print('\t We have an accuracy of %f for answer %d (tested on %d)'%(accuracy, X, len(a_index)))

def get_question_stats(train_questions, val_questions):
  train_idx, test_idx = [], []
  independent_train, independent_test = [], []
  for i, lst in enumerate(val_questions):
      if lst in train_questions:
          test_idx.append(i)
      else:
          independent_test.append(i)

  print("We have %d question in our validation set that the model has seen during training."%len(test_idx))
  return test_idx, independent_test

def print_bad_preds(model, vocab, val_features, val_questions, val_answers):
  y_pred = np.argmax(model.predict([val_features, val_questions]), axis=1)
  temp = np.argmax(val_answers, axis=1) == y_pred
  print(sum(temp))
  for i in range(len(temp)):
    if temp[i] == 0:
      print(vocab.decode_sentence(val_questions[i].astype(int)))
