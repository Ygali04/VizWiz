def get_categories():
  answer_categories = {"binary": ['no', 'yes'],
    "color": ['white', 'grey', 'black', 'blue', 'red', 'pink', 'brown', 'green', 'purple', 'yellow', 'tan', 'orange'], 
    "food": ['coffee', 'soup', 'wine', 'food', 'coca cola', 'corn', 'pepsi'],
    "object":['keyboard', 'phone', 'laptop', 'nothing', 'lotion', 'shampoo', 'tv','cell phone'],
    "animal": ['dog']
    }
  print("We have %d categories:"%len(answer_categories.keys()))
  for cat in answer_categories.keys():
    print(cat, ':', answer_categories[cat])
  return answer_categories

def get_pred(model, val_features, val_questions, val_answers, answer):
  if answer is None:
    for i in range(30):
      a_index = np.argwhere(np.argmax(val_answers, axis=1) == i).reshape(-1)
      y_pred = np.argmax(model.predict([val_features[a_index], val_questions[a_index]]), axis =1)
      accuracy = sum(np.argmax(val_answers, axis=1)[a_index] == y_pred) / len(a_index)
      print('we have an accuracy of %f for answer %s (tested on %d)'%(accuracy, answer_list[i], len(a_index)))
    from sklearn.metrics import classification_report
    y_pred = np.argmax(model.predict([val_features, val_questions]), axis =1)
    print(classification_report(np.argmax(val_answers, axis=1) , y_pred))
  else:
    a_index = np.argwhere(np.argmax(val_answers, axis=1) == answer).reshape(-1)
    y_pred =np.argmax(model.predict([val_features[a_index], val_questions[a_index]]), axis =1)
    accuracy = sum(np.argmax(val_answers, axis=1)[a_index] == y_pred) / len(a_index)
    print('we have an accuracy of %f for answer %s (tested on %d)'%(accuracy, answer_list[answer], len(a_index)))

def get_prediction_stats(category, vqa_model, val_feats, val_questions, answers):
  if answer is None:
    for i in range(30):
      a_index = np.argwhere(np.argmax(val_answers, axis=1) == i).reshape(-1)
      y_pred = np.argmax(model.predict([val_features[a_index], val_questions[a_index]]), axis =1)
      accuracy = sum(np.argmax(val_answers, axis=1)[a_index] == y_pred) / len(a_index)
      print('we have an accuracy of %f for answer %s (tested on %d)'%(accuracy, answer_list[i], len(a_index)))
    from sklearn.metrics import classification_report
    y_pred = np.argmax(model.predict([val_features, val_questions]), axis =1)
    print(classification_report(np.argmax(val_answers, axis=1) , y_pred))
  else:
    a_index = np.argwhere(np.argmax(val_answers, axis=1) == answer).reshape(-1)
    y_pred =np.argmax(model.predict([val_features[a_index], val_questions[a_index]]), axis =1)
    accuracy = sum(np.argmax(val_answers, axis=1)[a_index] == y_pred) / len(a_index)
    print('we have an accuracy of %f for answer %s (tested on %d)'%(accuracy, answer_list[answer], len(a_index)))

def get_category_mapping(answers_list, answers_category):
  answer_map = {}
  for key in answers_category.keys():
    if not key in answer_map.keys():
      answer_map[key] = []
    answer_map[key] += [answers_list.index(ans) for ans in answer_categories[key]]
  return answer_map

def get_prediction_stats(category, model, val_feats, 
                         val_questions, answers_list, answers_category):
  answer_map = get_category_mapping(answers_list, answers_category)
  if category is None:
    for cat in answer_map.keys():
      a_index = []
      for c in answer_map[cat]:
        a_index += np.argwhere(np.argmax(val_answers, axis=1) == c).reshape(-1)
      y_pred = np.argmax(model.predict([val_features[a_index], val_questions[a_index]]), axis =1)
      accuracy = sum(np.argmax(val_answers, axis=1)[a_index] == y_pred) / len(a_index)
      print('\t We have an accuracy of %f for category %s (tested on %d)'%(accuracy, cat, len(a_index)))
  else:
    a_index = []
    for c in answer_map[category]:
      a_index += np.argwhere(np.argmax(val_answers, axis=1) == c).reshape(-1).tolist()
    y_pred = np.argmax(model.predict([val_features[a_index], val_questions[a_index]]), axis =1)
    accuracy = sum(np.argmax(val_answers, axis=1)[a_index] == y_pred) / len(a_index)
    print('We have an accuracy of %f for category %s (tested on %d)'%(accuracy, category, len(a_index)))
