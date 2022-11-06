X = 29

get_stats(vqa_model, val_features, val_questions, val_answers, X)
get_stats(vqa_model, train_features, train_questions, train_answers, X)

test_idx, independent_test = get_question_stats(train_questions, val_questions)

print_bad_preds(vqa_model, vocab, val_features, val_questions, val_answers)
