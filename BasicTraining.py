train_features, train_questions, train_answers = encode_image(train_feats_path, train_path)
vocab = load_vocab(vocab_path)
embedding_matrix = get_embeddings(embeddings_path)

vqa_model = vqa_model(embedding_matrix, len(vocab), q_lengths = 20, 
          num_lstm_layers = 2, hidden_size=512, embedding_size=100,
          num_classes=30)
plot_model(vqa_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

val_features, val_questions, val_answers = encode_image(val_feats_path, val_path)
vqa_model.fit([train_features, train_questions], train_answers, 
              validation_data=([val_features, val_questions], val_answers),
              epochs=10, batch_size=64, shuffle=True)
