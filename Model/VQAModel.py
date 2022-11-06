def tiny_language_model(embedding, vocab_size, q_lengths = 20, 
                   num_layers = 2, hidden_size=512, 
                   embedding_size=100):

  language_model = Sequential()
  language_model.add(Embedding(vocab_size, embedding_size, 
                    weights=[embedding], input_length=q_lengths, 
                    trainable=False))
  language_model.add(LSTM(hidden_size, return_sequences=False))
  return language_model
  
def tiny_image_model(hidden_size=512):
  image_model = Sequential()
  image_model.add(Reshape(input_shape = (4096,), target_shape=(4096,)))
  image_model.add(Dense(hidden_size, activation = 'tanh'))
  return image_model
  
def tiny_vqa_model(embedding, vocab_size, q_lengths = 20, 
              num_lstm_layers = 2, hidden_size=512, 
              embedding_size=100, num_classes=30):
  image_model_ = tiny_image_model(hidden_size)

  language_model_ = tiny_language_model(embedding, vocab_size, 
                                   q_lengths = q_lengths, 
                                   num_layers = num_lstm_layers, 
                                   hidden_size=hidden_size, 
                                   embedding_size=embedding_size)
  ### CHOOSE BETWEEN CONCATENATING OR MULTIPLYING LAYERS
  #combined = multiply([image_model_.output, language_model_.output])
  combined = concatenate([image_model_.output, language_model_.output])
  model = Dense(256, activation = 'tanh')(combined)
  model = Dropout(0.5)(model)

  model = Dense(num_classes)(model)
  model = Activation("softmax")(model)

  model = Model(inputs=[image_model_.input, language_model_.input], outputs=model)

  model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

  return model
  
 
tiny_vqa_model_ = tiny_vqa_model(embedding_matrix, len(vocab), q_lengths = 20, 
          num_lstm_layers = 2, hidden_size=512, embedding_size=100, num_classes=30)
plot_model(tiny_vqa_model_, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

val_features, val_questions, val_answers = encode_image(val_feats_path, val_path)
tiny_vqa_model_.fit([train_features, train_questions], train_answers, 
              validation_data=([val_features, val_questions], val_answers),
              epochs=10, batch_size=64, shuffle=True)
