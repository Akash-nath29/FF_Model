from FF_NN import Trainer, NextTokenPredictor

vocab_size = 50
embed_dim = 10
hidden_dim = 20
learning_rate = 0.01
epochs = 10


sequences = [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]
targets = [5, 8, 11]


predictor = NextTokenPredictor(vocab_size, embed_dim, hidden_dim)
trainer = Trainer(predictor, learning_rate)
trainer.train(sequences, targets, epochs)

predicted_token, _ = predictor.predict([1, 2, 3, 4])
print("Predicted next token:", predicted_token)
