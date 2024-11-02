from FF_NN import FeedForwardNetwork
from tokenization import Tokenizer
from dataprocessing import Splitter

seq_length = 20
with open('input.txt', 'r') as file:
    data = file.read()

tokenizer = Tokenizer(data[:10000])
tokens = tokenizer.encode()

splitter = Splitter(tokens, seq_len=seq_length)
X, y = splitter.split()
X_train, X_test, y_train, y_test = splitter.train_test_split(
    X, y, test_size=0.2)

embedding_dim = 50
vocab_size = len(tokenizer.preprocess())
model = FeedForwardNetwork(vocab_size, embedding_dim)

num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        model.update_weights(X_batch, y_batch)

    print(f'Epoch {epoch + 1}/{num_epochs} completed.')

sample_input = X_test[:1]
predicted_probs = model.predict(sample_input[0])
predicted_token = predicted_probs.index(max(predicted_probs))

print(f'Predicted token: {predicted_token} -> {tokenizer.decode([predicted_token])}')
