from tokenization import Tokenizer
from dataprocessing import Splitter

seq_length = 20

# Data Loading

with open('input.txt', 'r') as file:
    data = file.read()
    
# Tokenization

tokenizer = Tokenizer(data[:10000])
tokens = tokenizer.encode()

# Data Splitting
splitter = Splitter(tokens, seq_len=seq_length)

X, y = splitter.split()

X_train, X_test, y_train, y_test = splitter.train_test_split(X, y, test_size=0.2)

# print(X_train[:10], y_train[:10], X_test[:10], y_test[:10])

print(len(X_train), len(X_test), len(y_train), len(y_test))
print(tokenizer.decode(X_train[:10]))
print('-------------------')
print(tokenizer.decode(X_test[:10]))
print('-------------------')
print(tokenizer.decode(y_train[:10]))
print('-------------------')
print(tokenizer.decode(y_test[:10]))