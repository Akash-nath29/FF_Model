class FeedForwardNetwork:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weights_input = [[0.01 for _ in range(embedding_dim)] for _ in range(vocab_size)]
        self.weights_hidden = [[0.01 for _ in range(embedding_dim)] for _ in range(embedding_dim)]
        self.weights_output = [[0.01 for _ in range(vocab_size)] for _ in range(embedding_dim)]
    
    def embed(self, token):
        return self.weights_input[token]
    
    def predict(self, inputs):
        hidden_layer = [0] * self.embedding_dim
        
        for token in inputs:
            embedded_token = self.embed(token)
            for i in range(self.embedding_dim):
                hidden_layer[i] += embedded_token[i]

        hidden_layer = [max(0, x) for x in hidden_layer]

        exp_scores = [pow(2.718, x) for x in hidden_layer]
        sum_exp_scores = sum(exp_scores)
        probabilities = [score / sum_exp_scores for score in exp_scores]

        return probabilities

    def update_weights(self, X_batch, y_batch, learning_rate=0.01):
        for x, y in zip(X_batch, y_batch):
            predicted_probs = self.predict(x)
            predicted_token = predicted_probs.index(max(predicted_probs))

            if y < 0 or y >= self.vocab_size:
                print(f"Warning: Target token {y} is out of bounds.")
                continue

            for token in x:
                if token < 0 or token >= self.vocab_size:
                    print(f"Warning: Input token {token} is out of bounds.")
                    continue

                for i in range(self.embedding_dim):
                    self.weights_output[i][predicted_token] += learning_rate * (1 - predicted_probs[predicted_token])
                    
                    if predicted_token != y:
                        self.weights_output[i][y] -= learning_rate * predicted_probs[y]

            for token in x:
                if token < 0 or token >= self.vocab_size:
                    print(f"Warning: Input token {token} is out of bounds.")
                    continue
                for i in range(self.embedding_dim):
                    self.weights_input[token][i] += learning_rate * (1 - predicted_probs[y])

