from functions import *

class Trainer:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate
    
    def cross_entropy_loss(self, logits, target_index):
        probs = softmax(logits)
        loss = -1 * (2.71828 ** (probs[target_index]))
        return loss, probs
    
    def backpropagate(self, input_representation, hidden_layer_output, output_probs, target_index):
        delta_output = output_probs
        delta_output[target_index] -= 1

        dW2 = [[hidden_layer_output[i] * delta_output[j] for j in range(len(delta_output))] for i in range(len(hidden_layer_output))]
        db2 = delta_output

        delta_hidden = [1 - hidden_layer_output[i] ** 2 for i in range(len(hidden_layer_output))]  # Tanh derivative
        dW1 = [[input_representation[i] * delta_hidden[j] for j in range(len(delta_hidden))] for i in range(len(input_representation))]
        db1 = delta_hidden

        self.model.W1 = [[w - self.learning_rate * dw for w, dw in zip(row, d_row)] for row, d_row in zip(self.model.W1, dW1)]
        self.model.b1 = [b - self.learning_rate * db for b, db in zip(self.model.b1, db1)]
        self.model.W2 = [[w - self.learning_rate * dw for w, dw in zip(row, d_row)] for row, d_row in zip(self.model.W2, dW2)]
        self.model.b2 = [b - self.learning_rate * db for b, db in zip(self.model.b2, db2)]
    
    def train(self, sequences, targets, epochs=10):
        for epoch in range(epochs):
            total_loss = 0
            for sequence, target in zip(sequences, targets):
                embedded_sequence = self.model.embedding.get_embeddings_for_sequence(sequence)
                input_representation = [sum(x) / len(embedded_sequence) for x in zip(*embedded_sequence)]
                logits = self.model.forward(input_representation)
                
                loss, probs = self.cross_entropy_loss(logits, target)
                total_loss += loss
                
                hidden_layer_output = tanh(add_vectors(matmul([input_representation], self.model.W1)[0], self.model.b1))
                self.backpropagate(input_representation, hidden_layer_output, probs, target)
            
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(sequences)}")
