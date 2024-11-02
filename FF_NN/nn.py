from ..functions import matmul, add_vectors, tanh
import random

class FeedforwardNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = [[random.uniform(-0.01, 0.01) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = [0 for _ in range(hidden_dim)]
        self.W2 = [[random.uniform(-0.01, 0.01) for _ in range(output_dim)] for _ in range(hidden_dim)]
        self.b2 = [0 for _ in range(output_dim)]
    
    def forward(self, x):
        z1 = add_vectors(matmul([x], self.W1)[0], self.b1)
        a1 = tanh(z1)
        z2 = add_vectors(matmul([a1], self.W2)[0], self.b2)
        return z2
