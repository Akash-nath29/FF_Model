from functions import softmax
from embeddings import Embedding
from .nn import FeedForwardNetwork

class NextTokenPredictor:
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.model = FeedForwardNetwork(embed_dim, hidden_dim, vocab_size)
    
    def predict(self, sequence):
        embedded_sequence = self.embedding.get_embeddings_for_sequence(sequence)
        input_representation = [sum(x) / len(embedded_sequence) for x in zip(*embedded_sequence)]
        logits = self.model.forward(input_representation)
        probs = softmax(logits)
        next_token = probs.index(max(probs))
        return next_token, probs
