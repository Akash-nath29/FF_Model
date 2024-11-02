import random

class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.embeddings = [[random.uniform(-1, 1) for _ in range(embed_dim)] for _ in range(vocab_size)]
    
    def get_embedding(self, token_id):
        return self.embeddings[token_id]
    
    def get_embeddings_for_sequence(self, sequence):
        return [self.get_embedding(token_id) for token_id in sequence]
