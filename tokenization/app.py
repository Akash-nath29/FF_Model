class Tokenizer:
    def __init__(self, query):
        self.query = query
        
    def preprocess(self):
        text_seq = sorted(set(self.query))
        return text_seq
    
    def char_to_int(self, text_seq):
        return {char: idx for idx, char in enumerate(text_seq)}
    
    def int_to_char(self, text_seq):
        return {idx: char for idx, char in enumerate(text_seq)}
    
    def encode(self):
        tokens = []
        for char in self.query:
            tokens.append(self.char_to_int(self.preprocess())[char])
            
        return tokens
    
    def decode(self, encoded):
        text_seq = self.preprocess()
        int_to_char_map = self.int_to_char(text_seq)
        
        if isinstance(encoded[0], list):
            decoded_sequences = []
            for seq in encoded:
                chars = [int_to_char_map[token] for token in seq]
                decoded_sequences.append(''.join(chars))
            return decoded_sequences
        else:
            chars = [int_to_char_map[token] for token in encoded]
            return ''.join(chars)
