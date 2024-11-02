class Splitter: 
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def split(self):
        X, y = [], []
        
        for i in range(len(self.data) - self.seq_len):
            X.append(self.data[i:i + self.seq_len])
            y.append(self.data[i + self.seq_len])
        
        return X, y
    
    def train_test_split(self, X, y, test_size=0.2):
        split = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        return X_train, X_test, y_train, y_test
