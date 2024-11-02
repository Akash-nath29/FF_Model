def matmul(a, b):
    result = [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]
    return result

def add_vectors(v1, v2):
    return [x + y for x, y in zip(v1, v2)]

def tanh(x):
    return [(2 / (1 + (2.71828 ** (-2 * val)))) - 1 for val in x]

def softmax(logits):
    exps = [2.71828 ** logit for logit in logits]
    total = sum(exps)
    return [exp / total for exp in exps]
