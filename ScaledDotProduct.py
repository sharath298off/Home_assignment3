import numpy as np
from numpy import dot
from numpy.linalg import norm
import math

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # for numerical stability
    return e_x / e_x.sum(axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]  # dimension of the key

    # 1. Dot product of Q and K^T
    scores = np.dot(Q, K.T)

    # 2. Scale the scores
    scaled_scores = scores / math.sqrt(d_k)

    # 3. Softmax to get attention weights
    attention_weights = softmax(scaled_scores)

    # 4. Multiply attention weights by V
    output = np.dot(attention_weights, V)

    return attention_weights, output

# Test inpt
Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
K = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
V = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# Run attention
attention_weights, output = scaled_dot_product_attention(Q, K, V)

# Display results
print("Attention Weights:\n", attention_weights)
print("\nOutput:\n", output)
