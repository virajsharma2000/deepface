from deepface import DeepFace
import numpy as np

def cosine_distance_matrix(x, y):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    
    # Normalize vectors to unit length (L2 norm)
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)
    
    # Compute cosine similarity via dot product
    similarity = np.dot(x_norm, y_norm.T)
    
    # Convert similarity to distance
    distance_matrix = 1.0 - similarity
    return distance_matrix

verification = DeepFace.verify(img1_path = '/home/yash/grayscale.jpeg', img2_path = '/home/yash/grayscale.jpeg')


#verification = DeepFace.verify(img1_path = '/home/yash/grayscale.jpeg', img2_path = '/home/yash/grayscale.jpeg', distance_metric = cosine_distance_matrix, threshold = 0.68)

print(verification)