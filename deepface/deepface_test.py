from deepface import DeepFace
import math

def cosine_similarity(v1, v2):
    # Calculate the dot product
    dot_product = sum(a * b for a, b in zip(v1, v2))
    
    # Calculate the magnitude of each vector
    mag1 = math.sqrt(sum(a ** 2 for a in v1))
    mag2 = math.sqrt(sum(b ** 2 for b in v2))
    
    # Return 0.0 if either magnitude is zero to avoid division by zero
    if mag1 == 0 or mag2 == 0:
        return 0.0
        
    return dot_product / (mag1 * mag2)


verification = DeepFace.verify(img1_path = '/home/yash/grayscale.jpeg', img2_path = '/home/yash/grayscale.jpeg', distance_metric = cosine_similarity, threshold = 1)

print(verification)