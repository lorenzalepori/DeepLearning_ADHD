import tensorflow as tf

def euclidean_distance(inputs):
    embedding_A, embedding_B = inputs
    
    distance = tf.norm(embedding_A - embedding_B, axis=1, keepdims=True)
    
    return distance


def contrastive_loss(y_true, y_pred):
    margin = 1.0
    
    loss = (
        y_true * y_pred**2 +
        (1 - y_true) * tf.maximum(margin - y_pred, 0)**2
    )
    
    return loss