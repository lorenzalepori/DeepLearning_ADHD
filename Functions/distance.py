import tensorflow as tf
from tf_keras import backend as K

MARGIN = 1.0


def euclidean_distance(vectors):
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def contrastive_loss(y_true, y_pred, margin=MARGIN):
    y_true = tf.cast(y_true, y_pred.dtype)
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0.0))
    return y_true * square_pred + (1 - y_true) * margin_square