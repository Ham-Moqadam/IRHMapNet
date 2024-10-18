


import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy, MeanIoU



# Define built-in metrics
binary_accuracy_metric = BinaryAccuracy()
iou_metric = MeanIoU(num_classes=2)

# Define your custom losses
def dice_loss(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    numerator = tf.reduce_sum(y_true * y_pred)
    denominator = numerator + alpha * tf.reduce_sum((1 - y_true) * y_pred) + beta * tf.reduce_sum(y_true * (1 - y_pred))
    return 1 - numerator / denominator

def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75):
    tversky = tversky_loss(y_true, y_pred, alpha, beta)
    return tf.pow(tversky, gamma)

@tf.function
def binary_accuracy(y_true, y_pred):
    return tf.keras.metrics.binary_accuracy(y_true, y_pred)