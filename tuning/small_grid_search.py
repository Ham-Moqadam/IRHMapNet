import time
now = time.time()
print("Starting imports")
import numpy as np
print("numpy....done")
import pandas as pd
print("pandas....done")
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
print("tensorflow....done")
from sklearn.model_selection import ParameterGrid
print("sklearn...done")
from models.unet_base import Unet
print("models...done")
from utils.data_loader import load_data
from utils.metrics import iou_metric, binary_accuracy
print("utils...done")
import tensorflow as tf
print("tensorflow...done")
print("Finished imports...", time.time() - now)


# Define Dice loss
def dice_loss(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

# Define Tversky and Focal Tversky loss
def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    numerator = tf.reduce_sum(y_true * y_pred)
    denominator = numerator + alpha * tf.reduce_sum((1 - y_true) * y_pred) + beta * tf.reduce_sum(y_true * (1 - y_pred))
    return 1 - numerator / denominator

def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75):
    tversky = tversky_loss(y_true, y_pred, alpha, beta)
    return tf.pow(tversky, gamma)

print("Local definitions done")

# Hyperparameter grid
param_grid = {
    'batch_size': [16, 64],
    'learning_rate': [0.001],
    'optimizer': ['adam'],
    'dropout_rate': [0.2],
    'l2_lambda': [0.001],
    'activation_function': ['relu'],
    'model': [Unet],
    'loss_function': ['dice_loss', 'binary_crossentropy', 'focal_tversky_loss']
}

# Load data
X_train, y_train, X_val, y_val = load_data()
print("Finished loading data")

# Create the grid
grid = ParameterGrid(param_grid)
print("Finished creating grid")

# Storage for results
results = []

# Hyperparameter tuning
for params in grid:
    print(f"Testing with parameters: {params}")

    # Initialize model
    model_fn = params['model']
    model = model_fn(input_shape=(512, 512, 1))

    # Configure optimizer
    if params['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'sgd':
        optimizer = SGD(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = RMSprop(learning_rate=params['learning_rate'])
    
    # Select loss function
    if params['loss_function'] == 'dice_loss':
        loss = dice_loss
    elif params['loss_function'] == 'binary_crossentropy':
        loss = 'binary_crossentropy'
    elif params['loss_function'] == 'focal_tversky_loss':
        loss = focal_tversky_loss
    
    # Compile the model
    model.compile(optimizer=optimizer, 
                  loss=loss, 
                  metrics=['accuracy', binary_accuracy, iou_metric])

    
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    # Train model
    history = model.fit(X_train, y_train,
                        batch_size=params['batch_size'],
                        epochs=70,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        verbose=1)
    
    # Record results
    results.append({
        'batch_size': params['batch_size'],
        'learning_rate': params['learning_rate'],
        'optimizer': params['optimizer'],
        'dropout_rate': params['dropout_rate'],
        'l2_lambda': params['l2_lambda'],
        'activation_function': params['activation_function'],
        'model': params['model'].__name__,
        'loss_function': params['loss_function'],
        'val_loss': min(history.history.get('val_loss', [None])),
        'val_binary_accuracy': max(history.history.get('val_binary_accuracy', [None])),
        'val_iou': max(history.history.get('val_mean_io_u', [None])),
        'train_loss': min(history.history.get('loss', [None])),
        'train_binary_accuracy': max(history.history.get('binary_accuracy', [None])),
        'train_iou': max(history.history.get('mean_io_u', [None]))
    })


# Save results
df_results = pd.DataFrame(results)
df_results.to_csv('results/small_gridsearch_tuning_results.csv', index=False)
