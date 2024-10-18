import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import ParameterGrid
from models.unet_base import Unet
from models.unet_deeper import Unet_deeper
from models.unet_wide import Unet_wide
from models.unet_shallow import Unet_shallow
from utils.data_loader import load_data
from utils.metrics import mean_iou, binary_accuracy
import tensorflow as tf

# Define Dice loss
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


# Hyperparameter grid
param_grid = {
    'batch_size': [16, 32, 64],
    #'epoch': [50, 100],
    'learning_rate': [0.01, 0.001, 0.0001],
    'optimizer': ['adam'],  # 'optimizer': ['adam', 'sgd', 'rmsprop'],
    'dropout_rate': [0.2, 0.3],
    'l2_lambda': [0.1, 0.01, 0.001],
    'activation_function': ['relu', 'leaky_relu'],
    'model': [Unet, Unet_deeper, Unet_wide, Unet_shallow],
    'loss_function': ['binary_crossentropy', 'dice_loss']
}

# Load data
X_train, y_train, X_val, y_val = load_data()

# Create the grid
grid = ParameterGrid(param_grid)

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
    if params['loss_function'] == 'binary_crossentropy':
        loss = 'binary_crossentropy'
    elif params['loss_function'] == 'dice_loss':
        loss = dice_loss
    
    # Compile model
    model.compile(optimizer=optimizer, 
                  loss=loss, 
                  metrics=[binary_accuracy, mean_iou])
    
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    
    # Train model
    history = model.fit(X_train, y_train,
                        batch_size=params['batch_size'],
                        epochs=100,
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
        'val_loss': min(history.history['val_loss']),
        'val_accuracy': max(history.history['val_binary_accuracy']),
        'val_mean_iou': max(history.history['val_mean_iou']),
        'train_loss': min(history.history['loss']),
        'train_accuracy': max(history.history['binary_accuracy']),
        'train_mean_iou': max(history.history['mean_iou'])
    })

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv('results/tuning_results.csv', index=False)
