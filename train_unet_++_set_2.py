

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from models.nested_unet_base import nested_unet
from models.nested_unet_deep import nested_unet_deep
from models.nested_unet_deeper import nested_unet_deeper
from models.nested_unet_simple import nested_unet_simple
from models.nested_unet_wide import nested_unet_wide
from models.unet_shallow import Unet_shallow
from utils.data_loader_set2 import load_data
from utils.metrics import iou_metric, binary_accuracy
import tensorflow as tf

# Define Dice loss
def dice_loss(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

def create_model(model_name, input_shape, optimizer_name, learning_rate, dropout_rate, l2_lambda, activation_function, loss_function, momentum=0.0):
    # Define model architecture based on the model name
    if model_name == 'Unet':
        model = nested_unet(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
    elif model_name == 'Unet_simple':
        model = nested_unet_simple(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
    elif model_name == 'Unet_deep':
        model = nested_unet_deep(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
    elif model_name == 'Unet_deeper':
        model = nested_unet_deeper(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
    elif model_name == 'Unet_wide':
        model = nested_unet_wide(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
#    elif model_name == 'Unet_shallow':
#        model = Unet_shallow(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
    else:
        raise ValueError("Invalid model name")

    # Define optimizer based on the optimizer name
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    else:
        raise ValueError("Invalid optimizer name")
    
    # Select loss function
    if loss_function == 'dice_loss':
        loss = dice_loss
    elif loss_function == 'binary_crossentropy':
        loss = 'binary_crossentropy'
    else:
        raise ValueError("Invalid loss function")

    # Compile the model
    model.compile(optimizer=optimizer, 
                  loss=loss,
                  metrics=[binary_accuracy, iou_metric])
    
    return model

def train_and_evaluate(hyperparams, trial_num, result_dir):
    # Load data
    X_train, Y_train, X_test, Y_test = load_data()

    # Create model
    model = create_model(
        model_name=hyperparams['model_name'],
        input_shape=(512, 512, 1),
        optimizer_name=hyperparams['optimizer_name'],
        learning_rate=hyperparams['learning_rate'],
        dropout_rate=hyperparams['dropout_rate'],
        l2_lambda=hyperparams['l2_lambda'],
        activation_function=hyperparams['activation_function'],
        loss_function=hyperparams['loss_function'],
        momentum=hyperparams['momentum']
    )

    ## Early stopping
    ## restore_best_weights = True : this will restore the model's weights to the epoch where the monitored metric was best before stopping 
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    
    # Learning rate scheduler
    def scheduler(epoch, lr):
        if epoch < 10:
            return float(lr)
        else:
            return float(lr * tf.math.exp(-0.1))
    
    lr_scheduler = LearningRateScheduler(scheduler)

    # Train model
    history = model.fit(X_train, Y_train,
                        batch_size=16,
                        epochs=200,
                        validation_data=(X_test, Y_test),
                        callbacks=[early_stopping, lr_scheduler],
                        verbose=1)

    # Save model and history
    model.save(os.path.join(result_dir, f'trial_{trial_num}_best_model.keras'))
    np.save(os.path.join(result_dir, f'trial_{trial_num}_history.npy'), history.history)

    # Plot and save training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], '-x', label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.title(f"Trial {trial_num}: Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['binary_accuracy'], '-x', label='Training Accuracy')
    plt.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.title(f"Trial {trial_num}: Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'trial_{trial_num}_history.png'))
    plt.close()

    # Calculate the best validation loss
    best_val_loss = min(history.history['val_loss'])

    # Return relevant metrics
    return {
        'trial_number': trial_num,
        'model_name': hyperparams['model_name'],
        'optimizer_name': hyperparams['optimizer_name'],
        'learning_rate': hyperparams['learning_rate'],
        'dropout_rate': hyperparams['dropout_rate'],
        'l2_lambda': hyperparams['l2_lambda'],
        'activation_function': hyperparams['activation_function'],
        'loss_function': hyperparams['loss_function'],
        'momentum': hyperparams['momentum'],
        'best_val_loss': best_val_loss,
    }

def main():
    # Define 12 sets of pre-selected optimal hyper-parameters
    hyperparameter_sets = [
    {'model_name': 'Unet_deep', 'optimizer_name': 'sgd', 'learning_rate': 0.00856, 'dropout_rate': 0.25, 'l2_lambda': 0.00047, 'activation_function': 'leaky_relu', 'loss_function': 'binary_crossentropy', 'momentum': 0.1},
    {'model_name': 'Unet_deeper', 'optimizer_name': 'sgd', 'learning_rate': 0.01277, 'dropout_rate': 0.15, 'l2_lambda': 0.00063, 'activation_function': 'leaky_relu', 'loss_function': 'binary_crossentropy', 'momentum': 0.8},
    {'model_name': 'Unet_simple', 'optimizer_name': 'adam', 'learning_rate': 0.00002, 'dropout_rate': 0.3, 'l2_lambda': 0.00001, 'activation_function': 'leaky_relu', 'loss_function': 'dice_loss', 'momentum': 0.9},
    {'model_name': 'Unet_simple', 'optimizer_name': 'rmsprop', 'learning_rate': 0.00001, 'dropout_rate': 0.35, 'l2_lambda': 0.00543, 'activation_function': 'leaky_relu', 'loss_function': 'binary_crossentropy', 'momentum': 0.6},
    {'model_name': 'Unet_wide', 'optimizer_name': 'adam', 'learning_rate': 0.00042, 'dropout_rate': 0.1, 'l2_lambda': 0.00025, 'activation_function': 'relu', 'loss_function': 'dice_loss', 'momentum': 0.7},
    {'model_name': 'Unet', 'optimizer_name': 'sgd', 'learning_rate': 0.00784, 'dropout_rate': 0.4, 'l2_lambda': 0.00002, 'activation_function': 'relu', 'loss_function': 'dice_loss', 'momentum': 0.6},
    {'model_name': 'Unet', 'optimizer_name': 'adam', 'learning_rate': 0.00018, 'dropout_rate': 0.4, 'l2_lambda': 0.00051, 'activation_function': 'leaky_relu', 'loss_function': 'dice_loss', 'momentum': 0.1},
    {'model_name': 'Unet_simple', 'optimizer_name': 'adam', 'learning_rate': 0.00013, 'dropout_rate': 0.1, 'l2_lambda': 0.06657, 'activation_function': 'relu', 'loss_function': 'dice_loss', 'momentum': 0.3},
    {'model_name': 'Unet_deeper', 'optimizer_name': 'sgd', 'learning_rate': 0.00209, 'dropout_rate': 0.35, 'l2_lambda': 0.00002, 'activation_function': 'leaky_relu', 'loss_function': 'binary_crossentropy', 'momentum': 0.4},
    {'model_name': 'Unet_deep', 'optimizer_name': 'rmsprop', 'learning_rate': 0.00207, 'dropout_rate': 0.1, 'l2_lambda': 0.03746, 'activation_function': 'leaky_relu', 'loss_function': 'dice_loss', 'momentum': 0.5},
    {'model_name': 'Unet_wide', 'optimizer_name': 'sgd', 'learning_rate': 0.00036, 'dropout_rate': 0.1, 'l2_lambda': 0.00167, 'activation_function': 'relu', 'loss_function': 'binary_crossentropy', 'momentum': 0.2},
    {'model_name': 'Unet_deep', 'optimizer_name': 'rmsprop', 'learning_rate': 0.00709, 'dropout_rate': 0.2, 'l2_lambda': 0.08835, 'activation_function': 'relu', 'loss_function': 'dice_loss', 'momentum': 0.9},
    {'model_name': 'Unet_deep', 'optimizer_name': 'rmsprop', 'learning_rate': 0.00675, 'dropout_rate': 0.2, 'l2_lambda': 0.04871, 'activation_function': 'relu', 'loss_function': 'dice_loss', 'momentum': 0.9},
    {'model_name': 'Unet', 'optimizer_name': 'rmsprop', 'learning_rate': 0.01414, 'dropout_rate': 0.25, 'l2_lambda': 0.01072, 'activation_function': 'relu', 'loss_function': 'dice_loss', 'momentum': 0.7},
    {'model_name': 'Unet_simple', 'optimizer_name': 'adam', 'learning_rate': 0.00001, 'dropout_rate': 0.3, 'l2_lambda': 0.00017, 'activation_function': 'relu', 'loss_function': 'dice_loss', 'momentum': 0.8},
    {'model_name': 'Unet_deeper', 'optimizer_name': 'rmsprop', 'learning_rate': 0.00016, 'dropout_rate': 0.2, 'l2_lambda': 0.00006, 'activation_function': 'leaky_relu', 'loss_function': 'dice_loss', 'momentum': 0.8},
    {'model_name': 'Unet_simple', 'optimizer_name': 'rmsprop', 'learning_rate': 0.00003, 'dropout_rate': 0.15, 'l2_lambda': 0.00063, 'activation_function': 'relu', 'loss_function': 'binary_crossentropy', 'momentum': 0},
    {'model_name': 'Unet_wide', 'optimizer_name': 'rmsprop', 'learning_rate': 0.00005, 'dropout_rate': 0.15, 'l2_lambda': 0.00032, 'activation_function': 'leaky_relu', 'loss_function': 'dice_loss', 'momentum': 0.4},
    {'model_name': 'Unet_simple', 'optimizer_name': 'adam', 'learning_rate': 0.00001, 'dropout_rate': 0.25, 'l2_lambda': 0.00016, 'activation_function': 'relu', 'loss_function': 'binary_crossentropy', 'momentum': 0.2}
]



    # Directory to save results
    result_dir = 'results/learning/nested_set_2'
    os.makedirs(result_dir, exist_ok=True)

    # Prepare CSV to save metrics
    metrics_file = os.path.join(result_dir, 'training_metrics.csv')
    with open(metrics_file, 'w') as f:
        f.write('trial_number,model_name,optimizer_name,learning_rate,dropout_rate,l2_lambda,activation_function,loss_function,momentum,best_val_loss\n')

    # Iterate through each hyper-parameter set
    for i, hyperparams in enumerate(hyperparameter_sets):
        print(f"Training model with hyperparameters: {hyperparams}")
        trial_result = train_and_evaluate(hyperparams, trial_num=i+1, result_dir=result_dir)

        # Save trial results to CSV
        with open(metrics_file, 'a') as f:
            f.write(f"{trial_result['trial_number']},{trial_result['model_name']},{trial_result['optimizer_name']},{trial_result['learning_rate']},{trial_result['dropout_rate']},{trial_result['l2_lambda']},{trial_result['activation_function']},{trial_result['loss_function']},{trial_result['momentum']},{trial_result['best_val_loss']}\n")

        print(f"Completed trial {i+1}/{len(hyperparameter_sets)}. Best validation loss: {trial_result['best_val_loss']}")

if __name__ == "__main__":
    main()

