

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from models.unet_base import Unet
from models.unet_deeper import Unet_deeper
from models.unet_simple import Unet_simple
from models.unet_deeper2 import Unet_deeper2
from models.unet_wide import Unet_wide
from models.unet_shallow import Unet_shallow
from utils.data_loader import load_data
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
        model = Unet(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
    elif model_name == 'Unet_simple':
        model = Unet_simple(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
    elif model_name == 'Unet_deeper':
        model = Unet_deeper(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
    elif model_name == 'Unet_deeper2':
        model = Unet_deeper2(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
    elif model_name == 'Unet_wide':
        model = Unet_wide(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
    elif model_name == 'Unet_shallow':
        model = Unet_shallow(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
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
        {'model_name': 'Unet_deeper2', 'optimizer_name': 'rmsprop', 'learning_rate': 0.00351394877194278, 'dropout_rate': 0.238020124135863, 'l2_lambda': 9.07171146644635E-05, 'activation_function': 'relu', 'loss_function': 'dice_loss', 'momentum': 0.0},
        {'model_name': 'Unet_deeper', 'optimizer_name': 'rmsprop', 'learning_rate': 0.00193957995987269, 'dropout_rate': 0.10, 'l2_lambda': 0.000126474216011107, 'activation_function': 'leaky_relu', 'loss_function': 'dice_loss', 'momentum': 0.9},
        {'model_name': 'Unet_simple', 'optimizer_name': 'rmsprop', 'learning_rate': 0.000379965353035444, 'dropout_rate': 0.15, 'l2_lambda': 0.000138703775770967, 'activation_function': 'leaky_relu', 'loss_function': 'dice_loss', 'momentum': 0.7},
        {'model_name': 'Unet_wide', 'optimizer_name': 'rmsprop', 'learning_rate': 0.0011685381367662, 'dropout_rate': 0.35, 'l2_lambda': 0.000232181110867993, 'activation_function': 'relu', 'loss_function': 'dice_loss', 'momentum': 0.6},
        {'model_name': 'Unet_wide', 'optimizer_name': 'rmsprop', 'learning_rate': 0.00157407997721135, 'dropout_rate': 0.35, 'l2_lambda': 0.000409148653646832, 'activation_function': 'leaky_relu', 'loss_function': 'dice_loss', 'momentum': 0.9},
        {'model_name': 'Unet_wide', 'optimizer_name': 'rmsprop', 'learning_rate': 0.000625272318169693, 'dropout_rate': 0.2, 'l2_lambda': 9.40856490371518E-05, 'activation_function': 'leaky_relu', 'loss_function': 'dice_loss', 'momentum': 0},
        {'model_name': 'Unet_wide', 'optimizer_name': 'rmsprop', 'learning_rate': 0.000376906919158002, 'dropout_rate': 0.25, 'l2_lambda': 1.25278265564834E-05, 'activation_function': 'relu', 'loss_function': 'dice_loss', 'momentum': 0.1},
        {'model_name': 'Unet_deeper', 'optimizer_name': 'rmsprop', 'learning_rate': 0.0021033331444821, 'dropout_rate': 0.15, 'l2_lambda': 0.000117601559457191, 'activation_function': 'leaky_relu', 'loss_function': 'dice_loss', 'momentum': 0.9},
        {'model_name': 'Unet_shallow', 'optimizer_name': 'rmsprop', 'learning_rate': 0.00389013982528495, 'dropout_rate': 0.15, 'l2_lambda': 2.87791305263086E-05, 'activation_function': 'relu', 'loss_function': 'dice_loss', 'momentum': 0.5},
        {'model_name': 'Unet_simple', 'optimizer_name': 'adam', 'learning_rate': 0.000419905642944484, 'dropout_rate': 0.15, 'l2_lambda': 0.000923269604118875, 'activation_function': 'relu', 'loss_function': 'dice_loss', 'momentum': 0.1},
        {'model_name': 'Unet_simple', 'optimizer_name': 'adam', 'learning_rate': 0.00192081160880065, 'dropout_rate': 0.1, 'l2_lambda': 0.000510457031300437, 'activation_function': 'relu', 'loss_function': 'dice_loss', 'momentum': 0.5},
        {'model_name': 'Unet_simple', 'optimizer_name': 'rmsprop', 'learning_rate': 0.00629939514447868, 'dropout_rate': 0.3, 'l2_lambda': 3.05623953556791E-05, 'activation_function': 'relu', 'loss_function': 'dice_loss', 'momentum': 0.2},

#        {'model_name': '', 'optimizer_name': '', 'learning_rate': , 'dropout_rate': , 'l2_lambda': , 'activation_function': '', 'loss_function': '', 'momentum': },

        # Add the rest of the 12 sets
        # {'model_name': '...', 'optimizer_name': '...', 'learning_rate': ..., 'dropout_rate': ..., 'l2_lambda': ..., 'activation_function': '...', 'loss_function': '...', 'momentum': ...},
        # ...
    ]

    # Directory to save results
    result_dir = 'results/learning/set_2'
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

