
### -----------------------------------------------------------------------
### date:       19.07.2024
### author:     H.Moqadam
### desc:       Routine for bayesian Optimization for
###             yhper-parameter tuning
###
### -----------------------------------------------------------------------



<<<<<<< HEAD

import optuna
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from models.unet_base import Unet
from models.unet_deeper import Unet_deeper
from models.unet_simple import Unet_simple
from models.unet_deeper2 import Unet_deeper2
=======
import optuna
import numpy as np
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from models.unet_base import Unet
from models.unet_deeper import Unet_deeper
>>>>>>> 56b66cdcfa95967b6c9656a18e95d6e9a4bd862b
from models.unet_wide import Unet_wide
from models.unet_shallow import Unet_shallow
from utils.data_loader import load_data
from utils.metrics import iou_metric, binary_accuracy
<<<<<<< HEAD
import tensorflow as tf

=======
from keras.layers import LeakyReLU
import tensorflow as tf


>>>>>>> 56b66cdcfa95967b6c9656a18e95d6e9a4bd862b
# Define Dice loss
def dice_loss(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

<<<<<<< HEAD
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
=======
def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    numerator = tf.reduce_sum(y_true * y_pred)
    denominator = numerator + alpha * tf.reduce_sum((1 - y_true) * y_pred) + beta * tf.reduce_sum(y_true * (1 - y_pred))
    return 1 - numerator / denominator

def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75):
    tversky = tversky_loss(y_true, y_pred, alpha, beta)
    return tf.pow(tversky, gamma)


def create_model(model_name, input_shape, optimizer_name, learning_rate, dropout_rate, l2_lambda, activation_function, loss_function):
    # Define model architecture based on the model name
    if model_name == 'Unet':
        model = Unet(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
    elif model_name == 'Unet_deeper':
        model = Unet_deeper(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
>>>>>>> 56b66cdcfa95967b6c9656a18e95d6e9a4bd862b
    elif model_name == 'Unet_wide':
        model = Unet_wide(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
    elif model_name == 'Unet_shallow':
        model = Unet_shallow(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
    else:
        raise ValueError("Invalid model name")

    # Define optimizer based on the optimizer name
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
<<<<<<< HEAD
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
=======
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
>>>>>>> 56b66cdcfa95967b6c9656a18e95d6e9a4bd862b
    else:
        raise ValueError("Invalid optimizer name")
    
    # Select loss function
    if loss_function == 'dice_loss':
        loss = dice_loss
    elif loss_function == 'binary_crossentropy':
        loss = 'binary_crossentropy'
<<<<<<< HEAD
=======
    elif loss_function == 'focal_tversky_loss':
        loss = focal_tversky_loss
>>>>>>> 56b66cdcfa95967b6c9656a18e95d6e9a4bd862b
    else:
        raise ValueError("Invalid loss function")

    # Compile the model
    model.compile(optimizer=optimizer, 
                  loss=loss,
<<<<<<< HEAD
                  metrics=[binary_accuracy, iou_metric])
=======
                  metrics=['accuracy', binary_accuracy, iou_metric])
>>>>>>> 56b66cdcfa95967b6c9656a18e95d6e9a4bd862b
    
    return model

def objective(trial):
    # Load data
<<<<<<< HEAD
    X_train, Y_train, X_test, Y_test = load_data()

    # Define search space
    model_name = trial.suggest_categorical('model', ['Unet', 'Unet_simple', 'Unet_shallow', 'Unet_wide', 'Unet_deeper'])
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4, step=0.05)
    l2_lambda = trial.suggest_float('l2_lambda', 1e-5, 1e-1, log=True)
    activation_function = trial.suggest_categorical('activation_function', ['relu', 'leaky_relu'])
    loss_function = trial.suggest_categorical('loss_function', ['binary_crossentropy', 'dice_loss'])
    momentum = trial.suggest_float('momentum', 0.0, 0.9, step=0.1)

    # Print the parameters at the beginning of the trial
    print(f"Starting trial with parameters: model_name={model_name}, optimizer_name={optimizer_name}, "
          f"learning_rate={learning_rate}, dropout_rate={dropout_rate}, l2_lambda={l2_lambda}, "
          f"activation_function={activation_function}, loss_function={loss_function}, momentum={momentum}")

    # Create model
    model = create_model(model_name, (512, 512, 1), optimizer_name, learning_rate, dropout_rate, l2_lambda, activation_function, loss_function, momentum)

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    
    # Learning rate scheduler
    def scheduler(epoch, lr):
        if epoch < 10:
            return float(lr)  # Ensure the return value is a float
        else:
            return float(lr * tf.math.exp(-0.1))  # Ensure the return value is a float
    
    lr_scheduler = LearningRateScheduler(scheduler)

    # Train model
    history = model.fit(X_train, Y_train,
                        batch_size=32,
                        epochs=50,
                        validation_data=(X_test, Y_test),
                        callbacks=[early_stopping, lr_scheduler],
                        verbose=1)

    # Save results for this trial
    trial_result = {
        'trial_number': trial.number,
        'model_name': model_name,
        'optimizer_name': optimizer_name,
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'l2_lambda': l2_lambda,
        'activation_function': activation_function,
        'loss_function': loss_function,
        'momentum': momentum,
        'best_val_loss': min(history.history['val_loss'])
    }

###! this is the start of saving after each trial
    
    # Save trial results to a CSV file
    trial_results_file = 'results/hy_tuning/6th_try/trial_results.csv'
    with open(trial_results_file, 'a') as f:
        f.write(f"{trial_result['trial_number']},{trial_result['model_name']},{trial_result['optimizer_name']},{trial_result['learning_rate']},{trial_result['dropout_rate']},{trial_result['l2_lambda']},{trial_result['activation_function']},{trial_result['loss_function']},{trial_result['momentum']},{trial_result['best_val_loss']}\n")

    # Plot and save training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'],'-x', label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.title(f"Trial {trial.number}: Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['binary_accuracy'],'-x', label='Training Accuracy')
    plt.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.title(f"Trial {trial.number}: Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/hy_tuning/6th_try/trial_{trial.number}_history.png')
    plt.close()

    return trial_result['best_val_loss']

###! this is the end of saving after each trial

=======
    X_train, y_train, X_val, y_val = load_data()

    # Define search space
    model_name = trial.suggest_categorical('model', ['Unet', 'Unet_deeper', 'Unet_wide', 'Unet_shallow'])
    optimizer_name = trial.suggest_categorical('optimizer', ['adam'])
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    l2_lambda = trial.suggest_float('l2_lambda', 1e-3, 1e-2, log=True)
    activation_function = trial.suggest_categorical('activation_function', ['relu', 'leaky_relu'])
    loss_function = trial.suggest_categorical('loss_function', ['binary_crossentropy', 'dice_loss', 'focal_tversky_loss'])

    # Create model
    model = create_model(model_name, (512, 512, 1), optimizer_name, learning_rate, dropout_rate, l2_lambda, activation_function, loss_function)
    
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Train model
    history = model.fit(X_train, y_train,
                        batch_size=32,  # Fixed for simplicity, can be tuned as well
                        epochs=80,  # Fixed for simplicity, can be tuned as well
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        verbose=1)  # Turn off verbose to speed up optimization

    # Objective function is the validation loss
    return min(history.history['val_loss'])
>>>>>>> 56b66cdcfa95967b6c9656a18e95d6e9a4bd862b

def main():
    # Create a study object
    study = optuna.create_study(direction='minimize')
<<<<<<< HEAD
    study.optimize(objective, n_trials=20)  # Number of trials can be adjusted

    # Save results
    df_results = study.trials_dataframe()
    df_results.to_csv('results/hy_tuning/6th_try/tuning_results.csv', index=False)
=======
    study.optimize(objective, n_trials=7)  ## !!! Number of trials can be adjusted
    
    print(">>>>>>>>>>>>> THIS IS PROBABLY THE BEGINNING OF A TRAIL")

    # Save results
    df_results = study.trials_dataframe()
    df_results.to_csv('results/tuning_results.csv', index=False)
>>>>>>> 56b66cdcfa95967b6c9656a18e95d6e9a4bd862b
    print(f'Best trial: {study.best_trial.params}')

if __name__ == "__main__":
    main()



<<<<<<< HEAD
=======









>>>>>>> 56b66cdcfa95967b6c9656a18e95d6e9a4bd862b
    
    ### --- to save all the models:
        # Save the model
    # model_dir = 'results/models'
    # os.makedirs(model_dir, exist_ok=True)
    # model_path = os.path.join(model_dir, f'model_trial_{trial.number}.h5')
    # model.save(model_path)
    ### -------------   
    
    
    ### --- to save the best mode:
    #     # Save the model if it's the best one so far
    # val_loss = min(history.history['val_loss'])
    # if not hasattr(objective, 'best_val_loss') or val_loss < objective.best_val_loss:
    #     objective.best_val_loss = val_loss
    #     model.save('results/best_model.h5')
    #     objective.best_model_params = {
    #         'model_name': model_name,
    #         'optimizer_name': optimizer_name,
    #         'learning_rate': learning_rate,
    #         'dropout_rate': dropout_rate,
    #         'l2_lambda': l2_lambda,
    #         'activation_function': activation_function,
    #     }
    ### -------------   
    
    
