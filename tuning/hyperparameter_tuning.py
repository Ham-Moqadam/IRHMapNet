import optuna
import pandas as pd
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from models.unet_base import Unet
from models.unet_deeper import Unet_deeper
from models.unet_wide import Unet_wide
from models.unet_shallow import Unet_shallow
from utils.data_loader import load_data
from utils.metrics import mean_iou, binary_accuracy
import numpy as np

def create_model(model_name, input_shape, optimizer_name, learning_rate, dropout_rate, l2_lambda, activation_function):
    """
    Create and compile the model based on the given hyperparameters.
    
    Args:
        model_name (str): The name of the model to be created.
        input_shape (tuple): The shape of the input data.
        optimizer_name (str): The name of the optimizer to be used.
        learning_rate (float): The learning rate for the optimizer.
        dropout_rate (float): The dropout rate for the Dropout layers.
        l2_lambda (float): The L2 regularization lambda.
        activation_function (str): The activation function to use.
    
    Returns:
        model (tf.keras.Model): The compiled Keras model.
    """
    if model_name == 'Unet':
        model = Unet(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
    elif model_name == 'Unet_deeper':
        model = Unet_deeper(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
    elif model_name == 'Unet_wide':
        model = Unet_wide(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
    elif model_name == 'Unet_shallow':
        model = Unet_shallow(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
    else:
        raise ValueError("Invalid model name")

    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Invalid optimizer name")

    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy', 
                  metrics=[binary_accuracy, mean_iou])
    
    return model

def objective(trial):
    """
    Objective function for Optuna optimization.
    
    Args:
        trial (optuna.Trial): The current Optuna trial.
    
    Returns:
        float: The objective value to minimize (validation loss).
    """
    # Load data
    X_train, y_train, X_val, y_val = load_data()

    # Define search space
    model_name = trial.suggest_categorical('model', ['Unet', 'Unet_deeper', 'Unet_wide', 'Unet_shallow'])
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    l2_lambda = trial.suggest_loguniform('l2_lambda', 1e-6, 1e-1)
    activation_function = trial.suggest_categorical('activation_function', ['relu', 'leaky_relu'])

    # Create model
    model = create_model(model_name, (512, 512, 1), optimizer_name, learning_rate, dropout_rate, l2_lambda, activation_function)
    
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Train model
    history = model.fit(X_train, y_train,
                        batch_size=32,  # Fixed for simplicity, can be tuned as well
                        epochs=50,  # Fixed for simplicity, can be tuned as well
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        verbose=0)  # Turn off verbose to speed up optimization
    
    # Record results
    results.append({
        'batch_size': 32,  # Fixed for simplicity
        'epochs': 50,  # Fixed for simplicity
        'learning_rate': trial.params['learning_rate'],
        'optimizer': trial.params['optimizer'],
        'dropout_rate': trial.params['dropout_rate'],
        'l2_lambda': trial.params['l2_lambda'],
        'activation_function': trial.params['activation_function'],
        'model': trial.params['model'],
        'val_loss': min(history.history['val_loss']),
        'val_accuracy': max(history.history['val_binary_accuracy']),
        'val_mean_iou': max(history.history['val_mean_iou']),
        'train_loss': min(history.history['loss']),
        'train_accuracy': max(history.history['binary_accuracy']),
        'train_mean_iou': max(history.history['mean_iou'])
    })

    # Objective function is the validation loss
    return min(history.history['val_loss'])

def main():
    """
    Main function to run Optuna hyperparameter tuning.
    """
    global results
    results = []

    # Create Optuna study
    study = optuna.create_study(direction='minimize')
    
    # Optimize the objective function
    study.optimize(objective, n_trials=80)  # Number of trials can be adjusted

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/tuning_results.csv', index=False)
    
    # Print the best trial parameters
    print(f'Best trial: {study.best_trial.params}')

if __name__ == "__main__":
    main()

