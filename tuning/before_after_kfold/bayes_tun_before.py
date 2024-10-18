import optuna
import numpy as np
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from models.unet_base import Unet
from models.unet_deeper import Unet_deeper
from models.unet_deeper2 import Unet_deeper2
from models.unet_wide import Unet_wide
from models.unet_shallow import Unet_shallow
from utils.data_loader import load_data
from utils.metrics import iou_metric, binary_accuracy
from keras.layers import LeakyReLU
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


def create_model(model_name, input_shape, optimizer_name, learning_rate, dropout_rate, l2_lambda, activation_function, loss_function):
    # Define model architecture based on the model name
    if model_name == 'Unet':
        model = Unet(input_shape=input_shape, dropout_rate=dropout_rate, l2_lambda=l2_lambda, activation_function=activation_function)
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
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Invalid optimizer name")
    
    # Select loss function
    if loss_function == 'dice_loss':
        loss = dice_loss
    elif loss_function == 'binary_crossentropy':
        loss = 'binary_crossentropy'
    elif loss_function == 'focal_tversky_loss':
        loss = focal_tversky_loss
    else:
        raise ValueError("Invalid loss function")

    # Compile the model
    model.compile(optimizer=optimizer, 
                  loss=loss,
                  metrics=[binary_accuracy, iou_metric])
    
    return model


def objective(trial):
    # Load data
    X_train, y_train, X_val, y_val = load_data()

    # Define search space
    model_name = trial.suggest_categorical('model', ['Unet_wide']) #['Unet_deeper', 'Unet_deeper2', 'Unet_wide', 'Unet', 'Unet_shallow'])
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.25)
    l2_lambda = trial.suggest_float('l2_lambda', 1e-5, 1e-1, log=True)
    activation_function = trial.suggest_categorical('activation_function', ['relu', 'leaky_relu'])
    loss_function = trial.suggest_categorical('loss_function', ['binary_crossentropy'])





    # Print the parameters at the beginning of the trial
    print(f"Starting trial with parameters: model_name={model_name}, optimizer_name={optimizer_name}, "
          f"learning_rate={learning_rate}, dropout_rate={dropout_rate}, l2_lambda={l2_lambda}, "
          f"activation_function={activation_function}, loss_function={loss_function}")


    # Create model
    model = create_model(model_name, (512, 512, 1), optimizer_name, learning_rate, dropout_rate, l2_lambda, activation_function, loss_function)
    
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # Train model
    history = model.fit(X_train, y_train,
                        batch_size=32,  # Fixed for simplicity, can be tuned as well
                        epochs=300,  # Fixed for simplicity, can be tuned as well
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        verbose=1)  # Turn off verbose to speed up optimization

    # Print the end of the trial
    print(f"Ending trial with parameters: model_name={model_name}, optimizer_name={optimizer_name}, "
          f"learning_rate={learning_rate}, dropout_rate={dropout_rate}, l2_lambda={l2_lambda}, "
          f"activation_function={activation_function}, loss_function={loss_function}")
    

    # Objective function is the validation loss
    return min(history.history['val_loss'])

def main():
    # Create a study object
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=8)  ## !!! Number of trials can be adjusted
    
    print(">>>>>>>>>>>>> THIS IS PROBABLY THE BEGINNING OF A TRAIL")

    # Save results
    df_results = study.trials_dataframe()
    df_results.to_csv('results/tuning_results.csv', index=False)
    print(f'Best trial: {study.best_trial.params}')

if __name__ == "__main__":
    main()








