
### -----------------------------------------------------------------------
### date:       19.07.2024
### author:     H.Moqadam
### desc:       Routine for bayesian Optimization for
###             yhper-parameter tuning
###
### -----------------------------------------------------------------------

import tensorflow as tf
tf.config.optimizer.set_jit(False)


import optuna
import numpy as np
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping
from models.unet_base import Unet
from models.unet_deeper import Unet_deeper
from models.unet_simple import Unet_simple
from models.unet_deeper2 import Unet_deeper2
from models.unet_wide import Unet_wide
from models.unet_shallow import Unet_shallow
from utils.data_loader import load_data
from utils.metrics import iou_metric, binary_accuracy
from sklearn.model_selection import KFold
import tensorflow as tf

# Define Dice loss
def dice_loss(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

def create_model(model_name, input_shape, optimizer_name, learning_rate, dropout_rate, l2_lambda, activation_function, loss_function, momentum=None):
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

    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    else:
        raise ValueError("Invalid optimizer name")

    if loss_function == 'dice_loss':
        loss = dice_loss
    elif loss_function == 'binary_crossentropy':
        loss = 'binary_crossentropy'
    else:
        raise ValueError("Invalid loss function")

    model.compile(optimizer=optimizer, 
                  loss=loss,
                  metrics=[binary_accuracy, iou_metric])
    
    return model

def objective(trial):
    # Load full dataset (without train-test split)
    X_train, Y_train, X_test, Y_test = load_data()

    # Concatenate train and test sets for K-Fold Cross-Validation
    X = np.concatenate([X_train, X_test], axis=0)
    Y = np.concatenate([Y_train, Y_test], axis=0)

    # Define search space
    model_name = trial.suggest_categorical('model', ['Unet', 'Unet_simple', 'Unet_shallow', 'Unet_wide', 'Unet_deeper'])
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop','sgd'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.25)
    l2_lambda = trial.suggest_float('l2_lambda', 1e-5, 1e-1, log=True)
    activation_function = trial.suggest_categorical('activation_function', ['relu', 'leaky_relu'])
    loss_function = trial.suggest_categorical('loss_function', ['binary_crossentropy', 'dice_loss'])

    # Add momentum to the search space only if SGD is selected as the optimizer
    momentum = trial.suggest_float('momentum', 0.0, 0.9) if optimizer_name == 'sgd' else None


    print(f"Starting trial with parameters: model_name={model_name}, optimizer_name={optimizer_name}, "
          f"learning_rate={learning_rate}, dropout_rate={dropout_rate}, l2_lambda={l2_lambda}, "
          f"activation_function={activation_function}, loss_function={loss_function}")

    # K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_val_losses = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]

        model = create_model(model_name, (512, 512, 1), optimizer_name, learning_rate, dropout_rate, l2_lambda, activation_function, loss_function, momentum)

        early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        history = model.fit(X_train, Y_train,
                            batch_size=16,
                            epochs=200,
                            validation_data=(X_val, Y_val),
                            callbacks=[early_stopping],
                            verbose=0)

        fold_val_losses.append(min(history.history['val_loss']))

    return np.mean(fold_val_losses)


def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)

    df_results = study.trials_dataframe()
    df_results.to_csv('results/hy_tuning/tuning_results.csv', index=False)
    print(f'Best trial: {study.best_trial.params}')

if __name__ == "__main__":
    main()





    
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
    
    
