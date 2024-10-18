from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def Unet_wide(input_shape=(512, 512, 1), dropout_rate=0.1, l2_lambda=0.001, activation_function='relu'):
    inputs = Input(shape=input_shape)

    # Choose activation function
    if activation_function == 'relu':
        activation = 'relu'
    elif activation_function == 'leaky_relu':
        activation = LeakyReLU(alpha=0.1)
    else:
        raise ValueError("Invalid activation function")

    # Contracting Path
    c1 = Conv2D(32, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                kernel_regularizer=l2(l2_lambda))(inputs)
    c1 = Dropout(dropout_rate)(c1)
    c1 = Conv2D(32, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                kernel_regularizer=l2(l2_lambda))(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                kernel_regularizer=l2(l2_lambda))(p1)
    c2 = Dropout(dropout_rate)(c2)
    c2 = Conv2D(64, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                kernel_regularizer=l2(l2_lambda))(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                kernel_regularizer=l2(l2_lambda))(p2)
    c3 = Dropout(dropout_rate)(c3)
    c3 = Conv2D(128, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                kernel_regularizer=l2(l2_lambda))(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                kernel_regularizer=l2(l2_lambda))(p3)
    c4 = Dropout(dropout_rate)(c4)
    c4 = Conv2D(256, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                kernel_regularizer=l2(l2_lambda))(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(512, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                kernel_regularizer=l2(l2_lambda))(p4)
    c5 = Dropout(dropout_rate)(c5)
    c5 = Conv2D(512, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                kernel_regularizer=l2(l2_lambda))(c5)

    # Expansive Path
    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                kernel_regularizer=l2(l2_lambda))(u6)
    c6 = Dropout(dropout_rate)(c6)
    c6 = Conv2D(256, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                kernel_regularizer=l2(l2_lambda))(c6)

    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                kernel_regularizer=l2(l2_lambda))(u7)
    c7 = Dropout(dropout_rate)(c7)
    c7 = Conv2D(128, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                kernel_regularizer=l2(l2_lambda))(c7)

    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                kernel_regularizer=l2(l2_lambda))(u8)
    c8 = Dropout(dropout_rate)(c8)
    c8 = Conv2D(64, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                kernel_regularizer=l2(l2_lambda))(c8)

    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(32, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                kernel_regularizer=l2(l2_lambda))(u9)
    c9 = Dropout(dropout_rate)(c9)
    c9 = Conv2D(32, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                kernel_regularizer=l2(l2_lambda))(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

