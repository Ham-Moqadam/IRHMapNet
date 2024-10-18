from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def nested_unet_deeper(input_shape=(512, 512, 1), dropout_rate=0.1, l2_lambda=0.001, activation_function='relu'):
    inputs = Input(shape=input_shape)

    # Choose activation function
    if activation_function == 'relu':
        activation = 'relu'
    elif activation_function == 'leaky_relu':
        activation = LeakyReLU(alpha=0.1)
    else:
        raise ValueError("Invalid activation function")

    # Contracting Path
    c1_0 = Conv2D(16, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                  kernel_regularizer=l2(l2_lambda))(inputs)
    c1_0 = Dropout(dropout_rate)(c1_0)
    c1_0 = Conv2D(16, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                  kernel_regularizer=l2(l2_lambda))(c1_0)
    p1 = MaxPooling2D((2, 2))(c1_0)

    c2_0 = Conv2D(32, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                  kernel_regularizer=l2(l2_lambda))(p1)
    c2_0 = Dropout(dropout_rate)(c2_0)
    c2_0 = Conv2D(32, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                  kernel_regularizer=l2(l2_lambda))(c2_0)
    p2 = MaxPooling2D((2, 2))(c2_0)

    c3_0 = Conv2D(64, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                  kernel_regularizer=l2(l2_lambda))(p2)
    c3_0 = Dropout(dropout_rate)(c3_0)
    c3_0 = Conv2D(64, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                  kernel_regularizer=l2(l2_lambda))(c3_0)
    p3 = MaxPooling2D((2, 2))(c3_0)

    c4_0 = Conv2D(128, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                  kernel_regularizer=l2(l2_lambda))(p3)
    c4_0 = Dropout(dropout_rate)(c4_0)
    c4_0 = Conv2D(128, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                  kernel_regularizer=l2(l2_lambda))(c4_0)
    p4 = MaxPooling2D((2, 2))(c4_0)

    c5_0 = Conv2D(256, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                  kernel_regularizer=l2(l2_lambda))(p4)
    c5_0 = Dropout(dropout_rate)(c5_0)
    c5_0 = Conv2D(256, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                  kernel_regularizer=l2(l2_lambda))(c5_0)

    # Nested skip connections and expansive path
    u4_1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5_0)
    u4_1 = concatenate([u4_1, c4_0])
    c4_1 = Conv2D(128, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                  kernel_regularizer=l2(l2_lambda))(u4_1)
    c4_1 = Dropout(dropout_rate)(c4_1)
    c4_1 = Conv2D(128, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                  kernel_regularizer=l2(l2_lambda))(c4_1)

    u3_2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4_1)
    u3_2 = concatenate([u3_2, c3_0])
    c3_2 = Conv2D(64, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                  kernel_regularizer=l2(l2_lambda))(u3_2)
    c3_2 = Dropout(dropout_rate)(c3_2)
    c3_2 = Conv2D(64, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                  kernel_regularizer=l2(l2_lambda))(c3_2)

    u2_3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c3_2)
    u2_3 = concatenate([u2_3, c2_0])
    c2_3 = Conv2D(32, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                  kernel_regularizer=l2(l2_lambda))(u2_3)
    c2_3 = Dropout(dropout_rate)(c2_3)
    c2_3 = Conv2D(32, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                  kernel_regularizer=l2(l2_lambda))(c2_3)

    u1_4 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c2_3)
    u1_4 = concatenate([u1_4, c1_0])
    c1_4 = Conv2D(16, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                  kernel_regularizer=l2(l2_lambda))(u1_4)
    c1_4 = Dropout(dropout_rate)(c1_4)
    c1_4 = Conv2D(16, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same',
                  kernel_regularizer=l2(l2_lambda))(c1_4)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c1_4)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

