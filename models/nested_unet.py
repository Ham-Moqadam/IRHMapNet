# unet_nested.py
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def conv_block(x, filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal', l2_lambda=0.0):
    x = Conv2D(filters, kernel_size, activation=activation, padding=padding, kernel_initializer=kernel_initializer,
               kernel_regularizer=l2(l2_lambda))(x)
    x = Dropout(0.1)(x)
    x = Conv2D(filters, kernel_size, activation=activation, padding=padding, kernel_initializer=kernel_initializer,
               kernel_regularizer=l2(l2_lambda))(x)
    return x

def NestedUnet(input_shape=(512, 512, 1), filters=16, dropout_rate=0.1, l2_lambda=0.0, activation='relu'):
    inputs = Input(shape=input_shape)

    # Contracting Path
    c1 = conv_block(inputs, filters, activation=activation, l2_lambda=l2_lambda)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, filters*2, activation=activation, l2_lambda=l2_lambda)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, filters*4, activation=activation, l2_lambda=l2_lambda)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, filters*8, activation=activation, l2_lambda=l2_lambda)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = conv_block(p4, filters*16, activation=activation, l2_lambda=l2_lambda)

    # Expansive Path
    u6 = Conv2DTranspose(filters*8, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, filters*8, activation=activation, l2_lambda=l2_lambda)

    u7 = Conv2DTranspose(filters*4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, filters*4, activation=activation, l2_lambda=l2_lambda)

    u8 = Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block(u8, filters*2, activation=activation, l2_lambda=l2_lambda)

    u9 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block(u9, filters, activation=activation, l2_lambda=l2_lambda)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

