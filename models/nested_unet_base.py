from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def nested_unet(input_shape=(512, 512, 1), dropout_rate=0.1, l2_lambda=0.0, activation_function='relu'):
    activation = activation_function
    
    inputs = Input(shape=input_shape)
    
    # Contracting Path (Nested)
    c1_1 = Conv2D(16, (3, 3), activation=activation, kernel_initializer='he_normal',
                  padding='same', kernel_regularizer=l2(l2_lambda))(inputs)
    c1_1 = Dropout(dropout_rate)(c1_1)
    c1_1 = Conv2D(16, (3, 3), activation=activation, kernel_initializer='he_normal', 
                  padding='same', kernel_regularizer=l2(l2_lambda))(c1_1)
    p1 = MaxPooling2D((2, 2))(c1_1)
    
    c2_1 = Conv2D(32, (3, 3), activation=activation, kernel_initializer='he_normal', 
                  padding='same', kernel_regularizer=l2(l2_lambda))(p1)
    c2_1 = Dropout(dropout_rate)(c2_1)
    c2_1 = Conv2D(32, (3, 3), activation=activation, kernel_initializer='he_normal', 
                  padding='same', kernel_regularizer=l2(l2_lambda))(c2_1)
    p2 = MaxPooling2D((2, 2))(c2_1)
    
    c3_1 = Conv2D(64, (3, 3), activation=activation, kernel_initializer='he_normal', 
                  padding='same', kernel_regularizer=l2(l2_lambda))(p2)
    c3_1 = Dropout(dropout_rate)(c3_1)
    c3_1 = Conv2D(64, (3, 3), activation=activation, kernel_initializer='he_normal', 
                  padding='same', kernel_regularizer=l2(l2_lambda))(c3_1)
    p3 = MaxPooling2D((2, 2))(c3_1)

    c4_1 = Conv2D(128, (3, 3), activation=activation, kernel_initializer='he_normal', 
                  padding='same', kernel_regularizer=l2(l2_lambda))(p3)
    c4_1 = Dropout(dropout_rate)(c4_1)
    c4_1 = Conv2D(128, (3, 3), activation=activation, kernel_initializer='he_normal', 
                  padding='same', kernel_regularizer=l2(l2_lambda))(c4_1)
    p4 = MaxPooling2D((2, 2))(c4_1)
    
    c5_1 = Conv2D(256, (3, 3), activation=activation, kernel_initializer='he_normal', 
                  padding='same', kernel_regularizer=l2(l2_lambda))(p4)
    c5_1 = Dropout(dropout_rate)(c5_1)
    c5_1 = Conv2D(256, (3, 3), activation=activation, kernel_initializer='he_normal', 
                  padding='same', kernel_regularizer=l2(l2_lambda))(c5_1)

    # Expansive Path (Nested)
    u6_1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5_1)
    u6_1 = concatenate([u6_1, c4_1])
    c6_1 = Conv2D(128, (3, 3), activation=activation, kernel_initializer='he_normal', 
                  padding='same', kernel_regularizer=l2(l2_lambda))(u6_1)
    c6_1 = Dropout(dropout_rate)(c6_1)
    c6_1 = Conv2D(128, (3, 3), activation=activation, kernel_initializer='he_normal', 
                  padding='same', kernel_regularizer=l2(l2_lambda))(c6_1)

    # Adding Nested Path
    u7_1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6_1)
    u7_1 = concatenate([u7_1, c3_1])
    c7_1 = Conv2D(64, (3, 3), activation=activation, kernel_initializer='he_normal', 
                  padding='same', kernel_regularizer=l2(l2_lambda))(u7_1)
    c7_1 = Dropout(dropout_rate)(c7_1)
    c7_1 = Conv2D(64, (3, 3), activation=activation, kernel_initializer='he_normal', 
                  padding='same', kernel_regularizer=l2(l2_lambda))(c7_1)

    # More Nested Path
    u8_1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7_1)
    u8_1 = concatenate([u8_1, c2_1])
    c8_1 = Conv2D(32, (3, 3), activation=activation, kernel_initializer='he_normal', 
                  padding='same', kernel_regularizer=l2(l2_lambda))(u8_1)
    c8_1 = Dropout(dropout_rate)(c8_1)
    c8_1 = Conv2D(32, (3, 3), activation=activation, kernel_initializer='he_normal', 
                  padding='same', kernel_regularizer=l2(l2_lambda))(c8_1)

    u9_1 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8_1)
    u9_1 = concatenate([u9_1, c1_1])
    c9_1 = Conv2D(16, (3, 3), activation=activation, kernel_initializer='he_normal', 
                  padding='same', kernel_regularizer=l2(l2_lambda))(u9_1)
    c9_1 = Dropout(dropout_rate)(c9_1)
    c9_1 = Conv2D(16, (3, 3), activation=activation, kernel_initializer='he_normal', 
                  padding='same', kernel_regularizer=l2(l2_lambda))(c9_1)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9_1)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

