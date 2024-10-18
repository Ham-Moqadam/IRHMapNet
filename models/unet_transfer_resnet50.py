

from keras.applications import VGG16
from keras.layers import Input, Conv2D, UpSampling2D, concatenate
from keras.models import Model

def create_unet_transfer_vgg_shallow(input_shape=(512, 512, 1), num_classes=1):
    # Define input layer
    inputs = Input(shape=input_shape)

    # Base model (VGG16) without the top layers
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)

    # Use only the first two blocks to reduce memory usage
    c1 = base_model.get_layer('block1_conv2').output  # 256x256
    c2 = base_model.get_layer('block2_conv2').output  # 128x128

    # U-Net decoder path (upsampling path)
    u7 = UpSampling2D(size=(2, 2))(c2)  # 256x256
    u7 = concatenate([u7, c1])  # Concatenate with corresponding encoder layer
    u7 = Conv2D(64, (3, 3), padding='same', activation='relu')(u7)

    # Output layer
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(u7)  # Final output layer

    # Create the model
    model = Model(inputs=[inputs], outputs=[outputs])

    return model



"""
from keras.applications import VGG16
from keras.layers import Input, Conv2D, UpSampling2D, concatenate
from keras.models import Model

def create_unet_transfer(input_shape=(512, 512, 1), num_classes=1):
    inputs = Input(shape=input_shape)

    # Base model (VGG16) without the top layers
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)

    # Use appropriate layers to concatenate
    c1 = base_model.get_layer('block1_conv2').output  # 256x256
    c2 = base_model.get_layer('block2_conv2').output  # 128x128
    c3 = base_model.get_layer('block3_conv3').output  # 64x64
    c4 = base_model.get_layer('block4_conv3').output  # 32x32

    # U-Net decoder path
    u6 = UpSampling2D(size=(2, 2))(c4)  # 64x64
    u6 = concatenate([u6, c3])
    u6 = Conv2D(256, (3, 3), padding='same')(u6)

    u7 = UpSampling2D(size=(2, 2))(u6)  # 128x128
    u7 = concatenate([u7, c2])
    u7 = Conv2D(128, (3, 3), padding='same')(u7)

    u8 = UpSampling2D(size=(2, 2))(u7)  # 256x256
    u8 = concatenate([u8, c1])
    u8 = Conv2D(64, (3, 3), padding='same')(u8)

    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(u8)  # Final output layer
    model = Model(inputs=[inputs], outputs=[outputs])

    return model


### -------------------------------------------------------------------------------------

from keras.applications import ResNet50
from keras.layers import Input, Conv2D, UpSampling2D, concatenate
from keras.models import Model

def create_unet_transfer(input_shape=(512, 512, 1), num_classes=1):
    inputs = Input(shape=input_shape)

    # Base model (ResNet50) without the top layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

    # Use appropriate layers to concatenate
    c1 = base_model.get_layer('conv1_relu').output  # 256x256
    c2 = base_model.get_layer('conv2_block3_out').output  # 128x128
    c3 = base_model.get_layer('conv3_block4_out').output  # 64x64
    c4 = base_model.get_layer('conv4_block6_out').output  # 32x32

    # U-Net decoder path
    u6 = UpSampling2D(size=(2, 2))(c4)  # 64x64
    u6 = concatenate([u6, c3])
    u6 = Conv2D(256, (3, 3), padding='same')(u6)

    u7 = UpSampling2D(size=(2, 2))(u6)  # 128x128
    u7 = concatenate([u7, c2])
    u7 = Conv2D(128, (3, 3), padding='same')(u7)

    u8 = UpSampling2D(size=(2, 2))(u7)  # 256x256
    u8 = concatenate([u8, c1])
    u8 = Conv2D(64, (3, 3), padding='same')(u8)

    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(u8)  # Final output layer
    model = Model(inputs=[inputs], outputs=[outputs])

    return model
"""
