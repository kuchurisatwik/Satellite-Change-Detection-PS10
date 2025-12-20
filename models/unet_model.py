def build_unet_model(input_size=(256, 256, 6)):
    inputs = Input(input_size)
    
    # Split the input into two separate RGB images
    input_image1 = inputs[..., :3]
    input_image2 = inputs[..., 3:]
    
    # Encoder for image 1
    c1_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_image1)
    c1_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1_1)
    c1_1 = Dropout(0.3)(c1_1)  # Dropout after convolutional blocks
    p1_1 = MaxPooling2D((2, 2))(c1_1)

    c2_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1_1)
    c2_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2_1)
    c2_1 = Dropout(0.3)(c2_1)  # Dropout after convolutional blocks
    p2_1 = MaxPooling2D((2, 2))(c2_1)

    c3_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2_1)
    c3_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3_1)
    c3_1 = Dropout(0.3)(c3_1)  # Dropout after convolutional blocks
    p3_1 = MaxPooling2D((2, 2))(c3_1)

    # Encoder for image 2 (same as image 1)
    c1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_image2)
    c1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1_2)
    c1_2 = Dropout(0.3)(c1_2)  # Dropout after convolutional blocks
    p1_2 = MaxPooling2D((2, 2))(c1_2)

    c2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1_2)
    c2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2_2)
    c2_2 = Dropout(0.3)(c2_2)  # Dropout after convolutional blocks
    p2_2 = MaxPooling2D((2, 2))(c2_2)

    c3_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2_2)
    c3_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3_2)
    c3_2 = Dropout(0.3)(c3_2)  # Dropout after convolutional blocks
    p3_2 = MaxPooling2D((2, 2))(c3_2)

    # Bottleneck (concatenate the features from both images)
    c4_1 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3_1)
    c4_1 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4_1)
    c4_1 = Dropout(0.4)(c4_1)  # Dropout after convolutional blocks

    c4_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3_2)
    c4_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4_2)
    c4_2 = Dropout(0.4)(c4_2)  # Dropout after convolutional blocks

    # Combine the features of both images at the bottleneck
    c4 = concatenate([c4_1, c4_2])

    # Decoder
    u5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3_1, c3_2])  # Skip connection from both images encoders
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
    c5 = Dropout(0.3)(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2_1, c2_2])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Dropout(0.3)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)
    c6 = Dropout(0.3)(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1_1, c1_2])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Dropout(0.3)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)
    c7 = Dropout(0.3)(c7)

    # Final output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)
    model = Model(inputs, outputs)
    
    return model
