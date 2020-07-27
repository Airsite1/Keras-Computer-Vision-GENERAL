from tensorflow.keras.preprocessing.images import ImageDataGenerator
from tensorflow.keras.optimizers import *
import tensorflow as tf


train_data_gen= ImageDataGenerator(rescale=1.0/255)
train_generator= train_data_gen.flow_from_directory(
    train_dir, #make sure this points to root directory, not sub
    target_size= (300,300), #uniformly pre-processes images at runtime
    batch_size=128, #more efficient with different batch sizes
    class_mode='binary' #in this case chooses between horses and humans (2)
)

model= tf.keras.model.Sequential([
tf.keras.layers.Conv2D(16, (3,3), activation='relu', 
input_shape=(300, 300, 3)),#Input shape 3 is for rgb
tf.keras.layers.MaxPooling2D(2, 2),
tf.keras.layers.Conv2D(32, (3, 3)),
tf.keras.layers.MaxPooling2D(2, 2),
tf.keras.layers.Conv2D(64, (3, 3)),
tf.keras.layers.MaxPooling2D(2, 2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation='relu'),
#one output node with sigmoid as its better for binary. 
#U could use 2 nodes and softmax if u want but sigmoid is better
tf.keras.layers.Dense(1, activation='sigmoid')  
])
model.compile(loss='binary_crossentropy',
                optimizer=RMSprop(lr=0.001), #lr= learning rate
                metrics=['accuracy'] )
history = model.fit_generator(
    train_generator, 
    steps_per_epoch=8, #1024 images/batches of 128 = 8
    epochs=15, 
    validation_data= validation_generator, 
    validation_steps=8, #256 images/batchest of 32= 8
    verbose=2 #how much to display when training occurs.@2, hides epoch progress
)

import numpy as np
from google.colab import files
from keras.preprocessing import images
uploaded= files.upload()
for fn in uploaded.keys():
    #predicting images
    path= '/contents/' + fn
    img= image.load_img(path, target_size=(300, 300))
    x= image.img_to_array(img)
    x= np.expand_dims(x, axis=0)
    
    images= np.vstack([x])
    classes= model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0] > 0.5:
        print(fn + "is a Human")
    else:
        print(fn + "is a Horse")