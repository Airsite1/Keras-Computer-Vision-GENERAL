import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import numpy as np

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

#Load Data
mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels)= mnist.load_data()

#Normalize Data
train_images= train_images/255  
test_images= test_images/255

model=keras.Sequential([
#the first layer in your network should be the same shape as your data.
tf.keras.layers.Flatten(input_shape=(28,28)), #converts the 28 by 28 image and flattens into 1d array
tf.keras.layers.Dense(1024, activation=tf.nn.relu),
tf.keras.layers.Dense(10, activation=tf.nn.softmax)]) #converts into the 10 categories
model.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])


#the number of neurons in the last layer should match the number of classes you are classifying for.
model.fit(train_images, train_labels, epochs=10, callbacks= [callbacks]) 
print("EVALUATION:")
model.evaluate(test_images, test_labels)
#classifications=model.predict(test_images)
#print(classifications[0]+ "=",test_labels[0])


'''
IMPORTANT THINGS TO KEEP IN MIND
- more epochs could lead to overfitting
- more nodes might not be benifical according to law of diminishing returns 
- normalize data: we divide by 255 cuz of rgb. Normalizing it allows model to better predict
        2 reasons:
            - integer division leads to roundoff error. so Float is better
            - easier for model to predict using 0 to 1 instead of 0 to 255
- the first layer in your network should be the same shape as your data.
- the number of neurons in the last layer should match the number of classes you are classifying for
- Callback used to stop training once you reach target loss/accuracy. Should be done on epoch end

'''


