import os
import keras
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.applications import vgg16, inception_resnet_v2

f = open("predictions.txt","w+")

vgg_model = vgg16.VGG16(weights='imagenet')

for file in os.listdir("./generatedImgs"):
    original = load_img("./generatedImgs/"+file, target_size=(224, 224))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)

    processed_image = vgg16.preprocess_input(image_batch.copy())

    predictions = vgg_model.predict(processed_image)
    label = decode_predictions(predictions)
    f.write(str(label[0]))
f.close()
