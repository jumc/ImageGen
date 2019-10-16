from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
import keras
from keras import backend as K
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.applications import vgg16, inception_resnet_v2
from imagenet_labels import *

def dimensionalityReductionTest(model):
    # Looking into each layer
    for i in range(1, len(layer_names)):
        print(layer_names[i])

        if('flatten' not in layer_names[i]):
            intermediate_layer_model = keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_names[i]).output)
            intermediate_output = intermediate_layer_model.predict(images[selected_imgs])
            imagenet_intermediate_output = intermediate_layer_model.predict(imagenet_images)
            if(intermediate_output.ndim > 2):
                intermediate_output = np.array([vec.flatten() for vec in intermediate_output[:]])
            if(imagenet_intermediate_output.ndim > 2):
                imagenet_intermediate_output = np.array([vec.flatten() for vec in imagenet_intermediate_output[:]])

            # Dimensionality reduction
            pca = PCA(n_components=2)
            pca.fit(intermediate_output)
            pca_features = pca.transform(intermediate_output)
            tx, ty = pca_features[:,0], pca_features[:,1]

            tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
            ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

            pca.fit(imagenet_intermediate_output)
            pca_features = pca.transform(imagenet_intermediate_output)
            imagenet_tx, imagenet_ty = pca_features[:,0], pca_features[:,1]

            imagenet_tx = (imagenet_tx-np.min(imagenet_tx)) / (np.max(imagenet_tx) - np.min(imagenet_tx))
            imagenet_ty = (imagenet_ty-np.min(imagenet_ty)) / (np.max(imagenet_ty) - np.min(imagenet_ty))

            plt.figure(figsize = (16,12))
            for j in range(num_classes):
                if j in img_classes:
                    y_i = [el == j for el in img_classes]
                    plt.scatter(tx[y_i], ty[y_i], label=imagenet_labels_list[j-1])
                    y_i = [el == j for el in imagenet_classes]
                    plt.scatter(imagenet_tx[y_i], imagenet_ty[y_i], marker="x", label=imagenet_labels_list[j-1])
            plt.legend(loc=4)
            plt.gca().invert_yaxis()
            plt.savefig(layer_names[i]+'.png', bbox_inches='tight')

def meanPredProb(pred):
    avg = np.average(pred, axis=0)
    index = []
    nMax = 10
    for i in range(nMax):
        arg = avg.argmax()
        index.append([arg, avg[arg]]) # format: [class, mean prediction probability]
        avg[arg] = 0
    print(index)
    #[[549, 0.2565923], [723, 0.14126533], [409, 0.054824915], [892, 0.03461164], [478, 0.024420911], [419, 0.022606201], [916, 0.022001646], [530, 0.015288568], [446, 0.015227827], [921, 0.014206272]]

def SVMTestRandToReal():
    intermediate_layer_model = keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_names[-1]).output)
    intermediate_output = intermediate_layer_model.predict(images[selected_imgs])
    imagenet_intermediate_output = intermediate_layer_model.predict(imagenet_images) #test
    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(intermediate_output, y_test_randImg)
    result = clf.predict(imagenet_intermediate_output)
    print(result)
    print(y_test_imagenet)

def SVMTestRealToRand():
    intermediate_layer_model = keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_names[-1]).output)
    intermediate_output = intermediate_layer_model.predict(images[selected_imgs])
    imagenet_intermediate_output = intermediate_layer_model.predict(imagenet_images) #test
    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(imagenet_intermediate_output, y_test_imagenet)
    result = clf.predict(intermediate_output)
    print(result)
    print(y_test_randImg)

# Loading model
num_classes = 1000
model = vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=num_classes)

# Loading random test images
images = []
for file in os.listdir("./generatedImgs"):
    original = load_img("./generatedImgs/"+file, target_size=(224, 224))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)

    processed_image = vgg16.preprocess_input(image_batch.copy())
    images.append(processed_image)
images = np.vstack(images)

#loading imagenet samples
imagenet_images = []
imagenet_classes = []
for file in os.listdir("./imagenetSamples"):
    original = load_img("./imagenetSamples/"+file, target_size=(224, 224))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    imagenet_classes.append(int(file.split("_")[0]))
    processed_image = vgg16.preprocess_input(image_batch.copy())
    imagenet_images.append(processed_image)
imagenet_images = np.vstack(imagenet_images)
imagenet_plt_labels = [imagenet_labels_list[lbl - 1] for lbl in imagenet_classes]

# Preparing to process each layer
outputs = []
layer_names = [layer.name for layer in model.layers]
pred = model.predict(images)
#y_test_randImg = [np.argmax(prediction) for prediction in pred]
imagenet_pred = model.predict(imagenet_images)
y_test_imagenet = [np.argmax(prediction) for prediction in imagenet_pred]

# selecting images where prediction is over tolerance
#tol = 0.45
#selected_imgs = []
#img_classes = []
#for i in range(len(images)):
#    if max(pred[i]) > tol :
#        selected_imgs.append(i) # index where the image in the images array has high pred
        #img_classes.append(y_test_randImg[i])
selected_imgs = range(len(images))

pred = model.predict(images[selected_imgs])
y_test_randImg = [np.argmax(prediction) for prediction in pred]

SVMTestRealToRand()
SVMTestRandToReal()

    # 409 macaw
    # 549 slide_rule
    # 723 worm_fence
