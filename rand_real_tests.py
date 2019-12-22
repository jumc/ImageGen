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
            intermediate_output = intermediate_layer_model.predict(selected_imgs)
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
                if j in rand_classes:
                    y_i = [el == j for el in rand_classes]
                    plt.scatter(tx[y_i], ty[y_i], label=imagenet_labels_list[j-1])
                    y_i = [el == j for el in imagenet_classes]
                    plt.scatter(imagenet_tx[y_i], imagenet_ty[y_i], marker="x", label=imagenet_labels_list[j-1])
            plt.legend(loc=4)
            plt.gca().invert_yaxis()
            plt.savefig('./dimensionalityReductionTest/'+layer_names[i]+'.png', bbox_inches='tight')

def meanPredProb(pred):
    # Average chance of prediction for a class in rand images
    avg = np.average(pred, axis=0)
    index = []
    nMax = 10
    for i in range(nMax):
        arg = avg.argmax()
        index.append([arg, avg[arg]]) # format: [class, mean prediction probability]
        avg[arg] = 0
    print(index)
    result = [[549, 0.2565923], [723, 0.14126533], [409, 0.054824915], [892, 0.03461164], [478, 0.024420911], [419, 0.022606201], [916, 0.022001646], [530, 0.015288568], [446, 0.015227827] , [921, 0.014206272]] # Class 446 (only 16 avaliable imgs above tol)

def SVMTestRealToRand(imagenet_intermediate_output, rand_intermediate_output, clf):
    #SVM test - trained on real, tested on rand using layer before prediction
    clf.fit(imagenet_intermediate_output, y_test_imagenet)
    result = clf.predict(rand_intermediate_output)
    print(result)
    print(y_test_randImg)
    a=[549, 549, 549, 769, 549, 426, 723, 769, 426, 549, 549, 549, 426, 507, 549, 692, 426, 549, 692, 419, 723, 419, 549, 692, 549, 549, 426, 549, 549, 692, 419, 419, 692, 692, 549, 549, 902, 692, 549, 549, 692, 549, 692, 692, 549, 549, 902, 549, 549, 549, 549, 549, 902, 723, 419, 419, 692, 419, 692, 902, 549, 549, 692, 549, 426, 549, 723, 419, 902, 723, 549, 549, 419, 419, 769, 723, 692, 426, 723, 419, 549, 769, 549, 419, 692, 692, 426, 549, 549, 549, 692, 692, 419, 723, 723, 549, 549, 692, 549, 692, 419, 426, 692, 723, 692, 549, 692, 549, 419, 692, 692, 798, 549, 798, 692, 549, 692, 723, 549, 769, 692, 723, 549, 692, 912, 692, 549, 723, 419, 723, 426, 769, 549, 549, 549, 549, 549, 549, 664, 549, 549, 549, 549, 426, 549, 419, 692, 549, 549, 692, 692, 692, 549, 426, 692, 692, 692, 426, 692, 723, 426, 912, 549, 692, 549, 692, 549, 426, 549, 549, 549, 549, 549, 426, 419, 549, 549, 549, 549, 549, 692, 769, 549, 692, 419, 549, 902, 723, 549, 549, 549, 549, 419, 426, 723, 419, 549, 549, 419, 549, 912, 549, 549, 692, 419, 723, 549, 723, 419, 549, 549, 769, 549, 549, 549, 692, 419, 549, 723, 692, 419, 692, 549, 912, 549, 419, 426, 549, 549, 723, 912, 723, 692, 723, 549, 723, 426, 426, 549, 549, 419, 419, 723, 419, 549, 426, 419, 549, 549, 419, 549, 549, 692, 549, 549, 769, 507, 549, 419, 419, 549, 692, 769, 549, 419, 723, 902, 549, 549, 426, 549, 692, 549, 692, 549, 692, 692, 426, 902, 692, 692, 723, 692, 419, 549, 507, 426, 549, 419, 549, 912, 549, 723, 723, 692, 692, 664, 692, 549, 549, 723, 692, 419, 549, 419, 419, 419, 692, 419, 549, 692, 549, 549, 419, 419, 769, 723, 419, 419, 549, 692, 912, 549, 419, 723, 419, 782, 549, 692, 426, 692, 692, 549, 692, 426, 692, 692, 549, 902, 723, 549, 549, 692, 549, 912, 419, 549, 912, 549, 419, 549, 692, 692, 769, 769, 769, 549, 692, 549, 692, 549, 426, 692, 692, 549, 692, 426, 549, 419, 549, 549, 692, 426, 782, 692, 549, 692, 549, 549, 549, 549, 692, 769, 507, 723, 549, 419, 549, 426, 769, 798, 692, 798, 549, 419, 769, 419, 692, 549, 692, 549, 549, 419, 769, 549, 549, 419, 769, 769, 692, 902, 798, 723, 723, 692, 723, 549, 549, 902, 692, 419, 549, 549, 549, 549, 624, 549, 549, 692, 549, 549, 419, 549, 419, 549, 769, 426, 549, 426, 507, 549, 549, 549, 549, 723, 549, 419, 419, 426, 723, 426, 507, 723, 692, 507, 549, 426, 419, 723, 912, 692, 549, 902, 692, 692, 692]
    b=[549, 549, 549, 530, 478, 892, 723, 478, 892, 892, 530, 530, 892, 530, 892, 921, 892, 916, 478, 419, 723, 419, 892, 892, 549, 409, 892, 892, 409, 921, 419, 419, 921, 409, 916, 549, 530, 921, 549, 916, 916, 916, 478, 478, 916, 409, 446, 916, 549, 916, 549, 916, 530, 723, 419, 419, 478, 419, 723, 530, 921, 549, 921, 921, 530, 530, 478, 530, 530, 723, 478, 478, 530, 419, 409, 723, 478, 409, 723, 419, 549, 409, 916, 419, 478, 409, 892, 892, 723, 723, 921, 916, 916, 723, 723, 921, 549, 478, 892, 921, 419, 409, 478, 723, 921, 478, 921, 409, 530, 921, 723, 916, 478, 916, 478, 478, 478, 723, 921, 409, 409, 723, 723, 530, 446, 478, 478, 723, 419, 723, 409, 409, 916, 478, 478, 921, 916, 549, 409, 409, 892, 916, 530, 409, 409, 419, 921, 409, 921, 892, 921, 921, 549, 892, 921, 478, 921, 892, 892, 723, 892, 530, 916, 921, 916, 921, 921, 409, 916, 892, 916, 549, 549, 892, 530, 549, 916, 916, 916, 549, 916, 892, 478, 921, 530, 478, 530, 723, 549, 549, 549, 916, 419, 892, 723, 530, 446, 549, 419, 921, 446, 446, 549, 723, 478, 723, 549, 723, 549, 916, 446, 419, 549, 530, 549, 478, 419, 478, 723, 446, 419, 530, 892, 530, 530, 419, 892, 723, 892, 723, 530, 723, 892, 723, 892, 723, 892, 892, 892, 892, 419, 419, 723, 419, 723, 892, 419, 892, 530, 419, 549, 478, 478, 916, 478, 530, 530, 921, 419, 530, 549, 921, 409, 478, 530, 723, 530, 478, 892, 409, 409, 530, 549, 530, 478, 478, 409, 892, 916, 921, 530, 723, 530, 419, 916, 530, 409, 446, 419, 892, 530, 549, 723, 723, 921, 921, 409, 478, 921, 549, 723, 478, 419, 916, 419, 419, 419, 921, 419, 409, 478, 409, 549, 419, 419, 892, 723, 419, 419, 478, 921, 478, 549, 916, 723, 419, 409, 921, 409, 409, 723, 478, 892, 921, 409, 921, 921, 478, 916, 723, 921, 549, 921, 723, 409, 419, 478, 478, 409, 419, 549, 921, 478, 916, 916, 916, 446, 921, 478, 478, 549, 892, 530, 409, 916, 921, 409, 916, 419, 916, 549, 921, 409, 409, 921, 549, 921, 916, 409, 409, 892, 409, 892, 530, 409, 446, 419, 549, 409, 409, 916, 530, 916, 409, 419, 892, 419, 921, 549, 921, 916, 549, 419, 530, 549, 892, 419, 409, 892, 478, 916, 916, 723, 723, 892, 723, 446, 549, 530, 916, 419, 446, 916, 549, 446, 446, 892, 409, 530, 549, 446, 419, 549, 419, 723, 916, 892, 549, 409, 530, 549, 549, 549, 446, 723, 530, 530, 419, 892, 723, 530, 530, 723, 409, 530, 916, 892, 419, 723, 916, 478, 409, 530, 478, 921, 921]
    count = 0
    for i in range(len(a)):
        if a[i] != b[i]:
           count += 1
    print((len(a)-count)/len(a)) #accuracy 0.293991416

def SVMTestRandToReal(imagenet_intermediate_output, rand_intermediate_output, clf):
    #SVM test - trained on rand, tested on real using layer before prediction
    clf.fit(rand_intermediate_output, y_test_randImg)
    result = clf.predict(imagenet_intermediate_output)
    print(result)
    print(y_test_imagenet)
    a=[478, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 478, 916, 916, 916, 916, 916, 916, 916, 916, 916, 921, 921, 921, 921, 921, 916, 916, 921, 916, 916, 921, 921, 478, 478, 478, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 446, 921, 921, 916, 916, 921, 921, 921, 916, 921, 921, 916, 921, 921, 921, 921, 921, 921, 921, 921, 916, 921, 921, 921, 921, 916, 921, 916, 921, 921, 916, 921, 921, 921, 921, 419, 921, 921, 921, 916, 921, 916, 916, 921, 921, 916, 916, 916, 916, 921, 916, 916, 916, 916, 916, 916, 409, 419, 916, 916, 916, 916, 916, 916, 916, 446, 916, 916, 921, 916, 916, 921, 409, 916, 921, 916, 916, 916, 916, 921, 916, 446, 916, 916, 916, 916, 916, 916, 478, 916, 530, 916, 916, 892, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 478, 921, 921, 921, 921, 921, 921, 921, 921, 921, 446, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 921, 478, 921, 478, 478, 478, 419, 921, 478, 478, 478, 478, 478, 478, 478, 478, 478, 419, 478, 478, 478, 478, 478, 419, 478, 419, 478, 478, 446, 892, 478, 419, 921, 478, 478, 478, 419, 478, 478, 478, 478, 478, 478, 478, 478, 478, 478, 478, 478, 478, 921, 921, 419, 921, 916, 921, 916, 921, 419, 921, 921, 478, 549, 478, 478, 478, 419, 478, 921, 478, 921, 478, 921, 478, 478, 921, 921, 921, 419, 478, 921, 478, 921, 478, 419, 921, 921, 921, 419, 419, 419, 921, 549, 419, 549, 723, 478, 921, 921, 916, 419, 921, 916, 916, 916, 916, 916, 916, 916, 916, 916, 916, 478, 916, 916, 921, 419]
    b=[716, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 365, 98, 98, 98, 98, 98, 98, 20, 98, 98, 98, 98, 98, 98, 98, 142, 88, 448, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 417, 88, 88, 88, 90, 88, 88, 88, 88, 88, 98, 98, 52, 58, 53, 823, 823, 798, 798, 798, 912, 489, 422, 422, 422, 692, 98, 98, 137, 98, 98, 98, 98, 98, 98, 143, 98, 98, 98, 98, 98, 98, 98, 97, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 143, 53, 53, 54, 53, 53, 53, 858, 53, 600, 53, 53, 53, 53, 53, 53, 53, 53, 62, 53, 53, 53, 53, 53, 52, 53, 53, 53, 52, 506, 53, 58, 53, 53, 52, 57, 53, 53, 58, 53, 53, 53, 53, 56, 53, 53, 53, 53, 739, 617, 823, 823, 261, 467, 617, 823, 765, 650, 758, 823, 823, 823, 823, 902, 550, 823, 870, 823, 823, 823, 823, 902, 823, 823, 823, 823, 823, 841, 823, 798, 577, 769, 823, 414, 823, 823, 823, 823, 775, 617, 782, 823, 823, 823, 830, 585, 769, 971, 823, 507, 798, 798, 426, 798, 798, 798, 798, 798, 798, 798, 798, 798, 798, 798, 798, 798, 798, 798, 798, 769, 798, 798, 798, 798, 798, 975, 912, 912, 912, 912, 912, 912, 912, 912, 912, 912, 912, 912, 912, 912, 912, 23, 912, 912, 912, 912, 912, 876, 912, 912, 15, 912, 839, 912, 912, 912, 912, 912, 912, 912, 912, 912, 47, 912, 912, 912, 425, 912, 456, 912, 912, 46, 912, 422, 422, 422, 543, 733, 422, 422, 422, 422, 422, 422, 422, 422, 422, 719, 422, 422, 422, 422, 422, 422, 422, 543, 870, 422, 422, 783, 422, 422, 422, 422, 422, 422, 422, 422, 422, 422, 422, 422, 422, 422, 422, 422, 422, 422, 422, 422, 582, 886, 767, 709, 692, 692, 664, 582, 692, 692, 692, 707, 549, 519, 692, 692, 419, 713, 692, 886, 692, 692, 750, 692, 692, 692, 760, 692, 692, 860, 692, 499, 692, 519, 692, 505, 692, 692, 720, 692, 419, 692, 692, 692, 692, 723, 692, 692, 827, 798, 563, 798, 703, 798, 798, 798, 798, 798, 798, 798, 798, 798, 624, 798, 798, 798, 418]
    count = 0
    for i in range(len(a)):
        if a[i] != b[i]:
           count += 1
    print((len(a)-count)/len(a)) #accuracy 0.01

def SVMTest():
    intermediate_layer_model = keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_names[-1]).output)
    rand_intermediate_output = intermediate_layer_model.predict(rand_images)
    print("Predicted rand_intermediate_output")
    imagenet_intermediate_output = intermediate_layer_model.predict(imagenet_images)
    print("Predicted imagenet_intermediate_output")
    clf = LinearSVC(random_state=0, tol=1e-5)
    print("SVM test - trained on real, tested on rand")
    SVMTestRealToRand(imagenet_intermediate_output, rand_intermediate_output, clf)
    print("SVM test - trained on rand, tested on real")
    SVMTestRandToReal(imagenet_intermediate_output, rand_intermediate_output, clf)

def selectRandImgs():
    # Selecting random test images that match the imagenet samples, predicted above tolerance
    selected_classes = {549:0, 409:0, 419:0, 478:0, 530:0, 723:0, 892:0, 916:0, 921:0}
    index = 0
    pred = []
    n_images = len(os.listdir("./generatedImgs/"))
    print("Total images " + str(n_images))
    imgsNeeded = len(selected_classes) * 50
    batch_size = 64
    while (imgsNeeded > 0 and index < n_images - batch_size - 1):
        images = []
        for i in range(batch_size):
            original = load_img("./generatedImgs/"+str(index + i)+".png", target_size=(224, 224))
            numpy_image = img_to_array(original)
            image_batch = np.expand_dims(numpy_image, axis=0)
            processed_image = vgg16.preprocess_input(image_batch.copy())
            images.append(processed_image)
        pred_images = np.vstack(images)
        print("Random images batch up to " + str(index + i))

        all_pred = model.predict(pred_images)
        all_y_test = [np.argmax(prediction) for prediction in all_pred]
        print("Predicted classes for the batch")

        # selecting images where prediction is over tolerance
        tol = 0.01
        selected_imgs = []
        #img_classes = [] # turn on for dimensionalityReductionTest
        for i in range(batch_size):
            if np.argmax(all_pred[i]) > tol and all_y_test[i] in selected_classes and selected_classes[all_y_test[i]] < 50:
                selected_imgs.append(images[i])
                os.system("cp ./generatedImgs/"+str(index + i)+".png ./generatedImgsSelected")
                selected_classes[all_y_test[i]] += 1
                imgsNeeded -= 1
                #img_classes.append(all_y_test[i]) # turn on for dimensionalityReductionTest
        print(selected_classes)
        if(len(selected_imgs) > 0):
            pred.append(model.predict(np.vstack(selected_imgs)))
        index += batch_size

def loadRandImgs():
#loading imagenet samples
    rand_images = []
    for file in os.listdir("./generatedImgsSelected"):
        original = load_img("./generatedImgsSelected/"+file, target_size=(224, 224))
        numpy_image = img_to_array(original)
        image_batch = np.expand_dims(numpy_image, axis=0)
        processed_image = vgg16.preprocess_input(image_batch.copy())
        rand_images.append(processed_image)
    rand_images = np.vstack(rand_images)
    print("Loaded random samples")

    rand_pred = model.predict(rand_images)
    y_test_randImg = [np.argmax(prediction) for prediction in rand_pred]
    print("Predicted random samples class")

    return rand_images, rand_pred, y_test_randImg

def loadImagenet():
    #loading imagenet samples
    imagenet_images = []
    # imagenet_classes = [] # turn on for dimensionalityReductionTest
    for file in os.listdir("./imagenetSamples"):
        original = load_img("./imagenetSamples/"+file, target_size=(224, 224))
        numpy_image = img_to_array(original)
        image_batch = np.expand_dims(numpy_image, axis=0)
        # imagenet_classes.append(int(file.split("_")[0])) # turn on for dimensionalityReductionTest
        processed_image = vgg16.preprocess_input(image_batch.copy())
        imagenet_images.append(processed_image)
    imagenet_images = np.vstack(imagenet_images)
    print("Loaded imagenet samples")

    imagenet_pred = model.predict(imagenet_images)
    y_test_imagenet = [np.argmax(prediction) for prediction in imagenet_pred]
    print("Predicted imagenet samples class")

    return imagenet_images, imagenet_pred, y_test_imagenet

# Loading vgg model
num_classes = 1000
model = vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=num_classes)
layer_names = [layer.name for layer in model.layers]
print("Loaded VGG model")

# Loading "dataset"
rand_images, rand_pred, y_test_randImg = loadRandImgs()
imagenet_images, imagenet_pred, y_test_imagenet = loadImagenet()
