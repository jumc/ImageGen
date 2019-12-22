import os
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import resnet50
from keras.models import load_model

pca = PCA(n_components=100)
scaler = StandardScaler()

# Loading resnet50 model
num_classes = 1000
resnet = resnet50.ResNet50(weights='resnet50_weights_tf_dim_ordering_tf_kernels.h5', include_top=True)
resnet_layer_names = [layer.name for layer in resnet.layers]
resnet_intermediate_layer_model = keras.models.Model(inputs=resnet.input, outputs=resnet.get_layer(resnet_layer_names[-1]).output)
print("Loaded ResNet50 model")

# Loading VOC2012 actions
actions = ['jumping', 'phoning', 'playinginstrument', 'reading', 'ridingbike', 'ridinghorse', 'running', 'takingphoto', 'usingcomputer', 'walking']
loaded_imgs = set()
all_latent_vecs = []
y = [] # index correspond to all_latent_vecs
for i in range(len(actions)):
    filenames = []
    f = open("./VOC2012/ImageSets/Action/" + actions[i] + "_train.txt", "r")
    contents = f.read().split()
    f.close()
    j = 0
    count = 0
    while j < len(contents):
        if contents[j+2] == '1':
            if contents[j] not in loaded_imgs:
                count += 1
                loaded_imgs.add(contents[j])
                filenames.append("./VOC2012/JPEGImages/" + contents[j] + ".jpg")
        j += 3
    print(actions[i], count)

    imgs = [img_to_array(load_img(file, target_size=(224, 224))) for file in filenames]
    imgs = np.stack(imgs)
    preprocess_img = resnet50.preprocess_input(imgs)
    latent_vecs = resnet_intermediate_layer_model.predict(preprocess_img)

    for j, vec in enumerate(latent_vecs):
        new_name = "./VOC2012/resnet_latent_vecs/" + filenames[j][21:-4]
        np.save(new_name, vec)

    all_latent_vecs.extend(latent_vecs)
    y.extend([i for k in range(count)])

np.save("./VOC2012/resnet_latent_vecs/y", y)

scaler.fit(all_latent_vecs)
all_latent_vecs = scaler.transform(all_latent_vecs)
pca.fit(all_latent_vecs)
all_latent_vecs = pca.transform(all_latent_vecs)

np.save("./VOC2012/resnet_latent_vecs/scaled_PCA", all_latent_vecs)

print("Loaded and predicted all classes")
