import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from load_yolo import BoundBox

actions = ['jumping', 'phoning', 'playinginstrument', 'reading', 'ridingbike', 'ridinghorse', 'running', 'takingphoto', 'usingcomputer', 'walking']

resnet_X = np.load("./VOC2012/resnet_latent_vecs/scaled_PCA.npy")
resnet_y = np.load("./VOC2012/resnet_latent_vecs/y.npy", allow_pickle=True)

resnet_X_train, resnet_X_test, resnet_y_train, resnet_y_test = train_test_split(resnet_X, resnet_y, test_size=0.33)

scaler = StandardScaler()
minMaxScaler = MinMaxScaler()

def get_yolo_vecs(yolo_X):
    all_semantic_vecs = []
    all_labels = []
    labels_variations = {}
    variations_count = 0
    encoded_labels = []

    for image_boxes in yolo_X:
        semantic_vecs = []
        labels = []
        n_boxes = len(image_boxes)
        for i in range(10):
            if i < n_boxes:
                semantic_vec, label = image_boxes[i].get_normalized_attr()
                semantic_vecs.extend(semantic_vec)
                labels.append(label)
            else:
                semantic_vecs.extend([0,0,0,0])
                labels.append(80)
        all_semantic_vecs.append(semantic_vecs)
        all_labels.append(labels)
        if tuple(labels) in labels_variations:
            encoded_labels.append(labels_variations[tuple(labels)])
        else:
            labels_variations[tuple(labels)] = variations_count
            encoded_labels.append(variations_count)
            variations_count += 1

    all_semantic_vecs = scaler.fit_transform(all_semantic_vecs)

    return all_semantic_vecs, encoded_labels, all_labels

yolo_boxes = np.load("./VOC2012/yolo_semantic_vecs/all_latent_vecs.npy", allow_pickle=True)

yolo_X, yolo_encoded_y, yolo_y = get_yolo_vecs(yolo_boxes)
yolo_X_train, yolo_X_test, yolo_y_train, yolo_y_test = train_test_split(yolo_X, yolo_encoded_y, test_size=0.33)

def SVMTest(X_train, y_train, X_test, y_test):
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    print("Train_score: ", train_score)
    test_score = clf.score(X_test, y_test)
    print("Test_score: ", test_score)
    y_pred = clf.predict(X_test)
    #print("y_pred: ", y_pred)
    #print("y_test: ", y_test)
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision: ",metrics.precision_score(y_test, y_pred, average="macro"))
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall: ",metrics.recall_score(y_test, y_pred, average="macro"))

print("Testing accuracy on VGG only")
SVMTest(resnet_X_train, resnet_y_train, resnet_X_test, resnet_y_test)
#Train_score:  0.8038167938931298
#Test_score:  0.521671826625387
#Accuracy:  0.521671826625387
#Precision:  0.5460098855797043
#Recall:  0.5103345019725893


print("Testing accuracy on YOLOv3 only")
SVMTest(yolo_X_train, yolo_y_train, yolo_X_test, yolo_y_test)
#Train_score:  0.39541984732824426
#Test_score:  0.38544891640866874
#Accuracy:  0.38544891640866874
#Precision:  0.014760610782089257
#Recall:  0.028722885693395074

yolo_y = scaler.fit_transform(yolo_y)
#resnet_X = minMaxScaler.fit_transform(resnet_X)

fused_X = [np.append(np.append(resnet_X[i], yolo_X[i]), yolo_y[i]) for i in range(len(resnet_X))]
fused_X_train, fused_X_test, fused_y_train, fused_y_test = train_test_split(fused_X, resnet_y, test_size=0.33)

print("Testing accuracy on VGG16 concatenated with YOLOv3")
SVMTest(fused_X_train, fused_y_train, fused_X_test, fused_y_test)
#Train_score:  0.8534351145038168
#Test_score:  0.5634674922600619
#Accuracy:  0.5634674922600619
#Precision:  0.5659267653012585
#Recall:  0.5340998575802659
