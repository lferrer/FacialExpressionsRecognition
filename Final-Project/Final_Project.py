import csv
import numpy as np
import cv2
import os
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, Isomap
import sys

def loadFiles(root):
    images = []
    labels = []
    classes = []
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    minSize = 100000000000000
    for dirs in os.walk(root):
        for dir in dirs[1]:
            newDir = root + "\\" + dir            
            for subDirs in os.walk(newDir):
                for subDir in subDirs[1]:
                    newSubDir = newDir + "\\" + subDir
                    for subSubDirs in os.walk(newSubDir):
                        files = len(subSubDirs[2])
                        #Only load the peak frame!
                        file = subSubDirs[2][files - 2]
                        filename =  newSubDir + "\\" + file
                        rawImg = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                        # Crop the image to the face only
                        faces = face_cascade.detectMultiScale(rawImg)
                        [x,y,w,h] = faces[0]
                        img = rawImg[y:y+h, x:x+w]
                        images.append(img)
                        if w*h < minSize:
                            minSize = w*h
                        # Load the Action Units
                        textFile = newSubDir + "\\" + subSubDirs[2][files - 1]
                        aus = []
                        with open(textFile, 'rt') as file:
                            data = csv.reader(file)
                            dataset = list(data)
                            for x in range(len(dataset)):
                                au = dataset[x][0].split()[0]
                                au = int(float(au))                               
                                aus.append(au)
                                if au not in classes:
                                    classes.append(au)
                        labels.append(np.array(aus))
    return np.array(images), np.array(labels), classes, minSize

def buildFilters(sigma, gamma):
    filters = []
    ksize = 17
    # Picked parameters from: Coding Facial Expressions with Gabor Wavelets by Lyons et al.
    for theta in np.arange(0, np.pi, np.pi / 17):
        for lambd in [np.pi / 2, np.pi / 4, np.pi / 8, np.pi / 16]:
            for psi in [0, np.pi / 2]:
                kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
                filters.append(kern)
    return filters

def gaborReduce(images):
    sigma = 4.0
    gamma = 0.5
    filters = buildFilters(sigma, gamma)
    newImages = []
    for img in images:
        reducedImage = [0] * 136
        for i in range(136):
            tImg = cv2.filter2D(img, cv2.CV_32FC1, filters[i])
            tImg = np.reshape(tImg, (1, -1))
            sum = np.sum(tImg)
            reducedImage[i] = np.sum(tImg)
        newImages.append(np.array(reducedImage))
    return np.array(newImages)


def eval(classes, clf, x, y):
    n = len(x)
    yScore = clf.decision_function(x)
    # Compute ROC curve and ROC area for each class    
    roc_auc = {}
    for i in range(len(classes)):
        if sum(y[:, i]) > 0:
            roc_auc[i] = roc_auc_score(y[:, i], yScore[:, i]) * 100.0
        else:
            roc_auc[i] = -1
    return roc_auc


if __name__ == "__main__":
    # The root directory where the CK+ database is located
    #images, labels, classes = loadFiles("test")
    images, labels, classes, minSize = loadFiles("C:\\CK\\train")
    labels = MultiLabelBinarizer().fit_transform(labels)
    # TBD: Change these two values based on the classifier's performance
    reducedImages = []    
    #sys.argv = ["", '-isomap']
    if sys.argv[1] == '-gabor':
        reducedImages = gaborReduce(images)
    elif sys.argv[1] == '-pca':
        trimmedImages = []
        for i in range(len(images)):
            images[i] = np.reshape(images[i], (-1))
            images[i] = images[i][:minSize]
            trimmedImages.append(images[i])            
        pca = PCA(n_components=136)
        reducedImages = pca.fit_transform(trimmedImages)
    elif sys.argv[1] == '-isomap':
        trimmedImages = []
        for i in range(len(images)):
            images[i] = np.reshape(images[i], (-1))
            images[i] = images[i][:minSize]
            trimmedImages.append(images[i])
        isomap = Isomap(n_components=136)
        reducedImages = isomap.fit_transform(trimmedImages)
    elif sys.argv[1] == '-lle':
        trimmedImages = []
        for i in range(len(images)):
            images[i] = np.reshape(images[i], (-1))
            images[i] = images[i][:minSize]
            trimmedImages.append(images[i])
        lle = LocallyLinearEmbedding(n_components=136)
        reducedImages = lle.fit_transform(trimmedImages)
    
    # Do cross-fold validation 
    kf = KFold(len(images), n_folds=2)
    minAreas = {}
    maxAreas = {}
    avgAreas = {}
    totals = {}
    for train_index, test_index in kf:        
        xTrain = reducedImages[train_index]
        yTrain = labels[train_index]
        clf = OneVsRestClassifier(LinearSVC(), 4)
        clf.fit(xTrain, yTrain)
        xTest = reducedImages[test_index]
        yTest = labels[test_index]
        areas = eval(classes, clf, xTest, yTest)
        for c in range(len(classes)):
            if areas[c] != -1:
                if c not in minAreas or areas[c] < minAreas[c]:
                    minAreas[c] = areas[c]
                if c not in maxAreas or areas[c] > maxAreas[c]:
                    maxAreas[c] = areas[c]
                if c not in avgAreas:
                    avgAreas[c] = areas[c]
                else:
                    avgAreas[c] += areas[c]
                if c not in totals:
                    totals[c] = 1
                else:
                    totals[c] += 1
    print "AU\tN\tROC area"
    for c in range(len(classes)):
        avgAreas[c] /= totals[c]
        print classes[c], "\t", avgAreas[c], "\t", maxAreas[c] - minAreas[c]
