import numpy as np
import cv2
import imutils
import os
import joblib
from sklearn.svm import LinearSVC
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imutils import paths

image_paths=[]
image_classes=[0]
class_id=0
svc = LinearSVC()
svc = joblib.load("svml.pkl")


player = input("please enter paper or scissor or stone\n")
img=cv2.imread("./player/" + (player+".jpg"))
img=cv2.resize(img,(300,300))

sift=cv2.xfeatures2d.SIFT_create()
des_list=[]

kpts=sift.detect(img)
kpts, des=sift.compute(img,kpts)
des_list.append(des)

#生成特徵描述子向量
descriptors=des_list[0][1]

for image_path,descriptor in des_list[1:]:
    descriptors=np.vstack((descriptors, descriptor))

k=30
voc, variance=kmeans(descriptors,k,1)

#生成特徵質方圖
im_features=np.zeros((1,k),"float32")
words, distance=vq(des_list[0][1],voc)
for w in words:
    im_features[0][w] += 1

#開始訓練 x=data ,y=target
x = im_features
y=np.array(image_classes)
print(y)
print(svc.predict(x))



cv2.imshow('ori',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
