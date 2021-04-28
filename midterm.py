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

#設定dataset路徑
train_path="./train/"
training_names=os.listdir(train_path)
#training_name=['people', 'pig']
image_paths=[]
image_classes=[]
class_id=0

#尋訪所有圖片並設置target
for training_name in training_names:
    dir = os.path.join(train_path,training_name)
    class_path=list(paths.list_images(dir))
    image_paths+=class_path
    image_classes += [class_id]*len(class_path)
    class_id+=1
#image_class=target=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


#創建sift特徵提取器
sift=cv2.xfeatures2d.SIFT_create()

des_list=[]

for image_path in image_paths:
    #讀取圖片
    im=cv2.imread(image_path)
    #調整圖片的大小
    im=cv2.resize(im,(300,300))
    #取出圖片的特徵資料當data
    kpts=sift.detect(im)
    kpts,des=sift.compute(im,kpts)
    des_list.append((image_path,des))
    print("image file path:",image_path)

#生成特徵描述子向量
descriptors=des_list[0][1]

for image_path,descriptor in des_list[1:]:
    descriptors=np.vstack((descriptors, descriptor))

#k-means分析，一般會取便是種類的10倍，所以k=2*10=20
k=30
voc,variance=kmeans(descriptors,k,1)

#生成特徵質方圖
im_features=np.zeros((len(image_paths),k),"float32")
for i in range(len(image_paths)):
    words, distance=vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

#開始訓練 x=data ,y=target
x = im_features
y=np.array(image_classes)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=0)

clf=LinearSVC()
clf.fit(x_train,y_train)
joblib.dump(clf,"svml.pkl")

print("predict:")
print(clf.predict(x_train))
print(clf.predict(x_test))

print("Accuracy:")
print(clf.score(x_train,y_train))
print(clf.score(x_test,y_test))

stdSlr=StandardScaler().fit(im_features)
im_features=stdSlr.transform(im_features)
