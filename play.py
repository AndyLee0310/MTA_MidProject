import random
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

def game(player):
    image_paths=[]
    image_classes=[]
    class_id=0
    svc = LinearSVC()
    svc = joblib.load("svml.pkl")

    img=cv2.imread("./player/" + (player+".jpg"))
    img=cv2.resize(img,(300,300))

    #取亂數
    opponent=random.randint(0,2)
    #存取對手圖片
    opponent_img=cv2.imread("./opponent/" + (str(opponent) + ".png"))
    opponent_img=cv2.resize(opponent_img,(300,300))

    sift=cv2.xfeatures2d.SIFT_create()
    des_list=[]

    kpts=sift.detect(img)
    kpts, des=sift.compute(img,kpts)
    des_list.append(("./player/" + (player+".jpg"),des))

    #生成特徵描述子向量
    descriptors=des_list[0][1]

    for image_path,descriptor in des_list[1:]:
        descriptors=np.vstack((descriptors, descriptor))

    k=30
    #使用先前產生的kmeans模組
    voc=joblib.load("kmeansl.pkl")

    #生成特徵質方圖
    im_features=np.zeros((1,k),"float32")


    words, distance=vq(des_list[0][1],voc)

    for w in words:
        im_features[0][w] += 1
    #print(im_features)

    #開始訓練 x=data ,y=target
    x = im_features
    y=np.array(image_classes)

    player_ans=svc.predict(x)

    #printt出判斷結果
    print()
    print(player_ans)

    WIN = "./WinOrLose/WIN.JPG"
    TIE = "./WinOrLose/TIE.JPG"
    LOSE = "./WinOrLose/LOSE.JPG"

    #判斷player的輸贏
    if(player_ans==0 and opponent==0):
        ans=cv2.imread(TIE)
        print('TIE')
    if(player_ans==0 and opponent==1):
        ans=cv2.imread(LOSE)
        print('You LOSE')
    if(player_ans==0 and opponent==2):
        ans=cv2.imread(WIN)
        print('You WIN')
    if(player_ans==1 and opponent==0):
        ans=cv2.imread(WIN)
        print('You WIN')
    if(player_ans==1 and opponent==1):
        ans=cv2.imread(TIE)
        print('TIE')
    if(player_ans==1 and opponent==2):
        ans=cv2.imread(LOSE)
        print('You LOSE')
    if(player_ans==2 and opponent==0):
        ans=cv2.imread(LOSE)
        print('You LOSE')
    if(player_ans==2 and opponent==1):
        ans=cv2.imread(WIN)
        print('You WIN')
    if(player_ans==2 and opponent==2):
        ans=cv2.imread(TIE)
        print('TIE')

    #將答案圖片resize
    ans=cv2.resize(ans,(300,300))
    #為了區分player與opponent，在各自圖片加上文字
    player_text='player'
    opponent_text='opponent'
    cv2.putText(img,player_text,(5,20),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,255),1,cv2.LINE_AA)
    cv2.putText(opponent_img,opponent_text,(135,20),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,255),1,cv2.LINE_AA)

    #將3張圖片結合在一起
    result=np.concatenate((img,ans,opponent_img),axis=1)

    cv2.imshow('result',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


while (True):
    try:
        print("\npaper=0,scissor=1,stone=2")
        player = input("please enter 'paper' or 'scissor' or 'stone' to play, or enter 'stop' to stop game.\n")

        if (player == 'stop'):
            print("\nYou stop the game.")
            break
        else:
            game(player)
    except:
        print("\nplease enter 'paper' or 'scissor' or 'stone' or enter 'stop'.\n")
