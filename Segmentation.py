import cv2
import os
import numpy as np

import torch
from torchvision import io
from torchvision import transforms as T
from PIL import Image
from os.path import isfile, join
import glob
from PIL import Image as im
import csv
from PIL import ImageDraw, ImageFilter


# FPS is 25
# Video Link:- https://www.youtube.com/watch?v=TPX5yDvLUjg
thersholdRatioWidth = 3
thersholdRatioHeight = 2
# thersholdRatioWidth = 2
# thersholdRatioHeight = 2
width = 1280
height = 720

def getFPS():
    video = cv2.VideoCapture("C:\\Users\\utkarshmohan\\Downloads\\CookingVideo.mp4")
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)<3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    video.release()


def convertVideToImage():
    vidcap = cv2.VideoCapture("C:\\Users\\utkarshmohan\\Downloads\\CookingVideo.mp4")
    success,image = vidcap.read()
    count = 0
    print(os.getcwd())
        
    while success:
        cv2.imwrite("cookingImgs\\frame%d.jpg" % count, image)  
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


def convertFramesToVideo(pathIn, pathOut, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter('video.avi', fourcc, 1, (30, 70))

    for j in range(1,50):
        img = cv2.imread(pathIn+str(j) + '.jpg')
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


def filterUsefulImages():
    for i in range(770, 1069):
        img = cv2.imread("cookingImgs\\frame%d.jpg" %i)
        cv2.imwrite("filteredImages\\frame%d.jpg" % i, img)

    for i in range(1177, 1270):
        img = cv2.imread("cookingImgs\\frame%d.jpg" %i)
        cv2.imwrite("filteredImages\\frame%d.jpg" % i, img)

    for i in range(1872, 1944):
        img = cv2.imread("cookingImgs\\frame%d.jpg" %i)
        cv2.imwrite("filteredImages\\frame%d.jpg" % i, img)

    for i in range(4932, 5099):
        img = cv2.imread("cookingImgs\\frame%d.jpg" %i)
        cv2.imwrite("filteredImages\\frame%d.jpg" % i, img)

    for i in range(6228, 6737):
        img = cv2.imread("cookingImgs\\frame%d.jpg" %i)
        cv2.imwrite("filteredImages\\frame%d.jpg" % i, img)


def show_image(image):
    print(type(image))
    if image.shape[2] != 3: image = image.permute(1, 2, 0)
    image = im.fromarray(image.numpy())
    image.show()
    return image


def checkSize(i, j, image):
    for a in range(i, i+(height//thersholdRatioHeight)):
        for b in range(j, j+(width//thersholdRatioWidth)):
            if i>height or j>width:
                return False

            if image[0][i][j]==120 and image[1][i][j]==120 and image[2][i][j]==120:
                continue
            else:
                return False

    return True


# def findFirstWindow(image):


def writeInFile(imageName, I, J, Size, filename):
    
    preData = []
    with open(filename, "r") as readFile:
        reader = csv.reader(readFile)
        for item in reader:
            if item:
                preData.append(item)

    with open(filename, "w") as writeFile :
        writer = csv.writer(writeFile)
        for item in preData:
            writer.writerow(item)

        writer.writerow([imageName, I, J, Size])


def findDimensionOfImage(path):
    image = io.read_image(path)
    res = []
    res.append(path)
    # print(image.shape)
    R = height
    C = width
    memo = []
    for i in range(R):
        temp = []
        for j in range(C):
            temp += 0,
        memo += temp,

    prevMax = 0
    indexI = 0
    indexJ = 0

    for i in range(1, height):
        for j in range(1, width):
            if image[0][i][j]==120 and image[1][i][j]==120 and image[2][i][j]==120:
                memo[i][j] = 1+min(memo[i-1][j], memo[i][j-1], memo[i-1][j-1])
                if memo[i][j]>prevMax:
                    prevMax = memo[i][j]
                    indexI = i
                    indexJ = j
            else:
                memo[i][j] = 0

    writeInFile(path, indexI, indexJ, memo[indexI][indexJ], "D://Work//script//dimensionData.csv")
    print("Done Writing In File")

    # height 720
    # width 1280

    
    # for i in range(indexI, indexI-memo[indexI][indexJ], -1):
    #   for j in range(indexJ, indexJ-memo[indexI][indexJ], -1):
    #       image[0][i][j] = 2
    #       image[1][i][j] = 2
    #       image[2][i][j] = 2

    # show_image(image)





def getImagesDir():
    rootDir = "D://Work//script//infered//"
    imagesDir = os.listdir(rootDir)
    dirs = []
    for img in imagesDir:
        img = rootDir+img
        dirs.append(img)
    return dirs


def convertToVideo():
    pathIn= 'filteredImages/'
    pathOut = 'videoR.avi'
    fps = 24
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    files.sort(key = lambda x: int(x[5:-4]))
    for i in range(len(files)):
        filename=pathIn + files[i]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()
    return

def putImageinVideo(frameNo, x, y,  count):
    img1 = Image.open("cookingImgs\\frame%d.jpg" %frameNo)
    img2 = Image.open("logo\\frame%d.jpg" %frameNo)
    for i in range(count):
        img1 = Image.open("cookingImgs\\frame%d.jpg" %frameNo)
        back_im = img1.copy()
        back_im.paste(img2, (x,y))
        back_im.save("pastedImg\\frame%d.jpg")
    return



def findBestPlace():

    # Open File and Sort Item First EveryTime
    dirs = []
    # For Testin Purpose
    for i in range (1,20):
        dirs.append([371, 1279, 40])

    # 4 sec
    time = fps*4

    dimSize = len(dirs)

    for i in range(dimSize):
        xmax = dirs[i][1]
        ymax = dirs[i][2]
        z = dirs[i][3]

        c = 1

        for j in range(i+1, dimSize):
            if abs(dirs[j][1]-xmax)<=3 and abs(dirs[j][2]-ymax)<=3 and abs(z-dirs[j][3])<=3:
                c+=1
            else:
                break

        preData = []
        filename = "D://Work//script//dimensionDataModified.csv"
        with open(filename, "r") as readFile:
            reader = csv.reader(readFile)
            for item in reader:
                if item:
                    preData.append(item)

        with open(filename, "w") as writeFile :
            writer = csv.writer(writeFile)
            for item in preData:
                writer.writerow(item)
        writer.writerow([dirs[i][0], dirs[i][1], dirs[i][2], dirs[i][3], c])
    return


convertToVideo()
# dirs = getImagesDir()
# for item in dirs:
#   findDimensionOfImage(item)


# def findTimeToDisplay():


# def findColourOfBackground():



# fps = getFPS()
# filterUsefulImages()
# for i in range(20):
#   writeInFile("aa", i, i*10, 20)
# findDimensionOfImage("D:\\Work\\script\\infered\\frame4948.jpg")


# filterUsefulImages()
# convertFramesToVideo("D:\\Work\\script\\cookingImgs\\*jpg","project.avi", fps)



    # count =0  
    # for i in range(20,height-20):
    #   for j in range(20, width-20):
    #       if image[0][i][j]==120 and image[1][i][j]==120 and image[2][i][j]==120:
    #           arrV[i][j] = 1+arrV[i][j-1]
    #           arrH[i][j] = 1+arrH[i-1][j]
    #           arrRes[i][j] = arrRes[i-1][j-1]
    #       else:
    #           arrV[i][j] = 0
    #           arrH[i][j] = 0
    #       if arrV[i][j]>(width//thersholdRatioWidth) and arrH[i][j]>(height//thersholdRatioHeight):
    #           count+=1
    #           lastI = i
    #           lastJ = j
    #           x = j
    #           y = i

    #       # if(i<100):
    #       #   print(arrH[i][j], end = " ")
    #   # print("******")

    # print(count)
    #   # print("***********")

# 678
# 235
# 144
# 256



# Top Down view 
# 767 to 1068, 1170 to 1270, 1872 to 1944,
# Front View:
# 4932 to 5099, 6228 to 6737  