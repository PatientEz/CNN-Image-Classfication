
import  numpy as np
import  os
import cv2 as cv


rootpath = './testing/'
list = os.listdir(rootpath) #列出文件夹下所有的目录与文件
#设定图像宽高
imgwidth = 220
imgheight = 220

imgdata = []
imgtag = []

for i in range(len(list)):
    #对于子目录进行处理
    # print(i)
    currentpath = rootpath+list[i]
    currentlist = os.listdir(currentpath)
    print(list[i])
    for j in range(len(currentlist)):
        #图像位置
        imgpath = currentpath + "/" + currentlist[j]
        #有后缀为db的文件
        if currentlist[j][-3:] == "jpg":
            #i为类目
            imgtag.append(i)
            #加载图像
            img = cv.imread(imgpath,0)
            img = cv.resize(img,(imgwidth,imgheight))
            print(img.shape)
            imgdata.append(img)



imgtag = np.array(imgtag)
imgdata = np.array(imgdata)
imgtag = imgtag.reshape(imgtag.shape[0],1)
#增加一维灰度维
imgdata = imgdata.reshape(imgdata.shape[0],imgdata.shape[1],imgdata.shape[2],1)
print(imgdata.shape,imgtag.shape)
#存储
np.save("./x_test.npy",imgdata)
np.save("./y_test.npy",imgtag)