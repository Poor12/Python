import cv2
import numpy as np
from matplotlib import pyplot as plt
image=cv2.imread("plane.jpg",cv2.IMREAD_GRAYSCALE)
image_bgr=cv2.imread("plane.jpg",cv2.IMREAD_COLOR)
image_rbg=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
# plt.imshow(image,cmap="gray")
# plt.axis("off")
# plt.show()
print(type(image))
print(image_bgr[0,0])
#image=cv2.imwrite("plane_new.jpg",image)
image_5050=cv2.resize(image,(50,50))
# plt.imshow(image_5050,cmap="gray")
# plt.axis("off")
# plt.show()
#截断
image_cropped=image[:,:128]
#模糊,将像素值转为周围值的平均
image_blurry=cv2.blur(image,(100,100))
# plt.imshow(image_blurry,cmap="gray")
# plt.axis("off")
# plt.show()
#锐化
kernel=np.array([[0,-1,0],
                 [-1,5,-1],
                  [0,-1,0]])
image_sharp=cv2.filter2D(image,-1,kernel)
#加强
image_enhanced=cv2.equalizeHist(image)
# plt.imshow(image_enhanced,cmap="gray")
# plt.axis("off")
# plt.show()
#去颜色
image_hsv=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2HSV)
lower_blue=np.array([50,100,50])
upper_blue=np.array([130,255,255])
mask=cv2.inRange(image_hsv,lower_blue,upper_blue)
image_bgr_masked=cv2.bitwise_and(image_bgr,image_bgr,mask=mask)
image_rgb=cv2.cvtColor(image_bgr_masked,cv2.COLOR_BGR2RGB)
# plt.imshow(image_rgb,cmap="gray")
# plt.axis("off")
# plt.show()
#二值化图片
max_output_value=255
neighborhood_size=99
subtract_from_mean=10
image_binarized=cv2.adaptiveThreshold(image,
                                      max_output_value,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY,
                                      neighborhood_size,
                                      subtract_from_mean)
# plt.imshow(image_binarized,cmap="gray")
# plt.axis("off")
# plt.show()

#移除背景
image_bgr=cv2.imread('plane2.jpg')
image_rgb=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
rectangle=(0,56,256,150)
mask=np.zeros(image_bgr.shape[:2],np.uint8)
bgdModel=np.zeros((1,65),np.float64)
print(image_rgb.shape)
fgdModel=np.zeros((1,65),np.float64)
#run grabcut
cv2.grabCut(image_rgb,
            mask,
            rectangle,
            bgdModel,
            fgdModel,
            5,
            cv2.GC_INIT_WITH_RECT)
mask_2=np.where((mask==2)|(mask==0),0,1).astype('uint8')
image_rgb_nobg=image_rgb*mask_2[:,:,np.newaxis]
# plt.imshow(image_rgb_nobg)
# plt.axis("off")
# plt.show()

#canny edge dector边缘检测
median_intensity=np.median(image)
lower_threshold=int(max(0,(1.0-0.33)*median_intensity))
upper_threshold=int(min(255,(1.0+0.33)*median_intensity))
image_canny=cv2.Canny(image,lower_threshold,upper_threshold)
# plt.imshow(image_canny)
# plt.axis("off")
# plt.show()

#角落检测--cornerHarris，goodFeaturesToTrack

#创建特征矩阵
image=cv2.imread("plane2.jpg",cv2.IMREAD_GRAYSCALE)
print(image.shape)
image_1010=cv2.resize(image,(10,10))
print(image_1010.shape)
print(image_1010.flatten())

#颜色编码
image_bgr=cv2.imread("plane2.jpg",cv2.IMREAD_COLOR)
image_rgb=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
print(image_bgr.shape)
channels=cv2.mean(image_bgr)
print(channels)
observation=np.array([(channels[2],channels[1],channels[0])])
plt.imshow(observation)
plt.axis("off")
plt.show()

features=[]
colors=("r","g","b")
for i,channel in enumerate(colors):
    histogram=cv2.calcHist([image_rgb],
                           [i],
                           None,
                           [256],
                           [0,256])
    features.extend(histogram)
    plt.plot(histogram,color=channel)
    plt.xlim([0,256])
plt.show()


