import os
import math
import cv2
import shutil
import numpy as np
from sklearn.utils import shuffle



from PIL import ImageEnhance
path1 = r"H:\dataset\pinecone4/"
path2 = r"H:\dataset\pinecone6/"
path3 = r"H:\dataset\anno3/"
path4 = r"H:\dataset\anno5/"
for i, file in enumerate(shuffle(os.listdir(path1))):
    if i == int(0.2*len(os.listdir(path1))):
        print(i)
        break
    # img = cv2.imread(path1 + file)
    else:
        shutil.move(path1 + file, path2 + file)
        shutil.move(path3 + file[:-3] + "json", path4 + file[:-3] + "json")
    # img2 = cv2.resize(img, (640, 480))
    # cv2.imwrite(path2 + file, img)


# def fog(img):
#     img_f = img / 255.0
#     (row, col, chs) = img.shape
#     # r_x = random.randrange(0, row)
#     # r_y = random.randrange(0, col)
#     A = 0.6  # 亮度
#     beta = 0.06 # 雾的浓度
#     size = math.sqrt(max(row, col))  # 雾化尺寸
#
#     # center = (r_x, r_y)  # 雾化中心
#     center = (row // 2, col // 2)
#     for j in range(row):
#         for l in range(col):
#             d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
#             td = math.exp(-beta * d)
#             img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
#
#     cv2.normalize(img_f, img_f, 0, 255, cv2.NORM_MINMAX)
#
#     img_f = np.array(img_f, dtype=np.uint8)
#     return img_f
#
# def _brightness(image, min=0.5, max=2.0):
#     '''
#     Randomly change the brightness of the input image.
#     Protected against overflow.
#     '''
#     hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#
#     random_br = np.random.uniform(min, max)
#     # random_br=1.9
#     # To protect against overflow: Calculate a mask for all pixels
#     # where adjustment of the brightness would exceed the maximum
#     # brightness value and set the value to the maximum at those pixels.
#     mask = hsv[:, :, 2] * random_br > 255
#     v_channel = np.where(mask, 255, hsv[:, :, 2] * random_br)
#     hsv[:, :, 2] = v_channel
#
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
#
#
# def motion_blur(image, degree=12, angle=45):
#
#     image = np.array(image)
#
#     # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
#
#     M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
#
#     motion_blur_kernel = np.diag(np.ones(degree))
#
#     motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
#
#     motion_blur_kernel = motion_blur_kernel / degree
#
#     blurred = cv2.filter2D(image, -1, motion_blur_kernel)
#
#     # convert to uint8
#
#     cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
#
#     blurred = np.array(blurred, dtype=np.uint8)
#
#     return blurred
#
#
# path1 = r"H:\dataset\anno5/"
# path2 = r"H:\dataset\pinecone6/"
#
# for file in os.listdir(path2):
#     img = cv2.imread(path2 + file)
#     img_t = fog(img)
#     cv2.imwrite(r"H:\dataset\pinecone6/"+"fog_" + file[:-4] + ".jpg", img_t)
#     shutil.copy(path1+file[:-4]+".json", path1+"fog_"+file[:-4]+".json")



