import cv2

path = "C:/Users/Marcin/source/repos/InzynieriaOprogramowania1/Trichoderma/Trichoderma_1.png"
def isgray(imgpath):
    img = cv2.imread(imgpath)
    if len(img.shape) < 3: return True
    if img.shape[2]  == 1: return True
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    if (b==g).all() and (b==r).all(): return True
    return False
print(isgray(path))