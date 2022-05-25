import cv2
import numpy as np
import os
import multiprocessing
from multiprocessing import Pool
import tqdm
import errno

root = os.getcwd()
folders = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
parent = os.path.join(root, os.path.pardir)

def make_dir(path):
    try:
        os.makedirs(path, exist_ok=True)  # Python>3.2
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise

def segment(imgpath):
    imgname = os.path.split(imgpath)[1]
    dir = os.path.split(os.path.split(imgpath)[0])[1]
    parent = os.path.join(os.path.split(imgpath)[0], os.path.pardir, os.path.pardir)
    img = cv2.imread(imgpath)
    #print(os.path.join(parent,'Segmented2', dir, imgname + '.png'))
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (2,2,253,253)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    rgb_planes = cv2.split(img)
    dilated_img = cv2.dilate(rgb_planes[0], np.ones((3,3), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 11)
    diff_img = 255 - cv2.absdiff(rgb_planes[0], bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result = cv2.merge((norm_img, rgb_planes[1],rgb_planes[2]))
    img = cv2.cvtColor(img,cv2.COLOR_LAB2BGR)
    mask = np.zeros(img.shape[:2],np.uint8)
    (mask, bgdModel, fgdModel) = cv2.grabCut(result, mask, rect, bgdModel,fgdModel, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    cv2.imwrite(os.path.join(parent,'Segmented2',dir, imgname + '.png'), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
if __name__ == '__main__':
    #create segmented directory structure
    for folder in folders:
        make_dir(os.path.join(parent, 'Segmented2', folder))
    images = []
    #get all images in a directory
    for _, dirs, files in os.walk(root):
        for dir in dirs:
            for imgpath in os.listdir(dir):
                images.append(os.path.join(root,dir,imgpath))
    cnt = len(images)
    
    with Pool(processes=multiprocessing.cpu_count()) as p:
        with tqdm.tqdm(total=cnt) as pbar:
            for _ in p.imap_unordered(segment, images):
                pbar.update()