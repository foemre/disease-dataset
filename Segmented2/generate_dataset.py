import cv2
import numpy as np
import os
import multiprocessing
from multiprocessing import Pool
import tqdm
import random
import errno
import shutil
import argparse

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

root = os.getcwd()
parent = os.path.join(root, os.path.pardir)
dataset_folders = ['Bacterial_spot', 'Early_blight', 'Healthy', 'Late_blight', 'Leaf_Mold', 'Mosaic_virus', 'Septoria_leaf_spot', 'Target_Spot', 'Two-spotted_spider_mite', 'Yellow_Leaf_Curl_Virus']
background_folder = 'complex_background'
filter_folder = 'filters'
backgrounds = [os.path.join(parent, background_folder, item) for item in os.listdir(os.path.join(parent, background_folder))]
filters = [os.path.join(parent, filter_folder, item) for item in os.listdir(os.path.join(parent, filter_folder))]
images = []
dataset_images = []
for _, dirs, files in os.walk(root):
    for dir in dirs:
        for imgpath in os.listdir(dir):
            images.append(os.path.join(root,dir,imgpath))
for dir in dataset_folders:
    for imgpath in os.listdir(dir):
        dataset_images.append(os.path.join(root,dir,imgpath))
dst_val_txt = os.path.join(parent, "disease", "labels", "val")
dst_test_txt = os.path.join(parent, "disease", "labels", "test")
dst_train_txt = os.path.join(parent, "disease", "labels", "train")
dst_val_img = os.path.join(parent, "disease", "images", "val")
dst_test_img = os.path.join(parent, "disease", "images", "test")
dst_train_img = os.path.join(parent, "disease", "images", "train")
make_dir(dst_val_txt)
make_dir(dst_test_txt)
make_dir(dst_train_txt)
make_dir(dst_val_img)
make_dir(dst_test_img)
make_dir(dst_train_img)

imggrps = []
for img in images:
    num = random.randint(4,7)
    grp =[]
    for i in range(num):
        selected = random.choice(images)
        grp.append(random.choice(images))
    imggrps.append(grp)

valcount = 0
testcount = 0
count = 0
# What I want is :
# Generate images with all images, but only label those that are in "folders"
# Create list of lists of 4-7 elements
def create_dataset(imgs, vallimit, testlimit):
    global valcount, testcount, cnt, count
    bg = random.choice(backgrounds)
    bg = cv2.imread(bg)
    if bg.shape[0] < bg.shape[1]:
        bg = cv2.resize(bg, (1600,1200))
    else:
        bg = cv2.resize(bg, (1200,1600))
    b_rows = np.linspace(0 + random.randint(0,127), bg.shape[0]-332-random.randint(1,32), num=len(imgs), dtype=np.uint32).tolist()
    b_cols = np.linspace(0 + random.randint(0,127), bg.shape[1]-332-random.randint(1,32), num=len(imgs), dtype=np.uint32).tolist()
    for img in imgs:
        image = cv2.imread(img)
        scale = random.random()/2 + 0.8
        if random.random() < 0.5:
            (h, w) = image.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            # rotate our image by 45 degrees around the center of the image
            M = cv2.getRotationMatrix2D((cX, cY), int(random.random()*360), 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)), interpolation=cv2.INTER_AREA)
        imgclass = os.path.split(os.path.split(img)[0])[1]
        imgname = os.path.split(img)[1]
        imggray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(imggray, 5, 255, cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key = cv2.contourArea)
        bndx,bndy,bndw,bndh = cv2.boundingRect(cnt)
        image = image[bndy:bndy+bndh,bndx:bndx+bndw]
        mask = mask[bndy:bndy+bndh,bndx:bndx+bndw]
        mask_inv = cv2.bitwise_not(mask)
        rows,cols,_ = image.shape
        b_row = random.choice(b_rows)
        b_rows.remove(b_row)
        b_col = random.choice(b_cols)
        b_cols.remove(b_col)
        if (b_row + rows) > bg.shape[0] or (b_col + cols) > bg.shape[1]:
            print("bg shape error, auto shifting")
            diff = int(max((b_row + rows) - bg.shape[0] + 3,(b_col + cols) - bg.shape[1] + 3))
            print(diff)
            b_row = b_row - diff
            b_col = b_col - diff
        roi = bg[b_row:b_row + rows, b_col:b_col + cols]        
        bg_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        img_fg = cv2.bitwise_and(image,image, mask = mask)
        dst = cv2.add(bg_bg, img_fg, dtype=cv2.CV_8UC3)
        
        if random.random() < 0.5:
            dst = cv2.GaussianBlur(dst,(3,3),0)
        bg[b_row:b_row + rows, b_col:b_col + cols] = dst
        if imgclass in dataset_folders:
            w = image.shape[1]
            h = image.shape[0]
            center_x = b_col + w/2
            center_y = b_row + h/2

            norm_x = center_x / bg.shape[1]
            norm_y = center_y / bg.shape[0]
            norm_w = w / bg.shape[1]
            norm_h = h / bg.shape[0]

            # for debugging
            # start_point = (int(norm_x*bg.shape[1]-w/2), int(norm_y*bg.shape[0]-h/2))
            # end_point = (int(norm_x*bg.shape[1]-w/2+norm_w*bg.shape[1]), int(norm_y*bg.shape[0]-h/2+norm_h*bg.shape[0]))
            # color = (255, 0, 0)
            # bg = cv2.rectangle(bg, start_point, end_point, color, 2)
            # bg = cv2.putText(bg,str(imgclass),start_point, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 0, 0),thickness=2, fontScale=2)
            
            imgclass = dataset_folders.index(imgclass)

            if valcount < vallimit:
                with open(os.path.join(dst_val_txt, str(count) + '.txt'), 'a') as f:
                        line_to_write = " ".join([str(imgclass), str(norm_x), str(norm_y), str(norm_w), str(norm_h)])
                        f.write(line_to_write + "\n")
            elif valcount >= vallimit and testcount < testlimit:
                with open(os.path.join(dst_test_txt, str(count) + '.txt'), 'a') as f:
                    line_to_write = " ".join([str(imgclass), str(norm_x), str(norm_y), str(norm_w), str(norm_h)])
                    f.write(line_to_write + "\n")
            else:
                with open(os.path.join(dst_train_txt, str(count) + '.txt'), 'a') as f:
                    line_to_write = " ".join([str(imgclass), str(norm_x), str(norm_y), str(norm_w), str(norm_h)])
                    f.write(line_to_write + "\n")
    if random.random() < 0.5:
        filter = random.choice(filters)
        filter = cv2.imread(filter)
        # filter = cv2.cvtColor(filter, cv2.COLOR_BGR2HSV)
        # themax = int(filter[:,:,2].max())
        # fac = 255 / themax
        # filter[:,:,2] = filter[:,:,2] * fac
        # filter = cv2.cvtColor(filter, cv2.COLOR_HSV2BGR)

        filter = cv2.resize(filter, (bg.shape[1], bg.shape[0]))
        bg = cv2.multiply(bg/255, filter/255)*255

    if valcount < vallimit:
        path_to_write = os.path.join(dst_val_img, str(count) + '.jpg')
        cv2.imwrite(path_to_write, bg, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        valcount += 1
    elif valcount >= vallimit and testcount < testlimit:
        path_to_write = os.path.join(dst_test_img, str(count) + '.jpg')
        cv2.imwrite(path_to_write, bg, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        testcount += 1
    else:
        path_to_write = os.path.join(dst_train_img, str(count) + '.jpg')
        cv2.imwrite(path_to_write, bg, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    count += 1

if __name__ == "__main__":
    description = f"Generate dataset, num: number of images"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--num", type=int, default=2000, help="Number of images to generate")
    args = parser.parse_args()
    vallimit = int(args.num * 0.15)
    testlimit = int(args.num * 0.15)
    #multithreaded
    # with Pool(processes=multiprocessing.cpu_count()-1) as p:
    #     with tqdm.tqdm(total=len(imggrps)) as pbar:
    #         for _ in p.imap_unordered(create_dataset, imggrps):
    #             pbar.update()
    #single thread            
    with tqdm.tqdm(total=args.num) as pbar:
        for _ in range(args.num):
            grp = random.sample(images, random.randint(4,7))
            create_dataset(grp, vallimit, testlimit)
            pbar.update()
    
    
