from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import os
from multiprocessing import Process

import os
import sys
import threading


def walk(top, threads=60):
  """Multi-threaded version of os.walk().
  This routine provides multiple orders of a magnitude performance improvement
  when top is mapped to a network filesystem where i/o operations are slow, but
  unlimited. For spinning disks it should still run faster regardless of thread
  count because it uses a LIFO scheduler that guarantees locality. For SSDs it
  will go tolerably slower.
  The more exotic coroutine features of os.walk() can not be supported, such as
  the ability to selectively inhibit recursion by mutating subdirs.
  Args:
    top: Path of parent directory to search recursively.
    threads: Size of fixed thread pool.
  Yields:
    A (path, subdirs, files) tuple for each directory within top, including
    itself. These tuples come in no particular order; however, the contents of
    each tuple itself is sorted.
  """
  if not os.path.isdir(top):
    return
  lock = threading.Lock()
  on_input = threading.Condition(lock)
  on_output = threading.Condition(lock)
  state = {'tasks': 1}
  paths = [top]
  output = []

  def worker():
    while True:
      with lock:
        while True:
          if not state['tasks']:
            output.append(None)
            on_output.notify()
            return
          if not paths:
            on_input.wait()
            continue
          path = paths.pop()
          break
      try:
        dirs = []
        files = []
        for item in sorted(os.listdir(path)):
          subpath = os.path.join(path, item)
          if os.path.isdir(subpath):
            dirs.append(item)
            with lock:
              state['tasks'] += 1
              paths.append(subpath)
              on_input.notify()
          else:
            files.append(item)
        with lock:
          output.append((path, dirs, files))
          on_output.notify()
      except OSError as e:
        print(e, file=sys.stderr)
      finally:
        with lock:
          state['tasks'] -= 1
          if not state['tasks']:
            on_input.notifyAll()

  workers = [threading.Thread(target=worker,
                              name="fastio.walk %d %s" % (i, top))
             for i in range(threads)]
  for w in workers:
    w.start()
  while threads or output:  # TODO(jart): Why is 'or output' necessary?
    with lock:
      while not output:
        on_output.wait()
      item = output.pop()
    if item:
      yield item
    else:
      threads -= 1

count = 0

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

def segment(img):
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
    
    # mask = cv2.imread(os.path.join(parent,'masks',dir, imgpath), cv2.IMREAD_GRAYSCALE)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # mask = cv2.GaussianBlur(mask, (5,5), 0)
    # mask[(mask > 32) & (mask < 255)] = cv2.GC_PR_FGD
    # mask[(mask <= 32) & (mask > 0)] = cv2.GC_PR_BGD
    # mask[mask == 255] = cv2.GC_FGD
    # mask[mask == 0] = cv2.GC_BGD
    # try:
    #     (mask,bgdModel,fgdModel) = cv2.grabCut(img,mask,rect,bgdModel,fgdModel, 10, cv2.GC_INIT_WITH_RECT + cv2.GC_INIT_WITH_MASK)
    # except:
    #     mask = np.zeros(img.shape[:2],np.uint8)
    #     (mask, bgdModel, fgdModel) = cv2.grabCut(img, mask, rect, bgdModel,fgdModel, iterCount=10, mode=cv2.GC_INIT_WITH_RECT)
    mask = np.zeros(img.shape[:2],np.uint8)
    (mask, bgdModel, fgdModel) = cv2.grabCut(result, mask, rect, bgdModel,fgdModel, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    return img

def main():
    global count
    root = os.getcwd()
    #print(root)
    folders = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name)) ]
    parent = os.path.join(root, os.path.pardir)

    #create segmented directory structure
    for folder in folders:
        make_dir(os.path.join(parent, 'Segmented', folder))
    
    #get all images in a directory
    for root, dirs, files in walk(root, threads=8):
        for dir in dirs:
            for imgpath in os.listdir(dir):
                #print(imgpath)
                count += 1
                print(count)
                if os.path.exists(os.path.join(parent,'Segmented',dir, os.path.splitext(imgpath)[0] + '.png')):
                    #print('Skipping ' + os.path.join(parent,'Segmented',dir, os.path.splitext(imgpath)[0] + '.png'))
                    continue
                img = cv2.imread(os.path.join(root, dir, imgpath))
                img = segment(img)
                cv2.imwrite(os.path.join(parent,'Segmented',dir, os.path.splitext(imgpath)[0] + '.png'), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                
if __name__ == '__main__':
    main()
