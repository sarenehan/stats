# flake8: noqa
from __future__ import division
import os
import sys
import time
import pickle
import itertools
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline

mpl.rcParams['figure.figsize'] = (8.0, 10.0)

import cv2
from IPython.display import set_matplotlib_formats
from PIL import Image, ImageDraw

set_matplotlib_formats('jpg')

dataDir = '/Users/stewart/projects/stats/548/data'
# dataType='val2014' # uncomment to access the validation set
dataType = 'train2014'  # uncomment to access the train set
# dataType='test2014' # uncomment to access the train set
# annotations
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)


print('Expected directory structure:')
print('-' * 60)
for path, dirs, files in os.walk(dataDir):
    if path.split("/")[-1] != '.ipynb_checkpoints':
        # do not disply jupyter related files
        print(path)
    if path.split("/")[-1] in ['features_small', 'features_tiny']:
        for f in files:
            print('-' * 8, f)


coco = COCO(annFile)

cats = coco.loadCats(coco.getCatIds()) # categories
cat_id_to_name = {cat['id']: cat['name'] for cat in cats} # category id to name mapping
cat_name_to_id = {cat['name']: cat['id'] for cat in cats} # category name to id mapping
cat_to_supercat = {cat['name']: cat['supercategory'] for cat in cats}
cat_id_to_supercat = {cat['id']: cat['supercategory'] for cat in cats}
supercat_to_cats = {}
for key, group in itertools.groupby(
    sorted([(sc, c) for (c, sc) in cat_to_supercat.items()]), lambda x: x[0]):
    lst = [thing[1] for thing in group]
    print(key, ":", '{1}{0}'.format("\n----".join(lst), "\n----"), '\n')
    supercat_to_cats[key] = lst


# good colormap
colors = [(30,144,255), (255, 140, 0), (34,139,34), (255,0,0), (147,112,219), (139,69,19), (255,20,147), (128,128,128),
         (85,107,47), (0,255,255)]
def get_color(i):
    return colors[i % len(colors)]

if dataType == 'train2014':
    i = 149331
elif dataType == 'val2014':
    i = 258789
else:
    i = 193972

img = coco.loadImgs([i])[0]
img_pil = Image.open('%s/%s/%s'%(dataDir, dataType, img['file_name']))
plt.imshow(img_pil)
plt.show()

# load annotations for this image
annIds = coco.getAnnIds(imgIds=img['id'],  iscrowd=None)
anns = coco.loadAnns(annIds)

categories = {j:i for (i,j) in dict(enumerate(set([ann['category_id'] for ann in anns]))).items()}

img_pil = Image.open('%s/%s/%s'%(dataDir, dataType, img['file_name'])) # make sure data dir is correct
draw = ImageDraw.Draw(img_pil)

for ann in anns:
    x, y, w, h = ann['bbox']
    cat_color_id = categories[ann['category_id']] # to give each category a different color
    draw.rectangle(((x, y, x+w, y+h)), fill=None, outline=get_color(cat_color_id)) #RGB
plt.imshow(img_pil)
plt.show()


t1 = time.time()
with open(os.path.join(dataDir, 'features_tiny', '{}.p'.format(dataType)), 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    [img_list, feats] = u.load()
print('time to load features =', time.time() - t1, 'sec')
print('num images =', len(img_list))
print('shape of features =', feats.shape)

img_id = img_list[0] # any image ID to find supercategory
annIds = coco.getAnnIds(imgIds=img['id'],  iscrowd=None)
anns = coco.loadAnns(annIds)

categories = set([ann['category_id'] for ann in anns])
supercategories = set([cat_id_to_supercat[ann['category_id']] for ann in anns ])
print(supercategories) # expect singleton