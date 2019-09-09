import json,os
import pdb, time
from pycocotools.coco import COCO

####################################################################################################
datadir = "/home/ubuntu/mscoco/annotations"
json_file = "instances_train2017.json"
new_json_file = "instances_train2017_small.json"

regenerate = True 
#regenerate = False 
num = 64
images_list = [391895, 522418, 184613, 318219, 554625, 574769, 60623, 309022, 5802, 222564, 118113, 193271, 224736, 483108, 403013, 374628]
#images_list = [391895]
####################################################################################################


with open(os.path.join(datadir, json_file), 'r') as f:  
    coco = json.load(f)

if regenerate:
    images_list = [item['id'] for item in coco['images'][:num]]
    print(images_list)

new_images = []
new_annotations = []

# look for images
for img in coco['images']:
    if img['id'] in images_list:
        new_images.append(img)

# look for annotations
for ann in coco['annotations']:
    if ann['image_id'] in images_list:
        new_annotations.append(ann)

# update and save
coco['images'] = new_images
coco['annotations'] = new_annotations

print("begin to save")
with open( os.path.join(datadir, new_json_file), 'w') as ff:
    json.dump(coco, ff)



#coco2 = COCO( os.path.join(datadir, json_file) )
#pdb.set_trace()
