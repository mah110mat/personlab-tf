import os
import datetime

from personlab.data import coco

base_dir = 'dataset/coco/'
anno_dir = base_dir + 'annotations/'
train_base_dir = base_dir + 'train2017/'
val_base_dir = base_dir + 'val2017/'

train_inst_json = anno_dir + 'instances_train2017.json'
train_person_json = anno_dir + 'person_keypoints_train2017.json'
val_inst_json = anno_dir + 'instances_val2017.json'
val_person_json = anno_dir + 'person_keypoints_val2017.json'


# # Training Script

from personlab.model import train
from personlab.models.mobilenet_v2 import mobilenet_v2_model
gen = coco.CocoDataGenerator(train_base_dir, train_inst_json, train_person_json)

#pm_check_path = 'pretrained/mobilenet/mobilenet_v2_1.0_224.ckpt'
pm_check_path = 'logs/sample/model.ckpt'
now = datetime.datetime.now()
log_dir = 'logs/{}/'.format(now.strftime('%Y%m%d_%H%M%S'))
os.makedirs(log_dir, exist_ok=True)

train(mobilenet_v2_model, gen.loader, pm_check_path, log_dir)
