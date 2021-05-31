#!/usr/bin/env python
# coding: utf-8

# # Basic Path Setup

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '')
from personlab.data import coco

base_dir = '/home/ubuntu/personlab-tf/dataset/coco/'
#base_dir = '/Users/minsubsim/work/personlab-tf/dataset/coco/'
anno_dir = base_dir + 'annotations/'
train_base_dir = base_dir + 'train2017/'
val_base_dir = base_dir + 'val2017/'

train_inst_json = anno_dir + 'instances_train2017.json'
train_person_json = anno_dir + 'person_keypoints_train2017.json'
val_inst_json = anno_dir + 'instances_val2017.json'
val_person_json = anno_dir + 'person_keypoints_val2017.json'


# # Training Script

# In[ ]:


from personlab.model import train
from personlab.models.mobilenet_v2 import mobilenet_v2_model
gen = coco.CocoDataGenerator(train_base_dir, train_inst_json, train_person_json)

pm_check_path = 'pretrained/mobilenet/mobilenet_v2_1.0_224.ckpt'
log_dir = 'logs/sample/'

train(mobilenet_v2_model, gen.loader, pm_check_path, log_dir)


# # Evaluation Script

# In[79]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '')
import tensorflow as tf
from personlab.model import evaluate
from personlab.models.mobilenet_v2 import mobilenet_v2_model
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

tf.reset_default_graph()
latest_ckp = tf.train.latest_checkpoint('./')


gen = coco.CocoDataGenerator(val_base_dir, val_inst_json, val_person_json)

checkpoint_dir = 'logs/sample/'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

#print_tensors_in_checkpoint_file(checkpoint_path, all_tensors=False, tensor_name='', all_tensor_names=True)
output = evaluate(mobilenet_v2_model, gen.loader, checkpoint_path, num_batches=2)


# In[80]:


from matplotlib import pyplot as plt
from personlab import display, config
import numpy as np
plt.rcParams['figure.figsize'] = [20, 20]
b_i = 5
plt.figure()
plt.title('Original Image')
plt.imshow(output['image'][b_i])

plt.figure()

plt.subplot(2, 2, 1)
plt.title('Skeleton(True)')
plt.imshow(display.summary_skeleton(output['image'][b_i], output['kp_map_true'][b_i]))

plt.subplot(2, 2, 2)
plt.title('Segmentation(True)')
plt.imshow(display.show_heatmap(output['image'][b_i], output['seg_true'][b_i]))

plt.subplot(2, 2, 3)
plt.title('Skeleton(Prediction)')
plt.imshow(display.summary_skeleton(output['image'][b_i], output['kp_map_pred'][b_i]))

plt.subplot(2, 2, 4)
plt.title('Segmentation(Prediction)')
plt.imshow(display.show_heatmap(output['image'][b_i], output['seg_pred'][b_i]))

