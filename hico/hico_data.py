# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Small library that points to the ImageNet data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
from hico.dataset import Dataset
import scipy.io
import numpy as np

class HicoData(Dataset):
  """ImageNet data set."""

  def __init__(self, subset, datadir):
    super(HicoData, self).__init__('Hico', subset, datadir)
    self.read_meta()
    self.read_label()
    self.read_bboxes()

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 600

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data set."""
    # Bounding box data consists of 615299 bounding boxes for 544546 images.
    if self.subset == 'train':
      return 38116
    if self.subset == 'test':
      return 9529

  def download_message(self):
    """Instruction to download and extract the tarball from Flowers website."""
    print("No download message")

  def read_meta(self):
    # read the list of objects
    self.list_obj = []
    list_obj_file = os.path.join(self.datadir, 'hico_list_obj.txt')
    with open(list_obj_file, 'r') as f:
      line_cnt = 0
      for line in f.readlines():
        line_cnt += 1
        if line_cnt >= 3:
          index, name = line.rstrip('\n').split()
          self.list_obj.append(name)
    
    # read the list of actions
    self.list_action = []
    list_action_file = os.path.join(self.datadir, 'hico_list_vb.txt')
    with open(list_action_file, 'r') as f:
      line_cnt = 0
      for line in f.readlines():
        line_cnt += 1
        if line_cnt >= 3:
          index, name = line.rstrip('\n').split()
          self.list_action.append(name)

    # read the list of relations
    anno_file = os.path.join(self.datadir, 'anno.mat')
    mat = scipy.io.loadmat(anno_file) 
    relations_raw = mat['list_action']
    self.list_relation = []
    for r in relations_raw:
      obj = r[0][0][0]
      obj_index = self.list_obj.index(obj)
      act = r[0][1][0]
      act_index = self.list_action.index(act)
      relation = [obj, act, obj_index, act_index]
      self.list_relation.append(relation)

    self.num_obj = len(self.list_obj)
    self.num_action = len(self.list_action)
    self.num_relation = len(self.list_relation)

  def read_label(self):
    #read the label data from the dataset
    path = os.path.join(self.datadir, 'anno.mat')
    mat = scipy.io.loadmat(path)
    if self.subset == 'train':
      label_raw = mat['anno_train']
    else:
      label_raw = mat['anno_test']
    self.label = np.zeros(label_raw.shape, dtype=np.int32) 

    for row in range(0, self.label.shape[0]):
      for col in range(0, self.label.shape[1]):
        entry = label_raw[row, col]
        if entry == 1.0:
          self.label[row, col] = 1
        elif entry == 0:
          self.label[row, col] = 0
        elif entry == -1.0:
          self.label[row, col] = -1
        else:
          self.label[row, col] = -2

  def read_bboxes(self):
    # read bounding boxes from the dataset
    path = os.path.join(self.datadir, 'anno_bbox.mat')
    mat = scipy.io.loadmat(path)
    if self.subset == 'train':
      bboxes_raw = mat['bbox_train']
    else:
      bboxes_raw = mat['bbox_test']
   
    self.bboxes = [] 
    for i in range(self.label.shape[1]):
      relations = bboxes_raw[0][i][2][0]
      bboxes = []
      for j in range(len(relations)):
        bboxes_relation = []
        connections = relations[j][3]
        for k in range(len(connections)):
          h1, h2 = connections[k][0], connections[k][1]
          h = relations[j][1][0][h1-1]
          o = relations[j][2][0][h2-1]
          ho = np.zeros([2, 4])
          ho[0, :] = np.array([int(h[m][0][0]) for m in range(4)])
          ho[1, :] = np.array([int(o[m][0][0]) for m in range(4)])
          bboxes_relation.append(ho)
        bboxes.append(bboxes_relation)
      self.bboxes.append(bboxes)
