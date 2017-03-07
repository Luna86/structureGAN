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



from hico.dataset import Dataset


class HicoData(Dataset):
  """ImageNet data set."""

  def __init__(self, subset):
    super(HicoData, self).__init__('Hico', subset)

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 60

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
    
