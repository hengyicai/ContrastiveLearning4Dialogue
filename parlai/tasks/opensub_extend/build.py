# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# Download and build the data if it does not exist.

import os


def build(opt):
    dpath = os.path.join(opt['datapath'], 'OpenSubExtend')
    assert os.path.exists(dpath), '[make sure the {} exists!]'.format(dpath)
