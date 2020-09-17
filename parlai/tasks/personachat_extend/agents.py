# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import copy
import os

from parlai.core.teachers import FbDialogTeacher
from .build import build


def _path(opt):
    # Build the data if it doesn't exist.
    build(opt)
    datatype = opt['datatype'].split(':')[0]
    dt = datatype
    return os.path.join(opt['datapath'], 'PersonaChatExtend', dt + '.txt')


class DefaultTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt)
        super().__init__(opt, shared)
