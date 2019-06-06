# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq import hub_utils, options
from fairseq.models import MODEL_REGISTRY


dependencies = [
    'torch',
    'sacremoses',
    'subword_nmt',
]


for model, cls in MODEL_REGISTRY.items():

    def get_model_fn(*args, **kwargs):
        model = cls.from_pretrained(*args, **kwargs)
        return model

    globals()[model] = get_model_fn


def generator(*args, **kwargs):
    return hub_utils.Generator.from_pretrained(*args, **kwargs)
