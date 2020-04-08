# coding=utf-8
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
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
""" Tokenization classes for ALBERT model."""


import logging
import os
import unicodedata
from shutil import copyfile
from typing import List, Optional

from .tokenization_albert import AlbertTokenizer


from .tokenization_utils import PreTrainedTokenizer


logger = logging.getLogger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "rollerbert-albert-base-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v1-spiece.model",
        "rollerbert-albert-large-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v1-spiece.model",
        "rollerbert-albert-xlarge-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-v1-spiece.model",
        "rollerbert-albert-xxlarge-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-v1-spiece.model",
        "rollerbert-albert-base-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v2-spiece.model",
        "rollerbert-albert-large-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-spiece.model",
        "rollerbert-albert-xlarge-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-v2-spiece.model",
        "rollerbert-albert-xxlarge-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-v2-spiece.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "rollerbert-albert-base-v1": 512,
    "rollerbert-albert-large-v1": 512,
    "rollerbert-albert-xlarge-v1": 512,
    "rollerbert-albert-xxlarge-v1": 512,
    "rollerbert-albert-base-v2": 512,
    "rollerbert-albert-large-v2": 512,
    "rollerbert-albert-xlarge-v2": 512,
    "rollerbert-albert-xxlarge-v2": 512,
}

SPIECE_UNDERLINE = "▁"


class RollerbertTokenizer(AlbertTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

