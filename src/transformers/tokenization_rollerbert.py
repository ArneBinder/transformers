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
""" Tokenization classes for ROLLERBERT model."""


import logging

from .tokenization_albert import AlbertTokenizer, PRETRAINED_VOCAB_FILES_MAP, VOCAB_FILES_NAMES, \
    PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

logger = logging.getLogger(__name__)


class RollerbertTokenizer(AlbertTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = {
        "vocab_file": {
            f"rollerbert-{k}": v for k, v in PRETRAINED_VOCAB_FILES_MAP["vocab_file"].items()
        }
    }
    max_model_input_sizes = {f"rollerbert-{k}": v for k, v in PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES.items()}

