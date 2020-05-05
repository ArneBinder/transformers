# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" GUIDEBERT model configuration """

from .configuration_albert import AlbertConfig


GUIDEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "guidebert-albert-base-v1": "https://raw.githubusercontent.com/ArneBinder/GuideBert/master/transformers/bert/guidebert-albert-base-v1-config.json",
    "guidebert-albert-large-v1": "https://raw.githubusercontent.com/ArneBinder/GuideBert/master/transformers/bert/guidebert-albert-large-v1-config.json",
    "guidebert-albert-xlarge-v1": "https://raw.githubusercontent.com/ArneBinder/GuideBert/master/transformers/bert/guidebert-albert-xlarge-v1-config.json",
    "guidebert-albert-xxlarge-v1": "https://raw.githubusercontent.com/ArneBinder/GuideBert/master/transformers/bert/guidebert-albert-xxlarge-v1-config.json",
    "guidebert-albert-base-v2": "https://raw.githubusercontent.com/ArneBinder/GuideBert/master/transformers/bert/guidebert-albert-base-v2-config.json",
    "guidebert-albert-large-v2": "https://raw.githubusercontent.com/ArneBinder/GuideBert/master/transformers/bert/guidebert-albert-large-v2-config.json",
    "guidebert-albert-xlarge-v2": "https://raw.githubusercontent.com/ArneBinder/GuideBert/master/transformers/bert/guidebert-albert-xlarge-v2-config.json",
    "guidebert-albert-xxlarge-v2": "https://raw.githubusercontent.com/ArneBinder/GuideBert/master/transformers/bert/guidebert-albert-xxlarge-v2-config.json",
}


class GuideBertConfig(AlbertConfig):
    r"""
        This is the configuration class to store the configuration of an :class:`~transformers.GuideBertModel`.
        It is used to instantiate an GUIDEBERT model according to the specified arguments, defining the model
        architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
        the GUIDEBERT `xxlarge <https://huggingface.co/albert-xxlarge-v2>`__ architecture.

        Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
        to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
        for more information.


        Args:
            vocab_size (:obj:`int`, optional, defaults to 30000):
                Vocabulary size of the GUIDEBERT model. Defines the different tokens that
                can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.GuideBertModel`.
            embedding_size (:obj:`int`, optional, defaults to 128):
                Dimensionality of vocabulary embeddings.
            hidden_size (:obj:`int`, optional, defaults to 4096):
                Dimensionality of the encoder layers and the pooler layer.
            num_hidden_layers (:obj:`int`, optional, defaults to 12):
                Number of hidden layers in the Transformer encoder.
            num_hidden_groups (:obj:`int`, optional, defaults to 1):
                Number of groups for the hidden layers, parameters in the same group are shared.
            num_attention_heads (:obj:`int`, optional, defaults to 64):
                Number of attention heads for each attention layer in the Transformer encoder.
            intermediate_size (:obj:`int`, optional, defaults to 16384):
                The dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
            inner_group_num (:obj:`int`, optional, defaults to 1):
                The number of inner repetition of attention and ffn.
            hidden_act (:obj:`str` or :obj:`function`, optional, defaults to "gelu_new"):
                The non-linear activation function (function or string) in the encoder and pooler.
                If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob (:obj:`float`, optional, defaults to 0):
                The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob (:obj:`float`, optional, defaults to 0):
                The dropout ratio for the attention probabilities.
            max_position_embeddings (:obj:`int`, optional, defaults to 512):
                The maximum sequence length that this model might ever be used with. Typically set this to something
                large (e.g., 512 or 1024 or 2048).
            type_vocab_size (:obj:`int`, optional, defaults to 2):
                The vocabulary size of the `token_type_ids` passed into :class:`~transformers.GuideBertModel`.
            initializer_range (:obj:`float`, optional, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
                The epsilon used by the layer normalization layers.
            classifier_dropout_prob (:obj:`float`, optional, defaults to 0.1):
                The dropout ratio for attached classifiers.

        Example::

            from transformers import GuideBertConfig, GuideBertModel
            # Initializing an GUIDEBERT-xxlarge style configuration
            albert_xxlarge_configuration = GuideBertConfig()

            # Initializing an GUIDEBERT-base style configuration
            albert_base_configuration = GuideBertConfig(
                hidden_size=768,
                num_attention_heads=12,
                intermediate_size=3072,
            )

            # Initializing a model from the GUIDEBERT-base style configuration
            model = GuideBertModel(albert_xxlarge_configuration)

            # Accessing the model configuration
            configuration = model.config

        Attributes:
            pretrained_config_archive_map (Dict[str, str]):
                A dictionary containing all the available pre-trained checkpoints.
    """

    pretrained_config_archive_map = GUIDEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "guidebert"

    def __init__(
        self,
        mask_token_id=4,
        # TODO: check/rework this condition!
        # If enabled only for training, generate masks as GuideBert from input_ids.
        # This would allow for default mask generation during evaluation.
        generate_masking_on_train=True,
        generate_masking_on_eval=False,
        lambda_mask_loss=0.5,
        lambda_adv_gradient=1.0,
        mlm_probability=0.15,
        # https://arxiv.org/pdf/1611.01144.pdf proposes values in {1e−5,1e−4}
        # and uses 3e-5 in "4.3 Generative Semi-Supervised-Classification".
        tau_r=3e-5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mask_token_id = mask_token_id
        self.generate_masking_on_train = generate_masking_on_train
        self.generate_masking_on_eval = generate_masking_on_eval
        self.lambda_mask_loss = lambda_mask_loss
        self.lambda_adv_gradient = lambda_adv_gradient
        self.mlm_probability = mlm_probability
        self.tau_r = tau_r