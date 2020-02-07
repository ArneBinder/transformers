# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" PyTorch T5 model. """


import copy
import itertools
import logging
import math
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _WeightedLoss

from .configuration_t5 import T5Config
from .file_utils import DUMMY_INPUTS, DUMMY_MASK, add_start_docstrings
from .modeling_utils import PreTrainedModel, prune_linear_layer


logger = logging.getLogger(__name__)

ENSURE_DEFAULT_RELATIVE_POSITION = False

####################################################
# This dict contrains shortcut names and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "t5-small": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-small-pytorch_model.bin",
    "t5-base": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-base-pytorch_model.bin",
    "t5-large": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-large-pytorch_model.bin",
    "t5-3b": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-3b-pytorch_model.bin",
    "t5-11b": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-11b-pytorch_model.bin",
}


####################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    tf_weights = {}
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array

    for txt_name in names:
        name = txt_name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            tf_weights.pop(txt_name, None)
            continue
        if "_slot_" in name[-1]:
            logger.info("Skipping {}".format("/".join(name)))
            tf_weights.pop(txt_name, None)
            continue
        pointer = model
        array = tf_weights[txt_name]
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] in ["kernel", "scale", "embedding"]:
                pointer = getattr(pointer, "weight")
            # elif scope_names[0] == 'scale':
            #     pointer = getattr(pointer, 'weight')
            # elif scope_names[0] == 'output_bias' or scope_names[0] == 'beta':
            #     pointer = getattr(pointer, 'bias')
            # elif scope_names[0] == 'squad':
            #     pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if scope_names[0] not in ["kernel", "scale", "embedding"]:
            pointer = getattr(pointer, "weight")
        if scope_names[0] != "embedding":
            logger.info("Transposing numpy weight of shape {} for {}".format(array.shape, name))
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)

    logger.info("Weights not copied to PyTorch model: {}".format(", ".join(tf_weights.keys())))
    # logger.info("Weights not copied to PyTorch model: {}".format(', '.join(tf_weights.keys())))
    return model


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of torch.nn.Module)
####################################################


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """ Construct a layernorm module in the T5 style
            No bias and no substraction of mean.
        """
        super(T5LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(variance + self.variance_epsilon)
        return self.weight * x


class T5DenseReluDense(nn.Module):
    def __init__(self, config):
        super(T5DenseReluDense, self).__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        h = self.wi(hidden_states)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.wo(h)
        return h


class T5LayerFF(nn.Module):
    def __init__(self, config):
        super(T5LayerFF, self).__init__()
        self.DenseReluDense = T5DenseReluDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        norm_x = self.layer_norm(hidden_states)
        y = self.DenseReluDense(norm_x)
        layer_output = hidden_states + self.dropout(y)
        return layer_output


class T5Attention(nn.Module):
    NEW_ID = itertools.count()
    RELATIVE_POSITION_SPECIAL_OFFSET = 1000000
    RELATIVE_POSITION_INF = RELATIVE_POSITION_SPECIAL_OFFSET
    RELATIVE_POSITION_UNK = RELATIVE_POSITION_SPECIAL_OFFSET + 1
    RELATIVE_POSITION_PAD = RELATIVE_POSITION_SPECIAL_OFFSET + 2
    RELATIVE_POSITION_NUM_BUCKETS_SPECIAL = 3

    def __init__(self, config, has_relative_attention_bias=False):
        super(T5Attention, self).__init__()
        self.layer_id = next(T5Attention.NEW_ID)
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.output_attentions = config.output_attentions
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_num_buckets_special = config.relative_attention_num_buckets_special
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.d_kv
        self.initializer_factor = config.initializer_factor

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets + self.relative_attention_num_buckets_special, self.n_heads)
        self.pruned_heads = set()

    def add_relative_attention_bias_special_embeddings(self, num_special):
        if self.has_relative_attention_bias:
            if self.relative_attention_num_buckets_special == 0:
                self.relative_attention_num_buckets_special = num_special
            if self.relative_attention_bias.num_embeddings < self.relative_attention_num_buckets + self.relative_attention_num_buckets_special:
                old_embeddings = self.relative_attention_bias
                old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
                self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets + self.relative_attention_num_buckets_special, self.n_heads)
                self.relative_attention_bias.to(old_embeddings.weight.device)
                # taken from self._init_weights(new_embeddings)
                self.relative_attention_bias.weight.data.normal_(mean=0.0, std=self.initializer_factor * ((self.d_model) ** -0.5))
                # Copy word embeddings from the previous weights
                self.relative_attention_bias.weight.data[:old_num_tokens, :] = old_embeddings.weight.data[:old_num_tokens, :]

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_heads, self.d_kv)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.d_kv * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    @staticmethod
    def _relative_position_bucket_to_indices(relative_position_buckets, num_buckets, bidirectional,
                                             max_distance=128):

        relative_position = relative_position_buckets
        if bidirectional:
            num_buckets //= 2
            is_neg = relative_position_buckets < num_buckets
            #relative_position[~is_neg] -= num_buckets
            relative_position = torch.where(is_neg, relative_position, relative_position - num_buckets)
        else:
            raise NotImplementedError('_relative_position_bucket_to_indices is nto implemented for bidirectional=False')
            is_neg = torch.zeros_like(relative_position_buckets, dtype=torch.bool) # this is wrong

        # values are now in [0, num_buckets]

        is_max = relative_position == (num_buckets - 1)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position_buckets < max_exact

        is_log_scaled = ~(is_small | is_max)
        # set to lowest log scaled value max_exact
        #relative_position[is_log_scaled] = max_exact
        #relative_position = torch.where(is_log_scaled, torch.ones_like(relative_position) * max_exact, relative_position)
        # closest possible approximation
        relative_position_normalized = (relative_position - max_exact).float() * (math.log(max_distance / max_exact) / (num_buckets - max_exact)) + math.log(max_exact)
        relative_position = torch.where(is_log_scaled, torch.exp(relative_position_normalized).to(torch.long), relative_position)

        # relative_position[is_max] = max_distance
        relative_position = torch.where(is_max, torch.ones_like(relative_position) * max_distance, relative_position)

        #relative_position[is_neg] *= -1
        relative_position = torch.where(is_neg, -relative_position, relative_position)

        return relative_position

    @staticmethod
    def _relative_position_bucket_with_special(relative_position, relative_attention_num_buckets,
                                               relative_attention_num_buckets_special, bidirectional,
                                               relative_position_special_offset=None
                                               ):

        if relative_attention_num_buckets_special > 0:
            assert relative_position_special_offset is not None, \
            f'relative_attention_num_buckets_special > 0 [{relative_attention_num_buckets_special}], ' \
            f'but missing relative_position_special_offset'
            mask_special = relative_position >= relative_position_special_offset
            rp_bucket_special = torch.where(mask_special,
                                            relative_position
                                            - relative_position_special_offset + relative_attention_num_buckets,
                                            torch.zeros_like(relative_position))
            #relative_position[mask_special] = 0
            relative_position = torch.where(mask_special, torch.zeros_like(relative_position), relative_position)
        else:
            mask_special = None
            rp_bucket_special = None

        rp_bucket = T5Attention._relative_position_bucket(
            relative_position,  # shape (qlen, klen) or (bsz, qlen, klen)
            bidirectional=bidirectional,
            num_buckets=relative_attention_num_buckets,
        )

        if mask_special is not None:
            rp_bucket = torch.where(mask_special, rp_bucket_special, rp_bucket)

        return rp_bucket

    @staticmethod
    def _relative_position_bucket_with_special_to_indices(relative_position_buckets, relative_attention_num_buckets,
                                                          relative_attention_num_buckets_special, bidirectional,
                                                          relative_position_special_offset=None):

        if relative_attention_num_buckets_special > 0:
            assert relative_position_special_offset is not None, \
            f'relative_attention_num_buckets_special > 0 [{relative_attention_num_buckets_special}], ' \
            f'but missing relative_position_special_offset'
            mask_special = relative_position_buckets >= relative_attention_num_buckets
            rp_indices_special = torch.where(mask_special,
                                             relative_position_buckets
                                             - relative_attention_num_buckets + relative_position_special_offset,
                                             torch.zeros_like(relative_position_buckets))
            relative_position_buckets[mask_special] = 0
        else:
            mask_special = None
            rp_indices_special = None

        rp_indices = T5Attention._relative_position_bucket_to_indices(
            relative_position_buckets=relative_position_buckets,
            num_buckets=relative_attention_num_buckets,
            bidirectional=bidirectional
        )
        if mask_special is not None:
            rp_indices = torch.where(mask_special, rp_indices_special, rp_indices)

        return rp_indices

    def compute_bias(self, relative_position):
        """
        Compute binned relative position bias.
        relative_position may be batch wise i.e. shape: (bsz, qlen, klen), same for all batches, i.e. shape: (qlen, klen),
        or same for all distances, i.e. shape: (1,).
        Args:
            relative_position: an int32 Tensor
        Returns:
            a float Tensor of shape (bsz, num_heads, qlen, klen)
        """

        rp_buckets = T5Attention._relative_position_bucket_with_special(
            relative_position=relative_position,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            relative_attention_num_buckets_special=self.relative_attention_num_buckets_special,
            relative_position_special_offset=T5Attention.RELATIVE_POSITION_SPECIAL_OFFSET,
            #bidirectional=not self.is_decoder,
            bidirectional=True  # we allow looking forward in the decoder TODO: check!
        )

        values = self.relative_attention_bias(rp_buckets)

        if len(values.size()) < 3:  # same relative_positions for all q and k positions (used for special cross-graph distance)
            assert len(values.size()) == 1 or values.size(0) == 1, f'expected a single embedding, but got {values.size(0)}'
            values = values.view(1, 1, values.size(-1))  # add q and k dimensions
        if len(values.size()) == 3:  # same relative_positions for all batches
            values = values.unsqueeze(0)  # add batch dimension

        values = values.permute([0, 3, 1, 2])  # shape (bsz, num_heads, qlen, klen) where bsz, qlen and klen may be 1 for broadcasting
        return values

    def forward(self, input, mask=None, kv=None, position_bias=None, cache=None, head_mask=None, relative_position=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if cache is None else cache["slen"] + qlen
        else:
            klen = kv.size(1)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, self.d_kv).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)

        q = shape(self.q(input))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(input))  # (bs, n_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(v))  # (bs, n_heads, qlen, dim_per_head)

        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                    v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)

        # q = q / math.sqrt(dim_per_head)                                     # No scaling in T5
        scores = torch.einsum("bnqd,bnkd->bnqk", q, k)  # (bs, n_heads, qlen, klen)

        if position_bias is None:
            if not self.has_relative_attention_bias:
                raise ValueError("No position_bias provided and no weights to compute position_bias")
            if relative_position is None or ENSURE_DEFAULT_RELATIVE_POSITION:
                relative_position_default = sequential_relative_position(qlen=qlen, klen=klen)  # shape (qlen, klen)
                if relative_position is not None: #and ENSURE_DEFAULT_RELATIVE_POSITION:
                    assert torch.equal(relative_position_default.view(*relative_position.size()), relative_position), \
                        'external relative_position do not match original calculation'
                relative_position = relative_position_default

            #assert relative_position is not None, 'no relative_position available'
            position_bias = self.compute_bias(relative_position)
            if mask is not None:
                position_bias = position_bias + mask  # (bs, n_heads, qlen, klen)

        scores += position_bias
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        context = self.o(context)

        outputs = (context,)
        if self.output_attentions:
            outputs = outputs + (weights,)
        if self.has_relative_attention_bias:
            outputs = outputs + (position_bias,)
        return outputs


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super(T5LayerSelfAttention, self).__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, head_mask=None, relative_position=None):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            norm_x, mask=attention_mask, position_bias=position_bias, head_mask=head_mask,
            relative_position=relative_position
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super(T5LayerCrossAttention, self).__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, kv, attention_mask=None, position_bias=None, head_mask=None,
                relative_position=None):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            norm_x, mask=attention_mask, kv=kv, position_bias=position_bias, head_mask=head_mask,
            relative_position=relative_position
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


def sequential_relative_position(qlen=None, klen=None, hidden_states=None, kv=None):
    if qlen is None:
        assert hidden_states is not None, 'if no qlen is given, hidden_states are required'
        qlen = hidden_states.size(1)
    if kv is not None:
        klen = kv.size(1)
    if klen is None:
        klen = qlen  # if cache is None else cache["slen"] + qlen
    context_position = torch.arange(qlen, dtype=torch.long)[:, None]
    memory_position = torch.arange(klen, dtype=torch.long)[None, :]
    relative_position = memory_position - context_position  # shape (qlen, klen)
    return relative_position


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super(T5Block, self).__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config, has_relative_attention_bias=has_relative_attention_bias))
            self.layer.append(T5LayerFF(config))
        else:
            self.layer.append(T5LayerFF(config))

    def add_relative_attention_bias_special_embeddings(self, num_special):
        self.layer[0].SelfAttention.add_relative_attention_bias_special_embeddings(num_special)
        if self.is_decoder:
            self.layer[1].EncDecAttention.add_relative_attention_bias_special_embeddings(num_special)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        head_mask=None,
        relative_position=None,
        encoder_decoder_relative_position=None, # in fact, that are the decoder-to-encoder relative_positions
    ):
        self_attention_outputs = self.layer[0](
            hidden_states, attention_mask=attention_mask, position_bias=position_bias, head_mask=head_mask,
            relative_position=relative_position,# sequential_relative_position(hidden_states=hidden_states)
        )
        hidden_states = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # Keep self-attention outputs and relative position weights

        if not self.is_decoder:
            hidden_states = self.layer[1](hidden_states)
        else:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                kv=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                head_mask=head_mask,
                relative_position=encoder_decoder_relative_position,
            )
            hidden_states = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:]
            )  # Keep cross-attention outputs and relative position weights
            hidden_states = self.layer[2](hidden_states)

        outputs = (hidden_states,) + outputs  # add attentions if we output them
        return outputs  # hidden-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)


class T5PreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = T5Config
    pretrained_model_archive_map = T5_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_t5
    base_model_prefix = "transformer"

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "encoder_input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """ Initialize the weights """
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (T5Model, T5WithLMHeadModel)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            d_kv = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * d_kv) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * d_kv) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))


class T5Stack(T5PreTrainedModel):
    def __init__(self, config):
        super(T5Stack, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()

    def add_relative_attention_bias_special_embeddings(self, num_special):
        for b in self.block:
            b.add_relative_attention_bias_special_embeddings(num_special)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        relative_position=None,
        encoder_decoder_relative_position=None # in fact, that are the decoder-to-encoder relative_positions
    ):
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length).to(hidden_states.device)
        if self.is_decoder and encoder_attention_mask is None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(batch_size, encoder_seq_length).to(hidden_states.device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                seq_ids = torch.arange(seq_length, device=hidden_states.device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(attention_mask)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -1e9 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
        # extended_attention_mask = (extended_attention_mask == extended_attention_mask.transpose(-1, -2))

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

        if self.is_decoder:
            # If a 2D ou 3D attention mask is provided for the cross-attention
            # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

            # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
            # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
            # encoder_extended_attention_mask = (encoder_extended_attention_mask == encoder_extended_attention_mask.transpose(-1, -2))

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_layers

        all_hidden_states = ()
        all_attentions = ()
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(hidden_states)
        for i, layer_module in enumerate(self.block):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                head_mask=head_mask[i],
                relative_position=relative_position,
                encoder_decoder_relative_position=encoder_decoder_relative_position
            )
            # layer_outputs is a tuple with:
            # hidden-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states = layer_outputs[0]
            if i == 0:
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[2 if self.output_attentions else 1]
                if self.is_decoder:
                    encoder_decoder_position_bias = layer_outputs[4 if self.output_attentions else 2]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)  # We keep only self-attention weights for now

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


T5_START_DOCSTRING = r"""    The T5 model was proposed in
    `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer`_
    by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.
    It's an encoder decoder transformer pre-trained in a text-to-text denoising generative setting.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer`:
        https://arxiv.org/abs/1910.10683

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.T5Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

T5_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, T5 input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``

            T5 is a model with relative position embeddings so you should be able to pad the inputs on
            the right or the left.

            Indices can be obtained using :class:`transformers.T5Tokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""


@add_start_docstrings(
    "The bare T5 Model transformer outputting raw hidden-states" "without any specific head on top.",
    T5_START_DOCSTRING,
    T5_INPUTS_DOCSTRING,
)
class T5Model(T5PreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5Model.from_pretrained('t5-small')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids=input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """

    def __init__(self, config):
        super(T5Model, self).__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        self.encoder = T5Stack(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = T5Stack(decoder_config)

        self.init_weights()


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, **kwargs):
        encoder_decoder_relative_position = kwargs.pop("encoder_decoder_relative_position", None)
        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as whole.
        # We let the specific kwargs override the common ones in case of conflict.
        kwargs_common = dict(
            (k, v) for k, v in kwargs.items() if not k.startswith("encoder_") and not k.startswith("decoder_")
        )
        kwargs_encoder = kwargs_common.copy()
        kwargs_decoder = kwargs_common.copy()
        kwargs_encoder.update(dict((k[len("encoder_") :], v) for k, v in kwargs.items() if k.startswith("encoder_")))
        kwargs_decoder.update(dict((k[len("decoder_") :], v) for k, v in kwargs.items() if k.startswith("decoder_")))

        # Encode if needed (training, first prediction pass)
        encoder_hidden_states = kwargs_encoder.pop("hidden_states", None)
        encoder_attention_mask = kwargs_encoder.get("attention_mask", None)
        if encoder_hidden_states is None:
            # Convert encoder inputs in embeddings if needed
            hidden_states = kwargs_encoder.pop("inputs_embeds", None)
            if hidden_states is None:
                encoder_inputs_ids = kwargs_encoder.pop("input_ids")
                hidden_states = self.shared(encoder_inputs_ids)  # Convert inputs in embeddings

            if encoder_attention_mask is not None:
                # Apply masking
                encoder_attention_mask = (encoder_attention_mask != 0).to(hidden_states)
                hidden_states = hidden_states * encoder_attention_mask.unsqueeze(-1)

            encoder_outputs = self.encoder(hidden_states, **kwargs_encoder)
            encoder_hidden_states = encoder_outputs[0]
        else:
            encoder_outputs = ()

        # Decode
        # Convert decoder inputs in embeddings if needed
        hidden_states = kwargs_decoder.pop("inputs_embeds", None)
        if hidden_states is None:
            decoder_inputs_ids = kwargs_decoder.pop("input_ids")
            hidden_states = self.shared(decoder_inputs_ids)

        kwargs_decoder["encoder_hidden_states"] = encoder_hidden_states
        kwargs_decoder["encoder_attention_mask"] = encoder_attention_mask
        kwargs_decoder["encoder_decoder_relative_position"] = encoder_decoder_relative_position
        decoder_outputs = self.decoder(hidden_states, **kwargs_decoder)

        return decoder_outputs + encoder_outputs


@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING, T5_INPUTS_DOCSTRING)
class T5WithLMHeadModel(T5PreTrainedModel):
    r"""
        **lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5WithLMHeadModel.from_pretrained('t5-small')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids=input_ids, lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """

    def __init__(self, config):
        super(T5WithLMHeadModel, self).__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        self.encoder = T5Stack(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = T5Stack(decoder_config)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, **kwargs):
        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as whole.
        # We let the specific kwargs override the common ones in case of conflict.

        lm_labels = kwargs.pop("decoder_lm_labels", None)
        encoder_decoder_relative_position = kwargs.pop("encoder_decoder_relative_position", None)

        kwargs_common = dict(
            (k, v) for k, v in kwargs.items() if not k.startswith("encoder_") and not k.startswith("decoder_")
        )
        kwargs_encoder = kwargs_common.copy()
        kwargs_decoder = kwargs_common.copy()
        kwargs_encoder.update(dict((k[len("encoder_") :], v) for k, v in kwargs.items() if k.startswith("encoder_")))
        kwargs_decoder.update(dict((k[len("decoder_") :], v) for k, v in kwargs.items() if k.startswith("decoder_")))

        # Encode if needed (training, first prediction pass)
        encoder_hidden_states = kwargs_encoder.pop("hidden_states", None)
        if encoder_hidden_states is None:
            # Convert encoder inputs in embeddings if needed
            hidden_states = kwargs_encoder.pop("inputs_embeds", None)
            if hidden_states is None:
                encoder_inputs_ids = kwargs_encoder.pop("input_ids")
                hidden_states = self.shared(encoder_inputs_ids)  # Convert inputs in embeddings

            encoder_outputs = self.encoder(hidden_states, **kwargs_encoder)
            encoder_hidden_states = encoder_outputs[0]
        else:
            encoder_outputs = ()

        # Decode
        # Convert decoder inputs in embeddings if needed
        hidden_states = kwargs_decoder.pop("inputs_embeds", None)
        if hidden_states is None:
            decoder_inputs_ids = kwargs_decoder.pop("input_ids")
            hidden_states = self.shared(decoder_inputs_ids)

        kwargs_decoder["encoder_hidden_states"] = encoder_hidden_states
        kwargs_decoder["encoder_attention_mask"] = kwargs_encoder.get("attention_mask", None)
        kwargs_decoder["encoder_decoder_relative_position"] = encoder_decoder_relative_position
        decoder_outputs = self.decoder(hidden_states, **kwargs_decoder)

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        decoder_outputs = (lm_logits,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            decoder_outputs = (
                loss,
            ) + decoder_outputs  # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        return decoder_outputs + encoder_outputs


class MultiGoldCrossEntropyLoss(_WeightedLoss):
    """
    Calculates Cross Entropy loss regarding multiple correct label instances. Only the loss for instances with
    minimal loss is considered. Instances where every label matches its ignore_index are omitted.
    Expects list of logits (inputs), each of shape (bs, ..., dim) and list of labels (targets), each of shape (bs, instances, ...).
    Works also for lists of logits and targets (loss is calculated between logits[i] and targets[i]).
    Returns list of aggregated minimal losses (one loss value for each entry in inputs / targets)
    """
    def __init__(self, weight=None, ignore_indices=-100, reduction='mean', weights=None):
        super(MultiGoldCrossEntropyLoss, self).__init__(weight=weight, reduction=reduction)
        self.ignore_indices = ignore_indices
        if not isinstance(self.ignore_indices, (list, tuple)):
            self.ignore_indices = (self.ignore_indices, )
        self.weights = weights

    def forward(self, inputs, targets):
        if not isinstance(inputs,  (list, tuple)):
            inputs = (inputs,)
        if not isinstance(targets,  (list, tuple)):
            targets = (targets,)
        assert len(inputs) == len(targets), \
            f'number of elements in inputs [{len(inputs)}] and targets [{len(targets)}] ' \
            f'do not match'
        all_losses = []
        for i in range(len(inputs)):
            input = inputs[i]       # shape (bs, ..., dims)
            target = targets[i]     # shape (bs, instances, ...)
            #bs = target.size(0)
            n = target.size(1)
            d = input.size(-1)
            assert tuple(input.size())[1:-1] == tuple(target.size())[2:], \
                'remaining dimensions of input and target must match'

            not_exp_dims = [-1] * (len(input.size())-1)
            # add and expand instance dim in input, then flat it
            input_expanded = input.unsqueeze(1).expand(-1, n, *not_exp_dims)
            current_losses = F.cross_entropy(input_expanded.reshape(-1, d),
                                             target.reshape(-1), weight=self.weight,
                                             ignore_index=self.ignore_indices[i], reduction='none').reshape(target.size())
            # weight loss for individual input-target pairs
            if self.weights is not None:
                current_losses = current_losses * self.weights[i]
            # set loss to infinite where all targets are ignored for one instance (caused by padding) because this
            # would otherwise cause minimal possible loss (0)
            mask_ignore = (target == self.ignore_indices[i])
            # ignore only complete gold instances (aggregate up to _gold_ instance level != instance lavel)
            for _ in range(len(mask_ignore.size()) - 2):
                mask_ignore, _ = mask_ignore.min(-1)
            current_losses[mask_ignore] = float('inf')
            all_losses.append(current_losses)

        losses = torch.cat(all_losses, -1)

        losses_reduced = []
        if self.reduction == 'mean':
            losses = losses.mean(-1)
            losses, _indices = losses.min(-1)
            for l in all_losses:
                # TODO: allows only one intermediate dim (i.e. sequence data)
                _indices_expanded = _indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, l.size(-1))
                current_losses = l.gather(dim=1, index=_indices_expanded)
                losses_reduced.append(current_losses.mean())
            #loss = losses.mean()
        elif self.reduction == 'sum':
            losses = losses.sum(-1)
            losses, _indices = losses.min(-1)
            for l in all_losses:
                # TODO: allows only one intermediate dim (i.e. sequence data)
                _indices_expanded = _indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, l.size(-1))
                current_losses = l.gather(dim=1, index=_indices_expanded)
                losses_reduced.append(current_losses.sum())
            #loss = losses.sum()
        elif self.reduction == 'none':
            raise AttributeError(f'"none" not allowed as reduction function')
        else:
            raise NotImplementedError(f'unknown reduction function: {self.reduction}')

        return losses_reduced


@add_start_docstrings("""T5 Model with a `language modeling` head and a `relative position prediction` head on top. """,
                      T5_START_DOCSTRING, T5_INPUTS_DOCSTRING)
class T5WithLMAndRPPHeadModel(T5PreTrainedModel):
    r"""
        **label_indices**: ``torch.LongTensor`` of shape ``(batch_size, ): indices of output
            (decoding) tokens to calculate the lm_logits and relative positions for (one per instance / sequence!)
        **lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, instance)``:
            Labels for computing the masked language modeling loss for multiple correct instances at positions indexed
            by ``label_indices``.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **relative_position_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, instance, sequence_length, )``:
            relative positions with respect to tokens indexed by ``label_indices``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5WithLMHeadModel.from_pretrained('t5-small')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids=input_ids, lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """

    ARGS_TRAIN = ["encoder_input_ids", "decoder_input_ids", "encoder_attention_mask", "decoder_attention_mask",
                  "encoder_relative_position", "decoder_relative_position", "encoder_decoder_relative_position",
                  "decoder_label_indices", "decoder_lm_labels", "decoder_relative_position_labels",
                  "encoder_decoder_relative_position_labels"]

    def __init__(self, config):
        super(T5WithLMAndRPPHeadModel, self).__init__(config)
        config.relative_position_hidden_states_dim = 100

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_num_buckets_special = config.relative_attention_num_buckets_special
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        self.encoder = T5Stack(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = T5Stack(decoder_config)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # TODO: decide for bias and activation
        self.encoder_relative_position_projection = nn.Linear(config.d_model,
                                                              config.relative_position_hidden_states_dim,
                                                              bias=False)
        self.decoder_relative_position_projection = nn.Linear(config.d_model,
                                                              config.relative_position_hidden_states_dim, bias=False)
        self.new_relative_postion_projection = nn.Linear(config.d_model,
                                                         config.relative_position_hidden_states_dim, bias=False)

        self.relative_position_head = nn.Linear(config.relative_position_hidden_states_dim * 3,
                                                config.relative_attention_num_buckets
                                                + T5Attention.RELATIVE_POSITION_NUM_BUCKETS_SPECIAL,
                                                bias=False)

        self.init_weights()

    def add_relative_attention_bias_special_embeddings(self, num_special=T5Attention.RELATIVE_POSITION_NUM_BUCKETS_SPECIAL):
        self.encoder.add_relative_attention_bias_special_embeddings(num_special)
        self.decoder.add_relative_attention_bias_special_embeddings(num_special)
        return num_special

    @staticmethod
    def args_train_encoder():
        return [arg for arg in T5WithLMAndRPPHeadModel.ARGS_TRAIN
                if not (arg.startswith('decoder_') or arg.startswith('encoder_decoder_'))]

    @staticmethod
    def args_train_decoder():
        return [arg for arg in T5WithLMAndRPPHeadModel.ARGS_TRAIN if arg not in T5WithLMAndRPPHeadModel.args_train_encoder()]

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, **kwargs):
        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as whole.
        # We let the specific kwargs override the common ones in case of conflict.

        encoder_decoder_relative_position = kwargs.pop("encoder_decoder_relative_position", None)

        lm_labels = kwargs.pop("decoder_lm_labels", None)
        return_predictions = kwargs.pop('return_predictions', False)

        encoder_decoder_relative_position_labels = kwargs.pop("encoder_decoder_relative_position_labels", None)
        decoder_relative_position_labels = kwargs.pop("decoder_relative_position_labels", None)
        decoder_label_indices = kwargs.pop("decoder_label_indices", None)

        kwargs_common = dict(
            (k, v) for k, v in kwargs.items() if not k.startswith("encoder_") and not k.startswith("decoder_")
        )
        kwargs_encoder = kwargs_common.copy()
        kwargs_decoder = kwargs_common.copy()
        kwargs_encoder.update(dict((k[len("encoder_") :], v) for k, v in kwargs.items() if k.startswith("encoder_")))
        kwargs_decoder.update(dict((k[len("decoder_") :], v) for k, v in kwargs.items() if k.startswith("decoder_")))

        # Encode if needed (training, first prediction pass)
        encoder_hidden_states = kwargs_encoder.pop("hidden_states", None)
        encoder_relative_position_hidden_states = kwargs_encoder.pop("encoder_relative_position_hidden_states", None)
        if encoder_hidden_states is None:
            # Convert encoder inputs in embeddings if needed
            hidden_states = kwargs_encoder.pop("inputs_embeds", None)
            if hidden_states is None:
                encoder_inputs_ids = kwargs_encoder.pop("input_ids")
                hidden_states = self.shared(encoder_inputs_ids)  # Convert inputs in embeddings

            encoder_outputs = self.encoder(hidden_states, **kwargs_encoder)
            encoder_hidden_states = encoder_outputs[0]
            encoder_output_names = ('hidden_states',)
            if self.encoder.output_hidden_states:
                encoder_output_names = encoder_output_names + ('all_hidden_states',)
            if self.encoder.output_attentions:
                encoder_output_names = encoder_output_names + ('all_attentions',)
        else:
            encoder_outputs = ()
            encoder_output_names = ()

        if encoder_relative_position_hidden_states is None:
            encoder_relative_position_hidden_states = self.encoder_relative_position_projection(encoder_hidden_states)
            encoder_outputs = encoder_outputs + (encoder_relative_position_hidden_states,)
            encoder_output_names = encoder_output_names + ('relative_position_hidden_states',)

        # Decode
        # Convert decoder inputs in embeddings if needed
        hidden_states = kwargs_decoder.pop("inputs_embeds", None)
        if hidden_states is None:
            decoder_inputs_ids = kwargs_decoder.pop("input_ids")
            hidden_states = self.shared(decoder_inputs_ids)

        kwargs_decoder["encoder_hidden_states"] = encoder_hidden_states
        kwargs_decoder["encoder_attention_mask"] = kwargs_encoder.get("attention_mask", None)
        kwargs_decoder["encoder_decoder_relative_position"] = encoder_decoder_relative_position
        decoder_outputs = self.decoder(hidden_states, **kwargs_decoder)
        decoder_hidden_states = decoder_outputs[0] # shape (bs, sl, dim)
        decoder_output_names = ('hidden_states',)
        if self.decoder.output_hidden_states:
            decoder_output_names = decoder_output_names + ('all_hidden_states',)
        if self.decoder.output_attentions:
            decoder_output_names = decoder_output_names + ('all_attentions',)

        len_decoder_seq = decoder_hidden_states.size(1)

        def gather_from_sequence(t, idx):
            _idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, t.size(-1))
            return t.gather(1, _idx).squeeze(1)

        new_decoder_hidden_state = gather_from_sequence(decoder_hidden_states, decoder_label_indices)
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = new_decoder_hidden_state * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        #losses = []
        #decoder_outputs = (lm_logits,) + decoder_outputs #[1:]  # Add hidden states and attention if they are here
        #if lm_labels is not None:
        #    #shift_logits = lm_logits[..., :-1, :].contiguous()
        #    #shift_labels = lm_labels[..., 1:].contiguous()
        #    loss_fct = MultiGoldCrossEntropyLoss(ignore_indices=(-100,))
        #    loss = loss_fct(lm_logits, lm_labels)
        #    #decoder_outputs = (
        #    #    loss,
        #    #) + decoder_outputs  # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
        #    losses = losses + (loss,)

        decoder_relative_position_hidden_states = self.decoder_relative_position_projection(decoder_hidden_states)
        decoder_outputs = decoder_outputs + (decoder_relative_position_hidden_states,)
        decoder_output_names = decoder_output_names + ('relative_position_hidden_states',)

        new_decoder_relative_position_hidden_state = self.new_relative_postion_projection(new_decoder_hidden_state)

        relative_position_hidden_states = torch.cat((encoder_relative_position_hidden_states,
                                                     decoder_relative_position_hidden_states), dim=1)

        if decoder_relative_position_labels is not None:
            assert decoder_label_indices is not None, \
                'argument decoder_relative_position_indices is missing, but required when decoder_relative_' \
                'position_labels is given'

        # multiply with each entry in relative_position_hidden_states
        new_decoder_relative_position_hidden_state_expanded = new_decoder_relative_position_hidden_state.unsqueeze(1).expand_as(relative_position_hidden_states)
        prod = new_decoder_relative_position_hidden_state_expanded * relative_position_hidden_states
        # finally, project with self.relative_position_head
        # TODO: test using only product
        relative_position_logits = self.relative_position_head(
            torch.cat([prod,
                       new_decoder_relative_position_hidden_state_expanded,
                       relative_position_hidden_states],
                      dim=-1))

        outputs = (lm_logits, relative_position_logits) + decoder_outputs + encoder_outputs
        output_names = ('lm_logits', 'relative_position_logits') + tuple('decoder_'+ don for don in decoder_output_names) + tuple('encoder_' + eon for eon in encoder_output_names)

        if return_predictions:
            _, lm_predictions = lm_logits.detach().max(-1)
            _, rp_indices = relative_position_logits.detach().max(-1)
            # shift relative position bucket indices back to relative positions and special indices
            rp_predictions = T5Attention._relative_position_bucket_with_special_to_indices(
                relative_position_buckets=rp_indices,
                relative_attention_num_buckets=self.relative_attention_num_buckets,
                relative_attention_num_buckets_special=self.relative_attention_num_buckets_special,
                relative_position_special_offset=T5Attention.RELATIVE_POSITION_SPECIAL_OFFSET,
                bidirectional=True
            )
            # slice indices to (encoder_decoder_relative_position_indices, decoder_relative_position_indices)
            # along dim=1
            encoder_decoder_relative_position_predictions = rp_predictions[:, :-len_decoder_seq]
            decoder_relative_position_predictions = rp_predictions[:, -len_decoder_seq:]
            # prepend to outputs
            outputs = (lm_predictions, decoder_relative_position_predictions,
                       encoder_decoder_relative_position_predictions) + outputs
            output_names = ('lm_predictions', 'decoder_relative_position_predictions',
                            'encoder_decoder_relative_position_predictions') + output_names

        relative_position_labels = decoder_relative_position_labels
        if relative_position_labels is not None and lm_labels is not None:
            # prepend encoder-to-decoder labels
            if encoder_decoder_relative_position_labels is not None:
                assert decoder_relative_position_labels is not None, \
                    'decoder_relative_position_labels is required if encoder_decoder_relative_position_labels is given'
                relative_position_labels = torch.cat((encoder_decoder_relative_position_labels,
                                                      relative_position_labels), dim=-1)
            else:
                # take only _decoder_ relative position logits for loss calculation, if just these labels are provided
                relative_position_logits = relative_position_logits[:, len_decoder_seq:]

            # convert to buckets
            relative_position_labels_buckets = T5Attention._relative_position_bucket_with_special(
                relative_position=relative_position_labels,
                relative_attention_num_buckets=self.relative_attention_num_buckets,
                relative_attention_num_buckets_special=self.relative_attention_num_buckets_special,
                relative_position_special_offset=T5Attention.RELATIVE_POSITION_SPECIAL_OFFSET,
                bidirectional=True
            )

            # language modeling and relative position loss calculation with multiple correct labels have to be
            # calculated together
            loss_fct = MultiGoldCrossEntropyLoss(ignore_indices=(-100, T5Attention.RELATIVE_POSITION_PAD))
            losses = loss_fct(inputs=(lm_logits.unsqueeze(1), relative_position_logits),
                              targets=(lm_labels.unsqueeze(-1), relative_position_labels_buckets))
            # prepend lm and distance loss
            outputs = tuple(losses) + outputs
            output_names = ('loss_lm', 'loss_rp') + output_names

        # REMINDER: convert relative_position_logits.max(-1)[1] indices back to relative distances
        # (reverse T5Attention._relative_position_bucket_with_special)
        return outputs + (output_names,)