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
"""PyTorch GUIDEBERT model. """

import logging
from typing import Optional
from math import exp

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import gumbel_softmax

from .modeling_albert import AlbertModel, AlbertPreTrainedModel, AlbertMLMHead, ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP, \
    AlbertForQuestionAnswering, AlbertForTokenClassification, AlbertForSequenceClassification
from .configuration_guidebert import GuideBertConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable


logger = logging.getLogger(__name__)


GUIDEBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    f"guidebert-{k}": v for k, v in ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP.items()
}


class GradientReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.saved_tensors[0]
        return scale * grad_output.neg(), None


def grad_reverse(x: torch.FloatTensor, scale: Optional[torch.FloatTensor] = None):
    if scale is None:
        scale = torch.ones_like(x)
    if not isinstance(scale, torch.Tensor):
        scale = torch.ones_like(x) * scale
    return GradientReverse.apply(x, scale)


class GuideBertPreTrainedModel(AlbertPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = GuideBertConfig
    pretrained_model_archive_map = GUIDEBERT_PRETRAINED_MODEL_ARCHIVE_MAP


GUIDEBERT_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Args:
        config (:class:`~transformers.GuideBertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

GUIDEBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.GuideBertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        input_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


@add_start_docstrings(
    "The bare GuideBert (Albert) Model transformer outputting raw hidden-states without any specific head on top.",
    GUIDEBERT_START_DOCSTRING,
)
class GuideBertModel(AlbertModel):

    config_class = GuideBertConfig
    pretrained_model_archive_map = GUIDEBERT_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings(
    "GuideBert Model with a `language modeling` head on top.", GUIDEBERT_START_DOCSTRING,
)
class GuideBertForMaskedLM(GuideBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.mask_token_id = config.mask_token_id
        self.pad_token_id = config.pad_token_id

        self.generate_masking_on_train = config.generate_masking_on_train
        self.generate_masking_on_eval = config.generate_masking_on_eval

        self.lambda_mask_loss = config.lambda_mask_loss
        self.lambda_adv_gradient = config.lambda_adv_gradient
        self.p_mask_target = config.p_mask_target
        self.tau_r = config.tau_r

        self.t = 0

        self.albert = AlbertModel(config)
        self.predictions = AlbertMLMHead(config)

        self.classifier = nn.Linear(config.hidden_size, 2)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        self._tie_or_clone_weights(self.predictions.decoder, self.albert.embeddings.word_embeddings)

    def get_output_embeddings(self):
        return self.predictions.decoder

    def get_tau(self):
        if self.training:
            # use schedule of https://arxiv.org/pdf/1611.01144.pdf
            tau = max(0.5, exp(-self.tau_r * self.t))
            self.t += 1
            return tau
        else:
            # in eval mode, use categorical distribution (tau -> 0)
            return 0.01

    @add_start_docstrings_to_callable(GUIDEBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None
    ):
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with
            labels in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GuideBertConfig`) and inputs:
        loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Example::

        from transformers import GuideBertTokenizer, GuideBertForMaskedLM
        import torch

        tokenizer = GuideBertTokenizer.from_pretrained('guidebert-albert-base-v2')
        model = GuideBertForMaskedLM.from_pretrained('guidebert-albert-base-v2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1

        # GuideBert defaults to generate masks for training...
        model.train()
        outputs = model(input_ids)
        loss, prediction_scores = outputs[:2]

        # ...but assumes masked_lm_labels for evaluation. The boolean config entries generate_masking_on_train and
        # generate_masking_on_eval allow to change this behavior.
        model.eval()
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]


        ## For a full training example, use (taken and adapted from
        ##  https://github.com/huggingface/transformers/tree/master/examples#robertabert-and-masked-language-modeling):
        #
        # export TRAIN_FILE=/path/to/dataset/wiki.train.raw
        # export TEST_FILE=/path/to/dataset/wiki.test.raw
        #
        # python run_language_modeling.py \
        #     --output_dir=output \
        #     --model_type=guidebert \
        #     --model_name_or_path=guidebert-albert-base-v2 \
        #     --do_train \
        #     --train_data_file=$TRAIN_FILE \
        #     --do_eval \
        #     --eval_data_file=$TEST_FILE \
        #     --mlm
        """

        # get initial embeddings
        embedding_output = self.albert.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        loss_mask = 0.0
        if (self.training and self.generate_masking_on_train) or (not self.training and self.generate_masking_on_eval):
            #if masked_lm_labels is not None:
            #    # this spams the console...
            #    logger.warning(f'GuideBert generates masking during {"training" if self.training else "evaluation"}, '
            #                   f'but masked_lm_labels is provided (will be overridden)!')
            input_ids_mask = torch.ones_like(input_ids) * self.mask_token_id
            embedding_mask = self.albert.embeddings(
                input_ids_mask, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=None
            )

            # first position is original input, second is masked
            embedding_choice = torch.stack((embedding_output, embedding_mask), dim=2)
            mask_padding = input_ids == self.pad_token_id

            outputs = self.albert(
                input_ids=None,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=embedding_output,
            )
            sequence_output = outputs[0]
            logits = self.classifier(sequence_output)
            # WARNING: tau starts from 1.0 when training is (re-)started, also if model is loaded from checkpoint!
            tau = self.get_tau()
            hard = gumbel_softmax(logits=logits, tau=tau, hard=True)

            # do not allow padding positions for masking
            hard[mask_padding.unsqueeze(-1) * torch.BoolTensor([True, False]).to(mask_padding.device).unsqueeze(0).unsqueeze(0)] = 1.0
            hard[mask_padding.unsqueeze(-1) * torch.BoolTensor([False, True]).to(mask_padding.device).unsqueeze(0).unsqueeze(0)] = 0.0

            # index == 1 indicates masking
            to_mask = hard[:, :, 1]
            to_mask[mask_padding] = 0.0

            to_mask_not = hard[:, :, 0]
            to_mask_not[mask_padding] = 0.0

            # calculate amount of
            n_mask = to_mask.sum()
            n_mask_not = to_mask_not.sum()
            p_mask = n_mask.float() / (n_mask + n_mask_not)

            loss_mask = (p_mask - self.p_mask_target).pow(2)
            loss_mask *= self.lambda_mask_loss

            # Apply gradient inversion. Scale by inverted amount of masked tokens:
            # As more tokens are masked, as less this is a good signal.
            # Or: High gradients from _few_ masked tokens are better.
            hard = grad_reverse(hard, scale=self.lambda_adv_gradient)

            # select mask or real input depending on "hard softmax"
            embedding_output = (embedding_choice * hard.unsqueeze(dim=-1)).sum(dim=2)

            # do not calculate loss for not-masked tokens by setting labels to ignore_index
            masked_lm_labels = input_ids.clone()
            masked_lm_labels[~to_mask.bool()] = -100     # CrossEntropyLoss ignore_index

        # default language masking functionality (see Albert model)
        outputs = self.albert(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=embedding_output,
        )
        sequence_output = outputs[0]

        prediction_scores = self.predictions(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss + loss_mask,) + outputs + (masked_lm_labels,)

        return outputs


@add_start_docstrings(
    """GuideBert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    GUIDEBERT_START_DOCSTRING,
)
class GuideBertForSequenceClassification(AlbertForSequenceClassification):
    pass


@add_start_docstrings(
    """GuideBert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    GUIDEBERT_START_DOCSTRING,
)
class GuideBertForTokenClassification(AlbertForTokenClassification):
    pass


@add_start_docstrings(
    """GuideBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    GUIDEBERT_START_DOCSTRING,
)
class GuideBertForQuestionAnswering(AlbertForQuestionAnswering):
    pass
