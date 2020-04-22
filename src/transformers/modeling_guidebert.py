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

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.modeling_albert import AlbertModel, AlbertPreTrainedModel, ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.configuration_guidebert import GuideBertConfig
from transformers.modeling_bert import ACT2FN

from .file_utils import add_start_docstrings, add_start_docstrings_to_callable


logger = logging.getLogger(__name__)


GUIDEBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    f"guidebert-{k}": v for k, v in ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP.items()
}


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


class GuidebertMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(config.embedding_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size)
        self.activation = ACT2FN[config.hidden_act]

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)

        prediction_scores = hidden_states

        return prediction_scores


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

        self.albert = AlbertModel(config)
        self.predictions = GuidebertMLMHead(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        self._tie_or_clone_weights(self.predictions.decoder, self.albert.embeddings.word_embeddings)

    def get_output_embeddings(self):
        return self.predictions.decoder

    @add_start_docstrings_to_callable(GUIDEBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
    ):
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with
            labels in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GuidebertConfig`) and inputs:
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

        from transformers import GuidebertTokenizer, GuidebertForMaskedLM
        import torch

        tokenizer = GuidebertTokenizer.from_pretrained('albert-base-v2')
        model = GuidebertForMaskedLM.from_pretrained('albert-base-v2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

        """

        # TODO: implement GuideBert here!

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_outputs = outputs[0]

        prediction_scores = self.predictions(sequence_outputs)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs


@add_start_docstrings(
    """Guidebert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    GUIDEBERT_START_DOCSTRING,
)
class GuideBertForSequenceClassification(GuideBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(GUIDEBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GuidebertConfig`) and inputs:
        loss: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        logits ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

        Examples::

            from transformers import GuidebertTokenizer, GuidebertForSequenceClassification
            import torch

            tokenizer = GuidebertTokenizer.from_pretrained('albert-base-v2')
            model = GuidebertForSequenceClassification.from_pretrained('albert-base-v2')
            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
            labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, labels=labels)
            loss, logits = outputs[:2]

        """

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings(
    """Guidebert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    GUIDEBERT_START_DOCSTRING,
)
class GuideBertForTokenClassification(GuideBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(GUIDEBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GuidebertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import GuidebertTokenizer, GuidebertForTokenClassification
        import torch

        tokenizer = GuidebertTokenizer.from_pretrained('albert-base-v2')
        model = GuidebertForTokenClassification.from_pretrained('albert-base-v2')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, scores = outputs[:2]

        """

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings(
    """Guidebert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    GUIDEBERT_START_DOCSTRING,
)
class GuideBertForQuestionAnswering(GuideBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(GUIDEBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GuidebertConfig`) and inputs:
        loss: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        end_scores: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        # The checkpoint albert-base-v2 is not fine-tuned for question answering. Please see the
        # examples/run_squad.py example to see how to fine-tune a model to a question answering task.

        from transformers import GuidebertTokenizer, GuidebertForQuestionAnswering
        import torch

        tokenizer = GuidebertTokenizer.from_pretrained('albert-base-v2')
        model = GuidebertForQuestionAnswering.from_pretrained('albert-base-v2')
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_dict = tokenizer.encode_plus(question, text, return_tensors='pt')
        start_scores, end_scores = model(**input_dict)

        """

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
