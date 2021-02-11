# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Classes to support Encoder-Decoder architectures """


from typing import Optional

import torch

from ...activations import ACT2FN
from ...configuration_utils import PretrainedConfig
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import Seq2SeqLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_cbert import CBertConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "CBertConfig"

CBERT_START_DOCSTRING = r"""
    THIS IS OUTDATED!!!

    This class can be used to initialize a sequence-to-sequence model with any pretrained autoencoding model as the
    encoder and any pretrained autoregressive model as the decoder. The encoder is loaded via
    :meth:`~transformers.AutoModel.from_pretrained` function and the decoder is loaded via
    :meth:`~transformers.AutoModelForCausalLM.from_pretrained` function. Cross-attention layers are automatically added
    to the decoder and should be fine-tuned on a downstream generative task, like summarization.

    The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation
    tasks was shown in `Leveraging Pre-trained Checkpoints for Sequence Generation Tasks
    <https://arxiv.org/abs/1907.12461>`__ by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi
    Zhou, Wei Li, Peter J. Liu.

    After such an Encoder Decoder model has been trained/fine-tuned, it can be saved/loaded just like any other models
    (see the examples for more information).

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.T5Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

CBERT_INPUTS_DOCSTRING = r"""
    THIS IS OUTDATED!!!

    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.PreTrainedTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.PreTrainedTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__

            If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).

            Provide for sequence to sequence training to the decoder. Indices can be obtained using
            :class:`~transformers.PretrainedTokenizer`. See :meth:`transformers.PreTrainedTokenizer.encode` and
            :meth:`transformers.PreTrainedTokenizer.__call__` for details.
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.
        encoder_outputs (:obj:`tuple(torch.FloatTensor)`, `optional`):
            This tuple must consist of (:obj:`last_hidden_state`, `optional`: :obj:`hidden_states`, `optional`:
            :obj:`attentions`) :obj:`last_hidden_state` (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,
            sequence_length, hidden_size)`) is a tensor of hidden-states at the output of the last layer of the
            encoder. Used in the cross-attention of the decoder.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. This is useful if you want more control over how to convert :obj:`decoder_input_ids`
            indices into associated vectors than the model's internal embedding lookup matrix.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss for the decoder. Indices should be in ``[-100, 0,
            ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            If set to ``True``, the model will return a :class:`~transformers.file_utils.Seq2SeqLMOutput` instead of a
            plain tuple.
        kwargs: (`optional`) Remaining dictionary of keyword arguments. Keyword arguments come in two flavors:

            - Without a prefix which will be input as ``**encoder_kwargs`` for the encoder forward function.
            - With a `decoder_` prefix which will be input as ``**decoder_kwargs`` for the decoder forward function.
"""

#### taken from https://github.com/lucidrains/sinkhorn-transformer/blob/master/sinkhorn_transformer/sinkhorn_transformer.py
# and modified
def log(t, eps = 1e-6):
    return torch.log(t + eps)

def sample_gumbel(shape, device, dtype, eps=1e-5):
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(0, 1)
    return -log(-log(u, eps), eps)

# modification: allow to exclude certain row/column TODO: check, if that works!
def get_delta(r, dim, exclude=None):
    d = torch.logsumexp(r, dim=dim, keepdim=True)
    if exclude is not None:
        if dim == 1:
            #d[:,:,exclude] = 0.0
            d_ = torch.zeros_like(d)
            d_[:,:,:exclude] = d[:,:,:exclude]
            d_[:,:,exclude:] = d[:,:,exclude:]
            d = d_
        elif dim == 2:
            #d[:,exclude,:] = 0.0
            d_ = torch.zeros_like(d)
            d_[:,:exclude,:] = d[:,:exclude,:]
            d_[:,exclude:,:] = d[:,exclude:,:]
        else:
            raise Exception(f"exclusion for dim={dim} not supported")
    return d

def sinkhorn_sorting_operator(r, n_iters=8):
    n = r.shape[1]
    for _ in range(n_iters):
        r = r - torch.logsumexp(r, dim=2, keepdim=True)
        r = r - torch.logsumexp(r, dim=1, keepdim=True)
    return torch.exp(r)

def gumbel_sinkhorn(r, n_iters=8, temperature=0.7):
    r = log(r)
    gumbel = sample_gumbel(r.shape, r.device, r.dtype)
    r = (r + gumbel) / temperature
    return sinkhorn_sorting_operator(r, n_iters)

####

class PredictionHeadTransform(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class LMPredictionHead(torch.nn.Module):
    def __init__(self, config, vocab_size = None):
        super().__init__()
        self.transform = PredictionHeadTransform(config)

        vocab_size = vocab_size or config.vocab_size
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = torch.nn.Linear(config.hidden_size, vocab_size, bias=False)

        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class Teacher(torch.nn.Module):
    def __init__(
        self,
        decoder: PreTrainedModel,
        encoder_config: PretrainedConfig,
        encoder_embeddings: torch.nn.Embedding,
        # If True, predict a certain replacement token from the encoder vocabulary for each slot. 
        # Otherwise, predict one of: mask token or random token
        predict_replacement_tokens: Optional[bool] = False,
        gumbel_temperature: Optional[float] = 1.,
        # TODO: get that from model/tokenizer
        mask_token_id: Optional[int] = 103,
        # TODO: take from model (attention: encoder.config.max_length is different!!)
        max_sequence_length: Optional[int] = 512,
        # TODO: parametrize 
        mlm_percentage: Optional[float] = 0.15,
    ):  
        super().__init__()
        self.encoder_config = encoder_config
        self.encoder_embeddings = encoder_embeddings
        self.predict_replacement_tokens = predict_replacement_tokens
        self.gumbel_temperature = gumbel_temperature
        self.mask_token_id = mask_token_id
        self.max_sequence_length = max_sequence_length
        self.mlm_percentage = mlm_percentage
        self.num_slots = round(self.max_sequence_length * self.mlm_percentage)
        self.decoder = decoder
        # TODO: add similar assertion for decoder
        #assert (
        #    self.encoder.get_output_embeddings() is None
        #), "The encoder {} should not have a LM Head. Please use a model without LM Head"

        self.slot_replacement_head = LMPredictionHead(encoder_config, vocab_size=encoder_config.vocab_size if self.predict_replacement_tokens else 2)
        # TODO: check initialization
        self.c_keep = torch.nn.Parameter(torch.ones(1))

        self.crit = torch.nn.MSELoss()

        self.decoder.resize_token_embeddings(self.num_slots)
        

    def forward(
        self,
        labels,
        encoder_inputs_embeds,
        encoder_hidden_states,
        encoder_attention_mask,
        use_cache,
        past_key_values,
        kwargs_decoder,
    ):
        bs = labels.size(0)
        sl = labels.size(1)

         # stop gradient flow here
        encoder_hidden_states = encoder_hidden_states.detach()
        
        # simply use one embedding per slot 
        decoder_input_ids = torch.stack(bs * [torch.arange(self.num_slots, dtype=torch.long, device=encoder_hidden_states.device)])
        
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            #attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            #inputs_embeds=decoder_inputs_embeds,
            #labels=labels,
            #output_attentions=output_attentions,
            output_attentions=False,
            #output_hidden_states=output_hidden_states,
            output_hidden_states=False,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=True,
            **kwargs_decoder,
        )       
        decoder_hidden_states =  decoder_outputs.last_hidden_state
        
        # Here, we calculate the masked input        
        # project token encodings (encoder_hidden_states) 
        # TODO
        # project slot encodings (decoder_hidden_states)
        # TODO
        # calc position replacement scores: assign slots to token positions
        # (batch, time, dim), (batch, slot, dim) -> (batch, time, slot)
        m = torch.matmul(encoder_hidden_states, decoder_hidden_states.transpose(1,2))
        # add "constant slots" (filled with mean) that imply keeping the input
        c = torch.ones(size=(bs, sl, sl-self.num_slots), device=m.device) * self.c_keep
        m = torch.cat([m, c], dim=-1)
        # "mask" special tokens (set to very negative value)
        # TODO

        # use gamble sinkhorn to create one hot matrix from logits 
        # TODO: check if gumbel_sinkhorn really requires positive values
        m = torch.nn.functional.relu(m)
        m = gumbel_sinkhorn(m, temperature=self.gumbel_temperature)#, exclude_dim_2=0)
        # create a hard mask to prevent any leakage of information
        m = torch.nn.functional.gumbel_softmax(torch.log(m), hard=True, dim=-1)

        # calc kept input
        keep_prob = m[:,:,self.num_slots:].sum(dim=-1, keepdims=True)
        keep_embeds = encoder_inputs_embeds * keep_prob
        # calc content replacement scores: assign slot content to token postions 
        token_slot_probs = m[:,:,:self.num_slots]
        replacement_type_scores = self.slot_replacement_head(decoder_hidden_states)
        # TODO: use gumbel softmax also here?
        replacement_type_probs = torch.softmax(replacement_type_scores, dim=-1)
        # (batch, time, slot), (batch, slot, replacement) -> (batch, time, replacement)
        #token_replacement_probs = torch.matmul(token_slot_probs, replacement_type_probs)
        replacement_probs = torch.einsum("bts, bsr -> btr", token_slot_probs, replacement_type_probs)

        if self.predict_replacement_tokens:
            embedding_weights = self.encoder_embeddings.weight
            # (batch, time, token), (token, dim) -> (batch, time, dim) 
            replace_embeds = torch.matmul(replacement_probs, embedding_weights)
        else:
            random_token_ids = torch.randint(self.encoder_config.vocab_size, labels.shape, dtype=torch.long, device=replacement_probs.device)
            mask_token_ids = torch.ones_like(random_token_ids) * self.mask_token_id
            replace_ids = torch.stack([mask_token_ids, random_token_ids], dim=-1)
            embedding_weights = self.encoder_embeddings(replace_ids)
            # (batch, time, i), (batch, time, i, embedding) -> (batch, time, embedding)
            replace_embeds = torch.einsum("bti, btie -> bte", replacement_probs, embedding_weights)
        
        new_input_embeds = keep_embeds + replace_embeds

        # create "label mask" from m: only replaced tokens should be considered for loss prediction
        m_argmax = m.max(dim=-1)[1]
        new_labels = labels.clone()
        new_labels[m_argmax>=self.num_slots] = -100
        
        return new_input_embeds, new_labels

@add_start_docstrings(CBERT_START_DOCSTRING)
class CBertModel(PreTrainedModel):
    r"""
    :class:`~transformers.CBert` is a generic model class that will be instantiated as a transformer
    architecture with one of the base model classes of the library as encoder and another one as decoder when created
    with the :meth`~transformers.AutoModel.from_pretrained` class method for the encoder and
    :meth`~transformers.AutoModelForCausalLM.from_pretrained` class method for the decoder.
    """
    config_class = CBertConfig
    base_model_prefix = "cbert"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
        **teacher_kwargs
    ):
        assert config is not None or (
            encoder is not None and decoder is not None
        ), "Either a configuration or an Encoder and a decoder has to be provided"
        if config is None:
            config = CBertConfig.from_encode_decoder_configs(encoder.config, decoder.config)
        else:
            assert isinstance(config, self.config_class), "config: {} has to be of type {}".format(
                config, self.config_class
            )
        # initialize with config
        super().__init__(config)

        if encoder is None:
            #from ..auto.modeling_auto import AutoModel
            from ..auto.modeling_auto import AutoModelForMaskedLM

            encoder = AutoModelForMaskedLM.from_config(config.encoder)
        if decoder is None:
            #from ..auto.modeling_auto import AutoModelForCausalLM
            from ..auto.modeling_auto import AutoModel

            decoder = AutoModel.from_config(config.decoder)

        
        self.student = encoder
        self.teacher = Teacher(
            decoder=decoder, 
            encoder_config=self.student.config, 
            encoder_embeddings=self.student.get_input_embeddings(), 
            **teacher_kwargs
        )

    def get_encoder(self):
        return self.student

    def get_decoder(self):
        return self.teacher.decoder

    def get_input_embeddings(self):
        return self.student.get_input_embeddings()

    def get_output_embeddings(self):
        return self.get_decoder().get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.get_decoder().set_output_embeddings(new_embeddings)

    @classmethod
    def from_cbert_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs
    ) -> PreTrainedModel:
        r"""
        THIS IS OUTDATED!!!

        Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model
        checkpoints.


        The model is set in evaluation mode by default using :obj:`model.eval()` (Dropout modules are deactivated). To
        train the model, you need to first set it back in training mode with :obj:`model.train()`.

        Params:
            encoder_pretrained_model_name_or_path (:obj: `str`, `optional`):
                Information necessary to initiate the encoder. Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                Information necessary to initiate the decoder. Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args (remaining positional arguments, `optional`):
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method.

            kwargs (remaining dictionary of keyword arguments, `optional`):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                :obj:`output_attentions=True`).

                - To update the encoder configuration, use the prefix `encoder_` for each configuration parameter.
                - To update the decoder configuration, use the prefix `decoder_` for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a :obj:`config` is provided or automatically loaded.

        Example::

            >>> from transformers import CBertModel
            >>> # initialize a bert2bert from two pretrained BERT models. Note that the cross-attention layers will be randomly initialized
            >>> model = CBertModel.from_cbert_pretrained('bert-base-uncased', 'bert-base-uncased')
            >>> # saving model after fine-tuning
            >>> model.save_pretrained("./bert2bert")
            >>> # load fine-tuned model
            >>> model = CBertModel.from_pretrained("./bert2bert")

        """

        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            assert (
                encoder_pretrained_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has to be defined"
            #from ..auto.modeling_auto import AutoModel
            from ..auto.modeling_auto import AutoModelForMaskedLM

            if "config" not in kwargs_encoder:
                from ..auto.configuration_auto import AutoConfig

                encoder_config = AutoConfig.from_pretrained(encoder_pretrained_model_name_or_path)
                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:

                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModelForMaskedLM.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            assert (
                decoder_pretrained_model_name_or_path is not None
            ), "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has to be defined"
            #from ..auto.modeling_auto import AutoModelForCausalLM
            from ..auto.modeling_auto import AutoModel

            if "config" not in kwargs_decoder:
                from ..auto.configuration_auto import AutoConfig

                decoder_config = AutoConfig.from_pretrained(decoder_pretrained_model_name_or_path)
                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` passed to `.from_cbert_pretrained(...)` are set to `True` or do not pass a `decoder_config` to `.from_cbert_pretrained(...)`"
                )

            decoder = AutoModel.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = CBertConfig.from_encode_decoder_configs(encoder.config, decoder.config, **kwargs)
        return cls(encoder=encoder, decoder=decoder, config=config)

    @add_start_docstrings_to_model_forward(CBERT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        THIS IS OUTDATED!!!

        Returns:

        Examples::

            >>> from transformers import CBertModel, BertTokenizer
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = CBertModel.from_cbert_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert from pre-trained checkpoints

            >>> # forward
            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)

            >>> # training
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
            >>> loss, logits = outputs.loss, outputs.logits

            >>> # save and load from pretrained
            >>> model.save_pretrained("bert2bert")
            >>> model = CBertModel.from_pretrained("bert2bert")

            >>> # generation
            >>> generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        #torch.autograd.set_detect_anomaly(True)

        generate_masking = self.training

        if generate_masking:
            # TODO: consider special tokens (label should be set to -100 for these positions)
            labels = input_ids

        if encoder_outputs is None:
            encoder_outputs = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=generate_masking or output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )

        if not generate_masking:
            return encoder_outputs
        
        # TODO: stop gradient flow here for student training (or use dedicated optimizer for the trainer and student)

        if inputs_embeds is None:
            inputs_embeds = self.student.get_input_embeddings()(input_ids)

        new_input_embeds, new_labels = self.teacher(
            labels=labels,
            encoder_inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_outputs.hidden_states[-1], 
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            kwargs_decoder=kwargs_decoder
        )

        # calc loss with new input
        encoder_outputs = self.student(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=new_input_embeds,
            labels=new_labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs_encoder,
        )
        # calculate trainer loss
        student_loss = encoder_outputs["loss"]
        n_predict = sum(new_labels != -100).sum() 
        student_loss_mean = student_loss / n_predict
        # TODO: is this a good idea?
        sl_e = torch.exp(-student_loss_mean)
        
        encoder_outputs["losses"] = {
            # TODO: parametrize target loss value (0.8) and weight
            'teacher': self.teacher.crit(sl_e.unsqueeze(-1), torch.tensor([0.8], device=sl_e.device)) * 100,
            'student': encoder_outputs["loss"],
        }

        ## TODO: dont do the following! handle losses by different optimizers that are linked only to student / teacher parameters
        #encoder_outputs["loss"] = sum(encoder_outputs["losses"].values())
        
        return encoder_outputs
        
    
    # TODO: remove(?)
    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        decoder_inputs = self.get_decoder().prepare_inputs_for_generation(input_ids, past=past)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        return self.get_decoder()._reorder_cache(past, beam_idx)
