"""Utilities for distributed encoding"""
from typing import Dict, List, Optional, Union, Tuple
import re
import os
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import Tensor
from torch.profiler import profile, record_function, ProfilerActivity
import transformers
from transformers.activations import ACT2FN
from lm_eval.tasks.dist_enc_tasks import SegmentedSample
from lm_eval.models.dist_enc_utils import BaseModelType, ParameterlessAttentionDecoder, TokenizerType
from lm_eval.models.dist_enc_utils import instrument, do_profile, trace_handler, get_max_length


def get_logger(logger_name: str, logger_level: Union[int, str, None] = None) -> logging.Logger:
    """Create logger, formatter, and console handler and return logger."""
    logger = logging.Logger(logger_name)  # using getLogger instead results in duplicate logs
    logger_level = os.environ.get('LOGLEVEL', 'WARNING').upper() if logger_level is None else logger_level
    logger.setLevel(logger_level)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logger_level)
        formatter = logging.Formatter('%(levelname)s: %(asctime)s: %(name)s: %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


_LOGGER = get_logger(__name__)


def str_to_bool(arg: Optional[str]) -> bool:
    """Convert parameter string to bool"""
    if arg is None:
        return False
    arg = arg.lower()
    assert arg in ['true', 'false']
    return arg == 'true'


def normalize(norm, T: Tensor, *, dim: int = -1, eps=1e-6) -> Tensor:
    """Compute norm."""
    if norm is None:
        return T
    elif norm == 'L2':
        return torch.nn.functional.normalize(T, p=2, dim=dim)
    elif norm == 'layer':
        raise ValueError('"layer" is not supported. Use zNorm or varNorm instead.')
    elif norm == 'varNorm':
        return T / T.std(dim=dim, keepdim=True)
    elif norm == 'zNorm':
        # return (T - T.mean(dim=dim, keepdim=True)) / T.std(dim=dim, keepdim=True)
        assert dim == -1
        return torch.nn.functional.layer_norm(T, T.shape[-1:])
    # elif norm == 'rms':
    #     variance = T.pow(2).mean(dim=dim, keepdim=True)
    #     return T * torch.rsqrt(variance + eps)
    else:
        raise NotImplementedError


class ModuleContainer(torch.nn.Module):
    """
    Container structure to hold assorted layers and benefit from torch.Module's functions.
    Needed for GPU memory management.
    """

    def __init__(self, lm: BaseModelType, tokenizer: TokenizerType) -> None:
        super().__init__()
        self.lm: BaseModelType = lm
        self.tokenizer: TokenizerType = tokenizer
        self.decoder: torch.nn.Module
        self.encoder: torch.nn.Module
        self.input_embeddings: torch.nn.Embedding
        self.out_token_embeddings: torch.nn.Embedding
        self.wpe: torch.nn.Module  # Position Embeddings
        self.agg_weights: torch.Tensor
        self.act: torch.nn.Module  # dense_act_fn
        self.is_enc_dec = hasattr(self.lm, 'get_encoder')
        self.max_length = get_max_length(self.lm, self.tokenizer)

    def no_grad(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def detach(module: torch.nn.Module):
        for param in module.parameters():
            param.detach_()
        return module


class DistEncSimMixin:
    def __init__(self,
                 *args,
                 WORD_AGG_SCHEME: Optional[str] = None,
                 OUT_WORD_AGG_SCHEME: Optional[str] = None,
                 SEGMENT_AGG_SCHEME: Optional[str] = None,  # TODO: was mean. need to retest with absent values
                 EXAMPLE_AGG_SCHEME: Optional[str] = None,  # TODO: was mean. need to retest with absent values
                 SIMILARITY_FUNC: Optional[str] = None,
                 NORM: Optional[str] = None,
                 ENCODING_LAYER: Optional[str] = None,
                 OUT_ENCODING_LAYER: Optional[str] = None,
                 #  PARALLELIZE: Optional[str] = None,
                 ADD_POS: Optional[str] = None,
                 del_lm: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # CONFIGURATION
        self.WORD_AGG_SCHEME: Optional[str] = WORD_AGG_SCHEME if WORD_AGG_SCHEME != 'None' else None
        self.OUT_WORD_AGG_SCHEME = self.WORD_AGG_SCHEME if OUT_WORD_AGG_SCHEME in [
            None, 'None'] else OUT_WORD_AGG_SCHEME
        self.SEGMENT_AGG_SCHEME: Optional[str] = SEGMENT_AGG_SCHEME if SEGMENT_AGG_SCHEME != 'None' else None
        self.EXAMPLE_AGG_SCHEME: Optional[str] = EXAMPLE_AGG_SCHEME if EXAMPLE_AGG_SCHEME != 'None' else None
        self.SIMILARITY_FUNC: Optional[str] = SIMILARITY_FUNC if SIMILARITY_FUNC != 'None' else None
        self.NORM: Optional[str] = NORM if NORM != 'None' else None
        self.ENCODING_LAYER: Union[str, int, None] = ENCODING_LAYER if ENCODING_LAYER != 'None' else None
        self.OUT_ENCODING_LAYER: Union[str, int, None] = OUT_ENCODING_LAYER if OUT_ENCODING_LAYER != 'None' else None
        self.ADD_POS: bool = str_to_bool(ADD_POS)

        # Setup the torch layers
        # All the torch layers go here for GPU memory management
        self.module = ModuleContainer(self.gpt2, self.tokenizer)
        del self.gpt2
        if hasattr(self.module.lm.config, 'activation_function'):
            self.module.act = ACT2FN[self.module.lm.config.activation_function]
        elif isinstance(self.module.lm, (transformers.T5PreTrainedModel, transformers.MT5PreTrainedModel)):  # T5, MT5
            self.module.act = ACT2FN[self.module.lm.config.dense_act_fn]
        elif isinstance(self.module.lm, transformers.BloomPreTrainedModel):
            self.module.act = self.module.lm.transformer.h[0].mlp.gelu_impl
        elif isinstance(self.module.lm, transformers.GPTNeoXPreTrainedModel):
            self.module.act = ACT2FN[self.module.lm.config.hidden_act]
        else:
            raise NotImplementedError(
                f"Don't know how to extract relu activation for model tuype {type(self.module.lm)}")

        if self.ENCODING_LAYER != 'E' or self.OUT_ENCODING_LAYER not in ['E', 'OE']:
            self.module.encoder = self.module.lm.get_encoder() if self.is_enc_dec else self.module.lm
        # self.module.decoder = self.module.lm
        self.module.input_embeddings = self.module.lm.get_input_embeddings()
        if self.OUT_ENCODING_LAYER == 'OE':
            oE: torch.nn.Linear = self.module.lm.get_output_embeddings()
            self.module.out_token_embeddings = torch.nn.Embedding(
                oE.weight.shape[0], oE.weight.shape[1], _weight=oE.weight.detach())

        if self.module.tokenizer.pad_token is None:
            self.module.tokenizer.pad_token = self.module.tokenizer.eos_token
        if self.ADD_POS:
            assert not self.is_enc_dec  # T5 uses relative pos-encoding, therefore ADD_POS does not apply.
            assert hasattr(self.module.lm, 'transformer') and hasattr(
                self.module.lm.transformer, 'wpe'), 'Could not find position embedding matrix'
            self.module.wpe = self.module.lm.transformer.wpe
        if (self.WORD_AGG_SCHEME is not None and 'w1mean' in self.WORD_AGG_SCHEME) or (
            self.OUT_WORD_AGG_SCHEME is not None and 'w1mean' in self.OUT_WORD_AGG_SCHEME
        ):
            self.module.agg_weights = torch.arange(
                1, self.max_length + 1, device=self.device).unsqueeze(0)  # (1, max_len)
        if del_lm:
            del self.module.lm
        self.module.no_grad()
        # torch.cuda.empty_cache()

    @property
    def device(self):
        return self._device

    @property
    def is_enc_dec(self) -> bool:
        """Return True if this is an encoder-decoder model like T5."""
        return self.module.is_enc_dec

    @property
    # Overriding HFLM.max_length
    def max_length(self):
        return self.module.max_length

    @classmethod
    def _verify_encoding_layer(cls, ENCODING_LAYER: Union[int, str, None]):
        assert (ENCODING_LAYER in ['middle', None, 'E']) or isinstance(
            ENCODING_LAYER, int) or (re.fullmatch(r'[-+]?\d+', ENCODING_LAYER) is not None), f'Invalid ENCODING_LAYER config {ENCODING_LAYER}'
        if isinstance(ENCODING_LAYER, str) and (match := re.fullmatch(r'[-+]?\d+', ENCODING_LAYER)):
            ENCODING_LAYER = int(ENCODING_LAYER)
        return ENCODING_LAYER

    def _verify_out_encoding_layer(self, OUT_ENCODING_LAYER: Union[int, str, None]):
        # A None config value always implies "revert to old behaviour before this config setting was introduced". In this case
        # it implies OUT_ENCODING_LAYER = IN_ENCODING_LAYER because that's how it was with earlier runs. This enables comparison
        # of new results with older ones.
        if OUT_ENCODING_LAYER is None:
            OUT_ENCODING_LAYER = self.ENCODING_LAYER
        elif (OUT_ENCODING_LAYER not in ['OE']):
            OUT_ENCODING_LAYER = self._verify_encoding_layer(OUT_ENCODING_LAYER)
        return OUT_ENCODING_LAYER

    def verify_config(self):
        if self.WORD_AGG_SCHEME is not None:
            assert re.fullmatch(r'([-]?relu[+]?\|(zNorm[+]?\|)?)?((w1)?mean|last|concat)', self.WORD_AGG_SCHEME), (
                f'Invlaid WORD_AGG_SCHEME {self.WORD_AGG_SCHEME}')
        if self.OUT_WORD_AGG_SCHEME is not None:
            assert re.fullmatch(r'([-]?relu[+]?\|(zNorm[+]?\|)?)?((w1)?mean|last|concat)', self.OUT_WORD_AGG_SCHEME), (
                f'Invlaid OUT_WORD_AGG_SCHEME {self.OUT_WORD_AGG_SCHEME}')
        # Whether to aggregate segments within a sample and if so, how.
        assert self.SEGMENT_AGG_SCHEME in ['mean', None]
        # Whether to aggregate segments across samples and if so, how.
        assert self.EXAMPLE_AGG_SCHEME in ['mean', None, 'soft_cluster']
        assert self.SIMILARITY_FUNC in ['dot_product', 'cosine_sim', None]  # Concept embedding similarity func
        assert self.NORM in ['L2', 'varNorm', 'zNorm', None]
        # Which transformer hidden layer to pick encodings from. None => top layer
        self.ENCODING_LAYER = self._verify_encoding_layer(self.ENCODING_LAYER)
        self.OUT_ENCODING_LAYER = self._verify_out_encoding_layer(self.OUT_ENCODING_LAYER)

    def _should_truncate(self, seq: List, shift_inp_right: bool) -> bool:
        """Check if the sequence should be truncated"""
        ret_val: bool = (len(seq) > (self.max_length + 1)) if shift_inp_right else (len(seq) > self.max_length)
        if ret_val:
            print(f'Encountered overflow seq. Len = {len(seq)}')
        return ret_val

    # @instrument
    def _normalize(self, T: Tensor, *, dim: int = -1, eps=1e-6, norm=None) -> Tensor:
        """Compute Lp norm i.e., normalize the magnitude of vectors to 1."""
        norm = self.NORM if norm is None else norm
        return normalize(norm, T, dim=dim, eps=eps)

    @staticmethod
    def _soft_cluster(Q: Tensor, M: Tensor) -> Tensor:
        """
        Attention based soft clustering. Merge vectors in M into vectors in Q weighted by similarity.
        :param Q
            query vectors of shape (Sq, D) where D is the vector size
        :param M
            memory vectors of shape (Sm, D)
        :return Tensor[Sq, D]
            attention aggregated (M)
        """
        dot_product = torch.matmul(Q, M.T)  # (Sq, Sm)
        attention_weights = torch.nn.functional.softmax(dot_product, dim=1)  # (Sq, Sm)
        return torch.matmul(attention_weights, M)  # (Sq, D)

    # @instrument
    def tok_encode(self, string: str) -> List[int]:
        return self.module.tokenizer.encode(string, truncation=False, add_special_tokens=(self.is_enc_dec))

    # def _tok_batch_encode_fast(self, strings: List[str]) -> Dict[str, Tensor]:
    #     # WARNING: August 2022: cannot rely on return_length=True because returned lengths are incorrect
    #     if self.module.tokenizer.pad_token_id is None:
    #         reset_pad_token_id = True
    #         self.module.tokenizer.pad_token_id = 0
    #     else:
    #         reset_pad_token_id = False
    #     batch = self.module.tokenizer(text=strings, is_split_into_words=False, padding=True,
    #                            add_special_tokens=False, return_tensors='pt')
    #     if reset_pad_token_id:
    #         self.module.tokenizer.pad_token_id = None
    #     batch['input_ids'] = batch['input_ids'].to(device=self.device)  # (N,T)
    #     batch['attention_mask'] = batch['attention_mask'].to(device=self.device)  # (N,T) boolean
    #     batch['seq_lens'] = batch['attention_mask'].sum(dim=-1)  # (N,)
    #     return batch

    # @instrument
    def _tok_batch_encode(self, strings1: List[str], *, strings2: Optional[List[str]] = None, shift_inp_right: bool = False) -> Dict:
        seq1s = [self.tok_encode(string) for string in strings1]
        seqs, seq_lens = [], []
        if strings2 is not None:
            assert len(strings2) == len(strings1)
            seq2s = []
        for i, seq1 in enumerate(seq1s):
            assert len(seq1) > 0
            if strings2 is not None:
                seq2 = self.tok_encode(strings2[i])
                assert 0 < len(seq2) <= self.max_length
                seq2s.append(seq2)
                seq = seq1 + seq2
            else:
                seq = seq1
            # Shift input right for autoregressive decoding and truncate from left if needed
            if self._should_truncate(seq, shift_inp_right):
                _LOGGER.warning(f'Sequence of length {len(seq)} will be truncated to {self.max_length}')
            seq = seq[-(self.max_length + 1): -1] if shift_inp_right else seq[-self.max_length:]
            seqs.append(seq)
            seq_lens.append(len(seq))

        batch_len = max(seq_lens)
        # pad
        seqs = [seq + [self.module.tokenizer.pad_token_id] * (batch_len - len(seq)) for seq in seqs]

        retdir = {
            'input_ids': torch.tensor(seqs, dtype=torch.long, device=self.device),  # (N,T)
            'seq_lens': torch.tensor(seq_lens, dtype=torch.long, device=self.device),  # (N,)
            'attention_mask': torch.tensor([[1] * len + [0] * (batch_len - len) for len in seq_lens],
                                           dtype=torch.long, device=self.device)  # (N,T)
        }
        if strings2 is not None:
            retdir['seq2s'] = [torch.tensor(seq, dtype=torch.long, device=self.device) for seq in seq2s]  # list of (t,)
        return retdir

    # @instrument
    def _reduce_word_sequences(self,
                               model_output: Dict,
                               model_input: Dict,
                               ENCODING_LAYER=None,
                               WORD_AGG_SCHEME=None) -> Tensor:
        """Extract word vectors from model hidden states and aggregate them into a single concept vector

        :param model_output: Dict.
            HuggingFace model_output dict.
        :param model_input: Dict
            Model input dict
        :return: Tensor, shape = (batch_size, hidden_size)
            Each sequence in the batch reduced to an embedding of size hidden_size
        """
        if ENCODING_LAYER is None:
            ENCODING_LAYER = self.ENCODING_LAYER
        if WORD_AGG_SCHEME is None:
            WORD_AGG_SCHEME = self.WORD_AGG_SCHEME
        assert WORD_AGG_SCHEME is not None
        # encoding_layer = len(model_output['hidden_states']) // 2 if ENCODING_LAYER == 'middle' else -1
        encoding_layer = None
        if isinstance(ENCODING_LAYER, int):
            encoding_layer = ENCODING_LAYER
        elif ENCODING_LAYER == 'middle':
            encoding_layer = len(model_output['hidden_states']) // 2
        elif ENCODING_LAYER is None:
            encoding_layer = -1
        elif ENCODING_LAYER in ['E', 'OE']:
            pass
        else:
            raise ValueError(f'Unsupported value "{ENCODING_LAYER}" for ENCODING_LAYER')

        if encoding_layer is not None:
            concept_seqs = model_output['hidden_states'][encoding_layer]  # (batch, padding-len, hidden_size)
        elif ENCODING_LAYER in ['E', 'OE']:
            concept_seqs = model_input['inputs_embeds']  # (batch, padding-len, hidden_size)
        else:
            raise RuntimeError()

        # concept_seqs = concept_seqs.to(device=self.device)

        if WORD_AGG_SCHEME.startswith('relu|zNorm+|'):
            concept_seqs = normalize('zNorm', self.module.act(concept_seqs)) + concept_seqs
        elif WORD_AGG_SCHEME.startswith('relu+|'):
            concept_seqs = self.module.act(concept_seqs) + concept_seqs
        elif WORD_AGG_SCHEME.startswith('-relu|zNorm+|'):
            concept_seqs = normalize('zNorm', -self.module.act(-concept_seqs)) + concept_seqs
        elif WORD_AGG_SCHEME.startswith('-relu+|'):
            concept_seqs = concept_seqs - self.module.act(-concept_seqs)
        elif WORD_AGG_SCHEME.startswith('relu|zNorm|'):
            concept_seqs = normalize('zNorm', self.module.act(concept_seqs))
        elif WORD_AGG_SCHEME.startswith('relu|'):
            concept_seqs = self.module.act(concept_seqs)
        elif WORD_AGG_SCHEME.startswith('-relu|zNorm|'):
            concept_seqs = normalize('zNorm', -self.module.act(-concept_seqs))
        elif WORD_AGG_SCHEME.startswith('-relu|'):
            concept_seqs = -self.module.act(-concept_seqs)
        elif 'relu' in WORD_AGG_SCHEME:
            raise ValueError(f'Unsupported WORD_AGG_SCHEME: {WORD_AGG_SCHEME}')
        elif WORD_AGG_SCHEME.startswith('zNorm|'):
            concept_seqs = normalize('zNorm', concept_seqs)
        elif 'zNorm' in WORD_AGG_SCHEME:
            raise ValueError(f'Unsupported WORD_AGG_SCHEME: {WORD_AGG_SCHEME}')

        do_normalize: bool = True
        if WORD_AGG_SCHEME.endswith('concat'):
            # assert self.task.TASK_TYPE == 'gen'
            batch_size = concept_seqs.shape[0]
            seq_lens = model_input['seq_lens']
            aggregated_vectors = torch.stack([vector for row, seq_len in enumerate(seq_lens)
                                             for vector in concept_seqs[row, :seq_len]])
        elif WORD_AGG_SCHEME.endswith('last'):
            aggregated_vectors = torch.stack([concept_seqs[row, seq_len - 1, :]
                                              for row, seq_len in enumerate(model_input['seq_lens'])])  # (batch, hidden_size)
            do_normalize = False
        elif WORD_AGG_SCHEME.endswith('w1mean'):
            padded_len = model_input['attention_mask'].shape[1]
            agg_weights = (self.module.agg_weights[0, :padded_len] * model_input['attention_mask']).unsqueeze(-1)
            aggregated_vectors = ((concept_seqs * agg_weights.to(device=concept_seqs.device)).sum(dim=1) / agg_weights.sum(dim=1))
        elif WORD_AGG_SCHEME.endswith('mean'):
            aggregated_vectors = ((concept_seqs * model_input['attention_mask'].unsqueeze(-1).to(device=concept_seqs.device)).sum(dim=1)
                                  / model_input['seq_lens'].unsqueeze(-1).to(device=concept_seqs.device))
        else:
            raise NotImplementedError

        if do_normalize:
            aggregated_vectors = self._normalize(aggregated_vectors)  # (batch, hidden_size)
        return aggregated_vectors  # (batch, hidden_size)

    # (#strings, hidden_size)
    # @instrument
    def _embed_strings(self, strings: List[str], pos: int, sequential_pos: bool = True, is_output: bool = False) -> Tuple[Tensor, int]:
        model_input = self._tok_batch_encode(strings)  # (#strings, padded_len)
        ENCODING_LAYER = self.ENCODING_LAYER if not is_output else self.OUT_ENCODING_LAYER
        token_embeddings = self.module.input_embeddings if ENCODING_LAYER != 'OE' else self.module.out_token_embeddings
        model_input['inputs_embeds'] = token_embeddings(model_input['input_ids'].to(device=token_embeddings.weight.device)
                                                        ).to(device=self.device)  # (#strings, padded_len, hidden_size)
        if self.ADD_POS:
            device = model_input['inputs_embeds'].device
            input_shape = model_input['inputs_ids'].shape
            pos_ids = torch.new_zeros(input_shape[0], dtype=model_input['seq_lens'].dtype) + pos  # (#strings,)
            if sequential_pos:
                pos_ids = pos_ids + \
                    torch.cat((torch.new_zeros(1, dtype=model_input['seq_lens'].dtype), model_input['seq_lens'][:-1]))
            pos_ids = pos_ids.to(device=device)  # (#strings,)
            pos_e = self.module.wpe(pos_ids)  # (#strings, hidden_size)
            if pos == 0:
                pos_e[0] = torch.where(pos_ids.unsqueeze(-1).expand(pos_e.shape) == 0, 0, pos_e)
            pos_e = pos_e.unsqueeze(1)  # (#strings, 1, hidden_size)
            model_input['inputs_embeds'] = model_input['inputs_embeds'] + pos_e
            if sequential_pos:
                pos = pos + model_input['seq_lens'].sum().item()
        if ENCODING_LAYER not in ['E', 'OE']:
            model_output = self.module.encoder(  # input_ids=model_input['input_ids'],
                inputs_embeds=model_input['inputs_embeds'],
                attention_mask=model_input['attention_mask'],
                output_hidden_states=True,
                return_dict=True)
            # self.module.lm.encoder(input_ids=input_ids, attention_mask=input_mask)['last_hidden_state']
        else:
            model_output = None
            # model_input['inputs_embeds'] = self.module.input_embeddings(model_input['input_ids'])

        WORD_AGG_SCHEME = self.OUT_WORD_AGG_SCHEME if is_output else self.WORD_AGG_SCHEME
        return self._reduce_word_sequences(model_output,
                                           model_input,
                                           ENCODING_LAYER,
                                           WORD_AGG_SCHEME), pos  # (#strings, hidden_size) or (concatenated-strings len, hidden_size)

    # @instrument
    def _embed_segments(self, sample: SegmentedSample, pos: int) -> Tuple[Tensor, int]:
        """Embed segments of an example"""
        assert 'segments' in sample
        segment_embeddings, pos = self._embed_strings(sample['segments'], pos=pos if self.ADD_POS else None)
        if self.SEGMENT_AGG_SCHEME == 'mean':
            _context_embedding = segment_embeddings.mean(dim=0)  # (hidden_size,)
            context_embeddings = self._normalize(_context_embedding).unsqueeze(0)  # (1, hidden_size,)
        elif self.SEGMENT_AGG_SCHEME is None:
            context_embeddings = segment_embeddings  # (#segments, hidden_size)
        else:
            raise NotImplementedError
        return context_embeddings, pos

    # @instrument
    def _embed_choices(self, sample: SegmentedSample, pos: Optional[int]) -> Tensor:
        """Embed choices of an example"""
        assert 'choices' in sample
        choices_embeddings, pos = self._embed_strings(
            sample['choices'], pos, sequential_pos=False, is_output=True)  # (#choices, hidden_size)
        return choices_embeddings

    # @instrument
    def _embed_context(self, examples: List[SegmentedSample]) -> Tuple[Tensor, Optional[int]]:
        """Embed a context (represented as a list of SegmentedSamples) into a single embedding vector"""
        example_embeddings, pos = [], 0 if self.ADD_POS else None
        for example in examples:
            _example_embeddings, pos = self._embed_segments(example, pos)
            example_embeddings.append(_example_embeddings)
        example_embeddings = torch.cat(example_embeddings, dim=0)  # (#chunks, hidden_size)
        if len(example_embeddings) > 1:
            if self.EXAMPLE_AGG_SCHEME == 'mean':
                context_embeddings = example_embeddings.mean(dim=0)  # (hidden_size,)
                context_embeddings = self._normalize(context_embeddings).unsqueeze(0)  # (1, hidden_size)
            elif self.EXAMPLE_AGG_SCHEME in [None, 'soft_cluster']:
                context_embeddings = example_embeddings  # (#chunks, hidden_size)
            else:
                raise NotImplementedError
        else:
            context_embeddings = example_embeddings  # (#chunks, hidden_size)
        return context_embeddings, pos  # (#chunks, hidden_size)

    def distributed_encoding_similarity(self, requests_args) -> List[Tensor]:
        """Compute similarity of choices.

        :param
            requests_args: A list of pairs (context, doc)
            context: List of context SegmentedSamples: Optional[description] + [few-shot samples] + [doc]
            doc: The query doc SegmentedSample
        :return:
            A list of results, [Tensor], one per request (doc)
            Tensor: A list of similarity scores of a request (doc), one per choice [score,]
        """
        # Eschewing batching in favor of simplicity for now
        results = []
        self.module.no_grad()
        with torch.no_grad():
            for doc_id, (context, doc) in tqdm(enumerate(requests_args), total=len(requests_args)):
                assert doc.task.ENCODING_SCHEME != 'cross_encoding', 'cross_encoding scheme is not support with this model_type'
                assert self.WORD_AGG_SCHEME is not None and 'concat' not in self.WORD_AGG_SCHEME
                context_embeddings, pos = self._embed_context(context)  # (#chunks, hidden_size)
                doc = doc.copy()
                del doc['segments']
                choice_embeddings = self._embed_choices(doc, pos)  # (#choices, hidden_size)
                if self.EXAMPLE_AGG_SCHEME == 'soft_cluster':
                    context_embeddings = self._soft_cluster(
                        choice_embeddings, context_embeddings)  # (#choices, hidden_size)
                if self.SIMILARITY_FUNC == 'cosine_sim':
                    # L2 normalize vectors for cosine-sim
                    context_embeddings = torch.nn.functional.normalize(context_embeddings, p=2, dim=1)
                    choice_embeddings = torch.nn.functional.normalize(choice_embeddings, p=2, dim=1)
                if self.EXAMPLE_AGG_SCHEME == 'soft_cluster':
                    scores = torch.sum(choice_embeddings * context_embeddings, dim=1)  # (#choices,)
                elif self.SIMILARITY_FUNC in ['dot_product', 'cosine_sim']:
                    scores = torch.matmul(choice_embeddings, context_embeddings.T)  # (#choices, #chunks)
                    scores = scores.amax(dim=1)  # (#choices,)
                else:
                    raise NotImplementedError
                scores = torch.nn.functional.softmax(scores, dim=0).cpu().numpy()
                results.append({'scores': scores})
                # if (doc_id % 100 == 0):
                #     torch.cuda.empty_cache()
        return results


class DistEncGenMixin(DistEncSimMixin):
    def __init__(self, *args, DECODING_SCHEME: Optional[str] = None, PARALLELIZE: Optional[str] = None, device, **kwargs) -> None:
        if DECODING_SCHEME != 'parameterless_attention':
            super().__init__(*args, PARALLELIZE=PARALLELIZE, device=device, del_lm=False, **kwargs)
            self.DECODING_SCHEME = DECODING_SCHEME if DECODING_SCHEME != 'None' else None
            self.module.decoder = self.module.lm
            self.module.no_grad()
        else:
            assert not str_to_bool(PARALLELIZE)
            # Make HFLM load the model in RAM first because we're going to throw it away later and PyTorch won't release GPU memory.
            # We'll switch it back to device later.
            super().__init__(*args, PARALLELIZE=PARALLELIZE, device='cpu', del_lm=False, **kwargs)
            # Following init is to be done after HFLM and DistEncSimMixin havd initialized
            self.DECODING_SCHEME = DECODING_SCHEME
            self.module.decoder = ParameterlessAttentionDecoder(self.module.lm)
            del self.module.lm
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            self.module.to(device=device)
            self.module.no_grad()
        # torch.cuda.empty_cache()

    def verify_config(self):
        super().verify_config()
        assert self.EXAMPLE_AGG_SCHEME in ['mean', None]
        assert self.SIMILARITY_FUNC is None
        assert self.DECODING_SCHEME in ['parameterless_attention', None]

    # @instrument('_embed_context2')
    def _embed_context(self, examples: List[SegmentedSample]) -> Tuple[Tensor, int]:
        """Embed a context (represented as a list of SegmentedSamples) into a sequence of embedding vectors"""
        example_embeddings, pos = [], 0 if self.ADD_POS else None
        for example in examples:
            _example_embeddings, pos = self._embed_segments(example, pos)
            example_embeddings.append(_example_embeddings)
        # example_embeddings = [self._embed_example_segments(example) for example in examples]
        example_embeddings = torch.cat(example_embeddings, dim=0)  # (seq_len, hidden_size)
        if len(example_embeddings) > 1 and self.EXAMPLE_AGG_SCHEME == 'mean':
            context_embeddings = example_embeddings.mean(dim=0)
            context_embeddings = self._normalize(context_embeddings).unsqueeze(0)  # (1, hidden_size)
        else:
            context_embeddings = example_embeddings
        return context_embeddings, pos  # (seq_len, hidden_size,)

    # @instrument
    def _input_embed_words(self, context_embeddings: Tensor, strings: List[str], *, pos: int, shift_inp_right: bool) -> Dict:
        """
        Encode and word-embed strings. Return embedding-sequence prepended with context embedding vectors.
        """
        # assert len(context_embeddings) == 1
        seqs, seq2s, seq_lens = [], [], []
        for i, string in enumerate(strings):
            tok_seq = torch.tensor(self.tok_encode(string), dtype=torch.long, device=self.device)  # (seq_len,)
            tok_emb = self.module.input_embeddings(tok_seq)  # (seq_len, hidden_size)
            if self.ADD_POS and pos != 0:
                tok_emb = tok_emb + self.module.wpe(tok_seq.new_full(tok_seq.shape, pos))
            assert 0 < len(tok_emb) < self.max_length
            seq2s.append(tok_seq)
            seq = torch.cat([context_embeddings, tok_emb])
            # Shift input right for autoregressive decoding and truncate from left if needed
            # Shift-right implies that the last token is dropped. But no <BOS> token is prefixed.
            if self._should_truncate(seq, shift_inp_right):
                _LOGGER.warning(f'Sequence of length {len(seq)} will be truncated to {self.max_length}')
            seq = seq[-(self.max_length + 1): -1] if shift_inp_right else seq[-self.max_length:]
            seqs.append(seq)
            seq_lens.append(len(seq))

        batch_len = max(seq_lens)
        # pad
        pad_embedding = self.module.input_embeddings(torch.tensor(
            [self.module.tokenizer.pad_token_id], dtype=torch.long, device=self.device)).squeeze(0)
        seqs = [torch.cat([seq, pad_embedding.repeat(batch_len - len(seq), 1)]) if batch_len > len(seq) else seq
                for seq in seqs]

        retdir = {
            'inputs_embeds': torch.stack(seqs),  # (N,T,D)
            'seq_lens': torch.tensor(seq_lens, dtype=torch.long, device=self.device),  # (N,)
            'attention_mask': torch.tensor([[1] * len + [0] * (batch_len - len) for len in seq_lens],
                                           dtype=torch.long, device=self.device),  # (N,T)
            'seq2s': seq2s  # list of (t,)
        }
        return retdir

    # @do_profile
    def distributed_encoding_generation(self, requests_args, prof: Optional[profile] = None) -> List[Tensor]:
        """Generate text by encoding context distributively.

        :param requests_args: list
            A list of pairs (context, doc)
            context: List of context SegmentedSamples: Optional[description] + [few-shot samples] + [doc]
            doc: The query doc SegmentedSample
        :return:
            A list of results, [Tensor], one per request (doc)
            Tensor: A list of similarity scores of a request (doc), one per choice [score,]
        """
        if self.is_enc_dec:
            return self.distributed_encdec_generation(requests_args)
        # Eschewing batching in favor of simplicity for now
        # self.verify_config()
        results = []
        self.module.no_grad()
        with torch.no_grad():
            for doc_id, (context, doc) in tqdm(enumerate(requests_args), total=len(requests_args)):
                if prof is not None:
                    prof.step()
                logprobs: List[Tensor] = []
                seq2s: List[Tensor] = []
                seq_lens: List[Tensor] = []
                if doc.task.ENCODING_SCHEME == 'cross_encoding':
                    assert len(context) == 1
                    assert len(context[0]['segments']) == 1
                    assert self.WORD_AGG_SCHEME is None
                    assert self.OUT_WORD_AGG_SCHEME is None
                    assert self.ENCODING_LAYER is None
                    assert self.OUT_ENCODING_LAYER is None
                    # TODO: If ctx == '' context_enc = [eot_token_id]. Line 193 in base.py
                    ctx = context[0]['segments'][0]
                    context_embeddings = None
                else:
                    context_embeddings, pos = self._embed_context(context)  # (seq_len, hidden_size)
                # Run each choice separately to avoid OOM with large models
                for choice in doc['choices']:
                    input_ids = inputs_embeds = None
                    if doc.task.ENCODING_SCHEME == 'cross_encoding':
                        model_input = self._tok_batch_encode([ctx], strings2=[choice], shift_inp_right=True)
                        input_ids, inputs_embeds = model_input['input_ids'], None
                    else:
                        model_input = self._input_embed_words(
                            context_embeddings, [choice], pos=pos, shift_inp_right=True)
                        input_ids, inputs_embeds = None, model_input['inputs_embeds']
                    model_output = self.module.decoder(input_ids=input_ids, inputs_embeds=inputs_embeds,
                                                       attention_mask=model_input['attention_mask'],
                                                       output_hidden_states=True,
                                                       return_dict=True)
                    logprobs.append(
                        torch.nn.functional.log_softmax(model_output.logits, dim=-1).squeeze(0)  # (seq_len, vocab)
                    )
                    seq2s.extend(model_input['seq2s'])
                    seq_lens.extend(model_input['seq_lens'])

                score_list, is_exact_match = [], []
                for i, choice_seq in enumerate(seq2s):
                    seq_len = seq_lens[i].item()
                    lp_slice = logprobs[i][seq_len - len(choice_seq):seq_len]  # (choice_len, vocab)
                    is_em = (choice_seq == lp_slice.argmax(dim=-1)).all().item()
                    is_exact_match.append(bool(is_em))
                    choice_seq = choice_seq.unsqueeze(-1)
                    choice_lprobs = torch.gather(lp_slice, -1, choice_seq).squeeze()  # (choice_len,)
                    score_list.append(choice_lprobs.sum().item())
                results.append({'scores': np.array(score_list), 'is_exact_match': is_exact_match})
                # if (doc_id % 100 == 0):
                #     torch.cuda.empty_cache()

        return results

    def distributed_encdec_generation(self, requests_args) -> List[Tensor]:
        """Generate text by encoding context distributively.

        :param requests_args: list
            A list of pairs (context, doc)
            context: List of context SegmentedSamples: Optional[description] + [few-shot samples] + [doc]
            doc: The query doc SegmentedSample
        :return:
            A list of results, [Tensor], one per request (doc)
            Tensor: A list of similarity scores of a request (doc), one per choice [score,]
        """
        # Eschewing batching in favor of simplicity for now
        # self.verify_config()
        assert not self.ADD_POS
        results = []
        self.module.no_grad()
        with torch.no_grad():
            for doc_id, (context, doc) in tqdm(enumerate(requests_args), total=len(requests_args)):
                logprobs: List[Tensor] = []
                seq2s: List[Tensor] = []
                seq_lens: List[Tensor] = []
                if doc.task.ENCODING_SCHEME == 'cross_encoding':
                    assert len(context) == 1
                    assert len(context[0]['segments']) == 1
                    assert self.WORD_AGG_SCHEME is None
                    assert self.OUT_WORD_AGG_SCHEME is None
                    assert self.ENCODING_LAYER is None
                    assert self.OUT_ENCODING_LAYER is None
                    # TODO: If ctx == '' context_enc = [eot_token_id]. Line 193 in base.py
                    ctx = context[0]['segments'][0]
                    ctx_ids = self.module.tokenizer(ctx, return_tensors='pt',
                                                    add_special_tokens=True).input_ids.to(device=self.device)
                else:
                    assert not self.ADD_POS
                    context_embeddings, pos = self._embed_context(context)  # (seq_len, hidden_size)
                # Run each choice separately to avoid OOM with large models
                for choice in doc['choices']:
                    labels = self.module.tokenizer(choice, return_tensors='pt',
                                                   add_special_tokens=False  # Omit the trailing </s> token
                                                   ).input_ids.to(device=self.device)
                    if doc.task.ENCODING_SCHEME == 'cross_encoding':
                        model_output = self.module.decoder(input_ids=ctx_ids,
                                                           labels=labels,
                                                           output_hidden_states=True,
                                                           return_dict=True)
                    else:
                        # self._t5(encoder_outputs=(encoded_input,),
                        #            attention_mask=attention_mask,
                        #            labels=batch['labels'],
                        #            decoder_attention_mask=batch['decoder_attention_mask'])
                        model_output = self.module.decoder(encoder_outputs=(context_embeddings.unsqueeze(0),),
                                                           labels=labels,
                                                           output_hidden_states=True,
                                                           return_dict=True)
                    logprobs.append(
                        torch.nn.functional.log_softmax(model_output.logits, dim=-1).squeeze(0)  # (seq_len, vocab)
                    )
                    seq2s.append(labels.squeeze(0))
                    seq_lens.append(labels.shape[-1])

                score_list, is_exact_match = [], []
                for i, choice_seq in enumerate(seq2s):
                    # seq_len = seq_lens[i].item()
                    lp_slice = logprobs[i]  # (choice_len, vocab)
                    choice_seq = choice_seq.to(device=lp_slice.device)
                    is_em = (choice_seq == lp_slice.argmax(dim=-1)).all().item()
                    _LOGGER.debug(f"choice = {doc['choices'][i]}\n"
                                  f"pred = {self.module.tokenizer.decode(lp_slice.argmax(dim=-1))}")
                    is_exact_match.append(bool(is_em))
                    choice_seq = choice_seq.unsqueeze(-1)
                    choice_lprobs = torch.gather(lp_slice, -1, choice_seq).squeeze()  # (choice_len,)
                    score_list.append(choice_lprobs.sum().item())
                results.append({'scores': np.array(score_list), 'is_exact_match': is_exact_match})
                # if (doc_id % 100 == 0):
                #     torch.cuda.empty_cache()

        return results
