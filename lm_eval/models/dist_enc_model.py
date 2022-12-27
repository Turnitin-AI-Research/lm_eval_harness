"""Utilities for distributed encoding"""
from typing import Dict, List, Optional
import re
import logging
from tqdm import tqdm
import torch
from torch import Tensor
from lm_eval.tasks.dist_enc_tasks import SegmentedSample
from transformers.activations import ACT2FN

_LOGGER = logging.getLogger(__name__)


class DistEncSimMixin:
    # WORD_AGG_SCHEME: str = None
    # SEGMENT_AGG_SCHEME: str = 'mean'
    # EXAMPLE_AGG_SCHEME: str = 'mean'
    # SIMILARITY_FUNC: str = None
    # NORM: str = None

    def __init__(self,
                 *args,
                 WORD_AGG_SCHEME: Optional[str] = None,
                 SEGMENT_AGG_SCHEME: Optional[str] = 'mean',
                 EXAMPLE_AGG_SCHEME: Optional[str] = 'mean',
                 SIMILARITY_FUNC: Optional[str] = None,
                 NORM: Optional[str] = None,
                 ENCODING_LAYER: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.WORD_AGG_SCHEME: str = WORD_AGG_SCHEME if WORD_AGG_SCHEME != 'None' else None
        self.SEGMENT_AGG_SCHEME: str = SEGMENT_AGG_SCHEME if SEGMENT_AGG_SCHEME != 'None' else None
        self.EXAMPLE_AGG_SCHEME: str = EXAMPLE_AGG_SCHEME if EXAMPLE_AGG_SCHEME != 'None' else None
        self.SIMILARITY_FUNC: str = SIMILARITY_FUNC if SIMILARITY_FUNC != 'None' else None
        self.NORM: str = NORM if NORM != 'None' else None
        self.ENCODING_LAYER: str = ENCODING_LAYER if ENCODING_LAYER != 'None' else None

    def verify_config(self):
        assert self.WORD_AGG_SCHEME in [None, 'last', 'relu|last', 'relu+|last', '-relu+|last', 'mean', 'relu|mean', 'relu+|mean', '-relu+|mean']
        assert self.SEGMENT_AGG_SCHEME in ['mean', None]  # Whether to aggregate segments within a sample and if so, how.
        assert self.EXAMPLE_AGG_SCHEME in ['mean', None, 'soft_cluster']  # Whether to aggregate segments across samples and if so, how.
        assert self.SIMILARITY_FUNC in ['dot_product', 'cosine_sim', None]  # Concept embedding similarity func
        assert self.NORM in ['L2', 'layer', None]
        # Which transformer hidden layer to pick encodings from. None => top layer
        assert self.ENCODING_LAYER in ['middle', None, 'E'] or re.fullmatch(r'\d+', self.ENCODING_LAYER)
        if (self.ENCODING_LAYER is not None) and (match := re.fullmatch(r'\d+', self.ENCODING_LAYER)):
            self.ENCODING_LAYER = int(self.ENCODING_LAYER)
        self.act = ACT2FN[self.gpt2.config.activation_function]

    def _should_truncate(self, seq: List, shift_inp_right: bool) -> bool:
        """Check if the sequence should be truncated"""
        if shift_inp_right:
            return len(seq) > (self.max_length + 1)
        else:
            return len(seq) > self.max_length

    def _normalize(self, T: Tensor, *, dim: int = -1) -> Tensor:
        """Compute Lp norm i.e., normalize the magnitude of vectors to 1."""
        if self.NORM is None:
            return T
        elif self.NORM == 'L2':
            return torch.nn.functional.normalize(T, p=2, dim=dim)
        elif self.NORM == 'layer':
            return T / T.std(dim=dim, keepdim=True)
        else:
            raise NotImplementedError

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
        attention_weights = torch.nn.functional.softmax(dot_product, dim=0)  # (Sq, Sm)
        return torch.matmul(attention_weights, M)  # (Sq, D)

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    # def _tok_batch_encode_fast(self, strings: List[str]) -> Dict[str, Tensor]:
    #     # WARNING: August 2022: cannot rely on return_length=True because returned lengths are incorrect
    #     if self.tokenizer.pad_token_id is None:
    #         reset_pad_token_id = True
    #         self.tokenizer.pad_token_id = 0
    #     else:
    #         reset_pad_token_id = False
    #     batch = self.tokenizer(text=strings, is_split_into_words=False, padding=True,
    #                            add_special_tokens=False, return_tensors='pt')
    #     if reset_pad_token_id:
    #         self.tokenizer.pad_token_id = None
    #     batch['input_ids'] = batch['input_ids'].to(device=self.device)  # (N,T)
    #     batch['attention_mask'] = batch['attention_mask'].to(device=self.device)  # (N,T) boolean
    #     batch['seq_lens'] = batch['attention_mask'].sum(dim=-1)  # (N,)
    #     return batch

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
        seqs = [seq + [0] * (batch_len - len(seq)) for seq in seqs]

        retdir = {
            'input_ids': torch.tensor(seqs, dtype=torch.long, device=self.device),  # (N,T)
            'seq_lens': torch.tensor(seq_lens, dtype=torch.long, device=self.device),  # (N,)
            'attention_mask': torch.tensor([[1] * len + [0] * (batch_len - len) for len in seq_lens],
                                           dtype=torch.long, device=self.device)  # (N,T)
        }
        if strings2 is not None:
            retdir['seq2s'] = [torch.tensor(seq, dtype=torch.long, device=self.device) for seq in seq2s]  # list of (t,)
        return retdir

    def _reduce_word_sequences(self, model_output: Dict, model_input: Dict) -> Tensor:
        """Extract word vectors from model hidden states and aggregate them into a single concept vector

        :param model_output: Dict.
            HuggingFace model_output dict.
        :param model_input: Dict
            Model input dict
        :return: Tensor, shape = (batch_size, hidden_size)
            Each sequence in the batch reduced to an embedding of size hidden_size
        """
        # encoding_layer = len(model_output['hidden_states']) // 2 if self.ENCODING_LAYER == 'middle' else -1
        encoding_layer = None
        if isinstance(self.ENCODING_LAYER, int):
            encoding_layer = self.ENCODING_LAYER
        elif self.ENCODING_LAYER == 'middle':
            encoding_layer = len(model_output['hidden_states']) // 2
        elif self.ENCODING_LAYER is None:
            encoding_layer = -1
        elif self.ENCODING_LAYER == 'E':
            pass
        else:
            raise ValueError(f'Unsupported value "{self.ENCODING_LAYER}" for ENCODING_LAYER')
        if encoding_layer is not None:
            concept_seqs = model_output['hidden_states'][encoding_layer]  # (batch, padding-len, hidden_size)
        elif self.ENCODING_LAYER == 'E':
            concept_seqs = model_input['inputs_embeds']  # (batch, padding-len, hidden_size)
        else:
            raise RuntimeError()

        if self.WORD_AGG_SCHEME.startswith('relu+|'):
            concept_seqs = self.act(concept_seqs) + concept_seqs
        elif self.WORD_AGG_SCHEME.startswith('-relu+|'):
            concept_seqs = concept_seqs - self.act(concept_seqs)
        elif self.WORD_AGG_SCHEME.startswith('relu|'):
            concept_seqs = self.act(concept_seqs)

        if self.WORD_AGG_SCHEME.endswith('last'):
            aggregated_vectors = torch.stack([concept_seqs[row, seq_len - 1, :]
                                              for row, seq_len in enumerate(model_input['seq_lens'])])  # (batch, hidden_size)
        elif self.WORD_AGG_SCHEME.endswith('mean'):
            aggregated_vectors = ((concept_seqs * model_input['attention_mask'].unsqueeze(-1)).sum(dim=1)
                                  / model_input['seq_lens'].unsqueeze(-1))
        else:
            raise NotImplementedError
        return self._normalize(aggregated_vectors)  # (batch, hidden_size)

    def _embed_strings(self, strings: List[str]) -> Tensor:  # (#strings, hidden_size)
        model_input = self._tok_batch_encode(strings)  # (#strings, padded_len)
        if self.ENCODING_LAYER != 'E':
            model_output = self.gpt2(input_ids=model_input['input_ids'],
                                     attention_mask=model_input['attention_mask'],
                                     output_hidden_states=True,
                                     return_dict=True)
        else:
            model_output = None
            model_input['inputs_embeds'] = self.gpt2.get_input_embeddings()(model_input['input_ids'])
        return self._reduce_word_sequences(model_output, model_input)  # (#strings, hidden_size)

    def _embed_sample(self, sample: SegmentedSample) -> SegmentedSample:
        """Embed segments if present, into a single embedding.
        If choices are present, then embed each individually."""
        if 'segments' in sample:
            segment_embeddings = self._embed_strings(sample['segments'])
            if self.SEGMENT_AGG_SCHEME == 'mean':
                _context_embedding = segment_embeddings.mean(dim=0)  # (hidden_size,)
                sample['context_embeddings'] = self._normalize(_context_embedding).unsqueeze(0)  # (1, hidden_size,)
            elif self.SEGMENT_AGG_SCHEME is None:
                sample['context_embeddings'] = segment_embeddings  # (#segments, hidden_size)
            else:
                raise NotImplementedError
        if 'choices' in sample:
            sample['choices_embeddings'] = self._embed_strings(sample['choices'])  # (#choices, hidden_size)

        return sample

    # def _embed_sample(self, sample: SegmentedSample) -> SegmentedSample:
    #     """Embed segments if present, into a single embedding.
    #     If choices are present, then embed each individually."""
    #     if 'segments' in sample:
    #         model_input = self._tok_batch_encode(sample['segments'])  # (#segments, padded_len)
    #         if self.ENCODING_LAYER != 'E':
    #             model_output = self.gpt2(input_ids=model_input['input_ids'],
    #                                      attention_mask=model_input['attention_mask'],
    #                                      output_hidden_states=True,
    #                                      return_dict=True)
    #         else:
    #             model_output = None
    #             model_input['inputs_embeds'] = self.gpt2.get_input_embeddings()(model_input['input_ids'])
    #         segment_embeddings = self._reduce_word_sequences(
    #             model_output, model_input)  # (#segments, hidden_size)
    #         # segment_embeddings = self._normalize(segment_embeddings)
    #         if self.SEGMENT_AGG_SCHEME == 'mean':
    #             _context_embedding = segment_embeddings.mean(dim=0)  # (hidden_size,)
    #             sample['context_embeddings'] = self._normalize(_context_embedding).unsqueeze(0)  # (1, hidden_size,)
    #         elif self.SEGMENT_AGG_SCHEME is None:
    #             sample['context_embeddings'] = segment_embeddings  # (#segments, hidden_size)
    #         else:
    #             raise NotImplementedError
    #     if 'choices' in sample:
    #         model_input = self._tok_batch_encode(sample['choices'])  # (#choices, padded_len)
    #         if self.ENCODING_LAYER != 'E':
    #             model_output = self.gpt2(input_ids=model_input['input_ids'],
    #                                      attention_mask=model_input['attention_mask'],
    #                                      output_hidden_states=True,
    #                                      return_dict=True)
    #         else:
    #             model_output = None
    #             model_input['inputs_embeds'] = self.gpt2.get_input_embeddings()(model_input['input_ids'])
    #         sample['choices_embeddings'] = self._reduce_word_sequences(
    #             model_output, model_input)  # (#choices, hidden_size)
    #         # sample['choices_embeddings'] = self._normalize(sample['choices_embeddings'])  # (#choices, hidden_size)

    #     return sample

    def _embed_context(self, examples: List[SegmentedSample]) -> Tensor:
        """Embed a context (represented as a list of SegmentedSamples) into a single embedding vector"""
        example_embeddings = [self._embed_sample(example)['context_embeddings'] for example in examples]
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
        return context_embeddings  # (#chunks, hidden_size)

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
        self.gpt2.eval()
        with torch.no_grad():
            for context, doc in tqdm(requests_args):
                assert doc.task.ENCODING_SCHEME != 'cross_encoding', 'cross_encoding scheme is not support with this model_type'
                context_embeddings = self._embed_context(context)  # (#chunks, hidden_size)
                doc = doc.copy()
                del doc['segments']
                choice_embeddings = self._embed_sample(doc)['choices_embeddings']  # (#choices, hidden_size)
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
                scores = torch.nn.functional.softmax(scores)
                results.append({'scores': scores})
        return results


class DistEncGenMixin(DistEncSimMixin):
    # WORD_AGG_SCHEME: Optional[str] = None
    # SEGMENT_AGG_SCHEME: Optional[str] = None
    # EXAMPLE_AGG_SCHEME: Optional[str] = None
    # SIMILARITY_FUNC: Optional[str] = None
    # NORM: Optional[str] = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def verify_config(self):
        assert self.WORD_AGG_SCHEME in [None, 'last', 'relu|last', 'relu+|last', '-relu+|last', 'mean', 'relu|mean', 'relu+|mean', '-relu+|mean']
        assert self.SEGMENT_AGG_SCHEME in ['mean', None]
        assert self.EXAMPLE_AGG_SCHEME in ['mean', None]
        assert self.SIMILARITY_FUNC in ['dot_product', 'cosine_sim', None]
        assert self.NORM in ['L2', 'layer', None]

    def _embed_sample(self, sample: SegmentedSample) -> SegmentedSample:
        raise RuntimeError('This method should not have been called')

    def _embed_segments(self, sample: SegmentedSample) -> Tensor:
        """Embed segments into a seqeunce of embeddings."""
        assert 'segments' in sample
        model_input = self._tok_batch_encode(sample['segments'])  # (#segments, padded_len)
        if self.ENCODING_LAYER != 'E':
            model_output = self.gpt2(input_ids=model_input['input_ids'],
                                     attention_mask=model_input['attention_mask'],
                                     output_hidden_states=True,
                                     return_dict=True)
        else:
            model_output = None
            model_input['inputs_embeds'] = self.gpt2.get_input_embeddings()(model_input['input_ids'])
        segment_embeddings = self._reduce_word_sequences(
            model_output, model_input)  # (#segments, hidden_size)
        # segment_embeddings = self._normalize(segment_embeddings)
        if self.SEGMENT_AGG_SCHEME == 'mean':
            context_embeddings = segment_embeddings.mean(dim=0).unsqueeze(0)  # (1, hidden_size,)
            context_embeddings = self._normalize(context_embeddings)  # (1, hidden_size,)
        else:
            context_embeddings = segment_embeddings  # (#segments, hidden_size)

        return context_embeddings  # (#segments, hidden_size)

    def _embed_context(self, examples: List[SegmentedSample]) -> Tensor:
        """Embed a context (represented as a list of SegmentedSamples) into a sequence of embedding vectors"""
        example_embeddings = [self._embed_segments(example) for example in examples]
        example_embeddings = torch.cat(example_embeddings, dim=0)  # (seq_len, hidden_size)
        if len(example_embeddings) > 1 and self.EXAMPLE_AGG_SCHEME == 'mean':
            context_embeddings = example_embeddings.mean(dim=0)
            context_embeddings = self._normalize(context_embeddings).unsqueeze(0)  # (1, hidden_size)
        else:
            context_embeddings = example_embeddings
        return context_embeddings  # (seq_len, hidden_size,)

    def _input_embed_words(self, context_embeddings: Tensor, strings: List[str], shift_inp_right: bool) -> Dict:
        """
        Encode and word-embed strings. Return embedding-sequence prepended with context embedding vectors.
        """
        # assert len(context_embeddings) == 1
        seqs, seq2s, seq_lens = [], [], []
        for i, string in enumerate(strings):
            tok_seq = torch.tensor(self.tok_encode(string), dtype=torch.long, device=self.device)  # (seq_len,)
            tok_emb = self.gpt2.get_input_embeddings()(tok_seq)  # (seq_len, hidden_size)
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
        pad_embedding = self.gpt2.get_input_embeddings()(torch.tensor(
            [0], dtype=torch.long, device=self.device)).squeeze(0)
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

    def distributed_encoding_generation(self, requests_args) -> List[Tensor]:
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
        self.verify_config()
        results = []
        self.gpt2.eval()
        with torch.no_grad():
            for context, doc in tqdm(requests_args):
                logprobs: List[Tensor] = []
                seq2s: List[Tensor] = []
                seq_lens: List[Tensor] = []
                if doc.task.ENCODING_SCHEME == 'cross_encoding':
                    assert len(context) == 1
                    assert len(context[0]['segments']) == 1
                    ctx = context[0]['segments'][0]
                    choices = context[0]['choices']
                    # TODO: If ctx == '' context_enc = [eot_token_id]
                    # Run each choice separately to avoid OOM with large models
                    for choice in choices:
                        model_input = self._tok_batch_encode([ctx], strings2=[choice], shift_inp_right=True)
                        model_output = self.gpt2(input_ids=model_input['input_ids'],
                                                 attention_mask=model_input['attention_mask'],
                                                 output_hidden_states=True,
                                                 return_dict=True)
                        logprobs.append(
                            torch.nn.functional.log_softmax(model_output.logits, dim=-1).squeeze(0)  # (seq_len, vocab)
                        )
                        seq2s.extend(model_input['seq2s'])
                        seq_lens.extend(model_input['seq_lens'])
                else:
                    context_embeddings = self._embed_context(context)  # (seq_len, hidden_size)
                    # Embed each choice one by one to avoid OOM with large models
                    for choice in doc['choices']:
                        # model_input = self._input_embed_words(context_embeddings, doc['choices'])
                        model_input = self._input_embed_words(context_embeddings, [choice], shift_inp_right=True)
                        model_output = self.gpt2(inputs_embeds=model_input['inputs_embeds'],
                                                 attention_mask=model_input['attention_mask'],
                                                 output_hidden_states=True,
                                                 return_dict=True)
                        logprobs.append(
                            torch.nn.functional.log_softmax(model_output.logits, dim=-1).squeeze(0)  # (seq_len, vocab)
                        )  # (#choices, seq_len, vocab)
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
                    score_list.append(choice_lprobs.sum())
                results.append({'scores': torch.stack(score_list), 'is_exact_match': is_exact_match})

        return results
