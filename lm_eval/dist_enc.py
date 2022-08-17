"""Utilities for distributed encoding"""
from typing import Dict, List, Optional
from collections import UserDict
from tqdm import tqdm
import torch
from lm_eval.base import rf
from lm_eval.metrics import mean


class SegmentedSample(UserDict):
    """Segmented Sample class that enables empty instantiation and verification"""

    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, **kwargs)
        # self['query']: str  # Main input text
        # self['gold']: int  # Index of correct answer if there's only one.
        # self['gold_indices']: List[int]  # Multiple indices of correct answers if there are multiple
        # self['choices']: List[str]  # choice strings
        # self['hints']: List[str]  # formatting cues, hints
        # self['segments']: List[str]  # query segments to encode independently
        self['task'] = task  # Pointer to task for passing in configuration to the model

    def copy(self):
        return self.__class__(**self)

    @property
    def task(self):
        return self['task']

    @task.setter
    def task(self, t):
        self['task'] = t


class DistEncTaskMixin:
    """
    Mixin for Distributed Encoding Task.
    Refer to new_multiple_choice_task.py for software design context.
    """
    SEGMENT_DELIMITER: str = None
    ANSWER_DELIMITER: str = None
    ENCODING_SCHEME: str = None  # 'segment_each_example'*, 'concat_each_example', 'concat_all_examples',
    KWARGS: dict = None

    def verify_config(self):
        """Verify arguments collected from various mixins and objects"""
        assert self.SEGMENT_DELIMITER is not None
        assert self.ANSWER_DELIMITER is not None
        assert self.ENCODING_SCHEME in ['concat_all_examples', 'concat_each_example', 'segment_each_example', 'cross_encoding']

    def config(self):
        return {
            'SEGMENT_DELIMITER': self.SEGMENT_DELIMITER,
            'ANSWER_DELIMITER': self.ANSWER_DELIMITER,
            'ENCODING_SCHEME': self.ENCODING_SCHEME,
            'kwargs': self.KWARGS
        }

    def process_segments(self, doc: SegmentedSample) -> SegmentedSample:
        """Reorganize doc segments based on encoding scheme"""
        if self.ENCODING_SCHEME in ['concat_all_examples', 'concat_each_example', 'cross_encoding'] and (len(doc['segments']) > 1):
            out_doc = doc.copy()
            out_doc['segments'] = [self.SEGMENT_DELIMITER.join(doc['segments'])]
            return out_doc
        else:
            return doc

    def _make_fewshotex(self, doc: SegmentedSample, exclude_answer: bool = False) -> SegmentedSample:
        """Reorganize a doc as a fewshot example. Remove all unnecessary info."""
        doc = self.process_segments(doc)
        if self.ENCODING_SCHEME in ['concat_all_examples', 'concat_each_example', 'cross_encoding']:
            assert len(doc['segments']) == 1
            context = doc['segments'][0]
            answer = '' if exclude_answer else self.ANSWER_DELIMITER + doc['choices'][doc['gold']]
            out_doc = SegmentedSample(task=doc.task, segments=[context + answer])
        else:
            answer = [] if exclude_answer else [doc['choices'][doc['gold']]]
            out_doc = SegmentedSample(task=doc.task, segments=doc['segments'] + answer)
        return out_doc

    def fewshot_context(
        self, doc: SegmentedSample, num_fewshot: int, provide_description: bool = None, rnd=None,
        description: str = None
    ) -> List[SegmentedSample]:
        """Returns a fewshot context that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: SegmentedSample
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param provide_description: bool
            Not implemented, and this option is deprecated and will be removed in a future version in favor of a
            different description providing method
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: List[SegmentedSample]
            List of samples comprising the fewshot context. Every segment of each of these samples is individually
            embedded and the resulting vectors are then combined to get the final embedding of the entire context.
        """
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print(
                "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
            )

        description = [] if not description else [
            self._make_fewshotex(SegmentedSample(task=doc.task, segments=[description]))]

        if num_fewshot == 0:
            fewshotex = []
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs()
                        if self.has_validation_docs()
                        else self.test_docs()
                    )

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

        context = description + [
            self._make_fewshotex(example) for example in fewshotex] + [
            self._make_fewshotex(doc, exclude_answer=True)]

        if self.ENCODING_SCHEME == 'concat_all_examples':
            # Merge all samples into one
            context = [self._make_fewshotex(
                SegmentedSample(
                    task=doc.task,
                    segments=[segment for example in context for segment in example['segments']]))]
        return context

    def construct_requests(self, doc: SegmentedSample, ctx: List[SegmentedSample]):
        """Uses RequestFactory to construct Requests and returns a single
        Request which will be sent to the LM.

        :param doc: Object
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: List[Object]
            The context, generated by fewshot_context. A list of few shot examples followed
            by the query `doc`.
        """
        # Using __getattr__ overload to construct a function. Yet another element of poor software design.
        factory_func = rf.distributed_encoding_similarity
        return factory_func(ctx, doc)
        # lls = [
        #     # The index [0] below references index of the returned metrics for a sample - in this case the first one.
        #     # Request.index similarly indexes the sequence of metrics returned by the request_type function
        #     # (e.g. loglikelihood, loglikelihood_rolling etc.).
        #     # Not only is this software 'design' not intuitive and undocumented, it is also unnecessarily wasteful
        #     # because it forces the model to be invoked repeatedly wiht the same input as many times as the number of
        #     # metrics computed. Poor 'design' overall. I am not changing this however, since that would imply
        #     # making changes to the entire harness and impacting all other tasks.
        #     factory_func(ctx, " {}".format(choice))[0] for choice in doc["choices"]
        # ]

        # return lls

    def process_results(self, doc: SegmentedSample, results: List[torch.Tensor]):
        gold = doc["gold_indices"]
        assert len(results) == 1
        acc = 1.0 if torch.argmax(results[0]) in gold else 0.0

        return {
            "acc": acc,
            "rand_acc": 1. / len(results[0])
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "rand_acc": False,
        }

    def aggregation(self):
        return {
            "acc": mean,
            "rand_acc": mean,
        }


class DistEncLMMixin:
    WORD_AGG_SCHEME: str = None
    SEGMENT_AGG_SCHEME: str = 'mean'
    EXAMPLE_AGG_SCHEME: str = 'mean'
    SIMILARITY_FUNC: str = None
    NORM: str = None

    def verify_config(self):
        assert self.WORD_AGG_SCHEME in ['last', 'mean']
        assert self.SEGMENT_AGG_SCHEME == 'mean'
        assert self.EXAMPLE_AGG_SCHEME == 'mean'
        assert self.SIMILARITY_FUNC in ['dot_product', 'cosine_sim']
        assert self.NORM in ['L2', None]  # TODO: layer norm

    def _normalize(self, T: torch.Tensor, *, dim: int = -1):
        """Compute Lp norm i.e., normalize the magnitude of vectors to 1."""
        if self.NORM is None:
            return T
        elif self.NORM == 'L2':
            return torch.nn.functional.normalize(T, p=2, dim=dim)
        else:
            raise NotImplementedError

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def _tok_batch_encode_fast(self, strings: List[str]) -> Dict[str, torch.Tensor]:
        # WARNING: August 2022: cannot rely on return_length=True because returned lengths are incorrect
        if self.tokenizer.pad_token_id is None:
            reset_pad_token_id = True
            self.tokenizer.pad_token_id = 0
        else:
            reset_pad_token_id = False
        batch = self.tokenizer(text=strings, is_split_into_words=False, padding=True,
                               add_special_tokens=False, return_tensors='pt')
        if reset_pad_token_id:
            self.tokenizer.pad_token_id = None
        batch['input_ids'] = batch['input_ids'].to(device=self.device)  # (N,T)
        batch['attention_mask'] = batch['attention_mask'].to(device=self.device)  # (N,T) boolean
        batch['seq_lens'] = batch['attention_mask'].sum(dim=-1)  # (N,)
        return batch

    def _tok_batch_encode(self, strings: List[str]) -> Dict[str, torch.Tensor]:
        seqs = [self.tok_encode(string) for string in strings]
        # Truncate from left if needed
        seqs = [seq if len(seq) <= self.max_length else seq[len(seq) - self.max_length:] for seq in seqs]
        seq_lens = [len(seq) for seq in seqs]
        max_len = max(seq_lens)
        # pad
        seqs = [seq + [0] * (max_len - len(seq)) for seq in seqs]

        return {
            'input_ids': torch.tensor(seqs, dtype=torch.long, device=self.device),  # (N,T)
            'seq_lens': torch.tensor(seq_lens, dtype=torch.long, device=self.device),  # (N,)
            'attention_mask': torch.tensor([[1] * len + [0] * (max_len - len) for len in seq_lens],
                                           dtype=torch.long, device=self.device)  # (N,T)
        }

    def _reduce_word_sequences(self, model_output: Dict, model_input: Dict) -> torch.Tensor:
        """Aggregate a sequence of word vectors into a single concept vector

        :param model_output: Dict.
            HuggingFace model_output dict.
        :param model_input: Dict
            Model input dict
        :return: torch.Tensor, shape = (batch_size, hidden_size)
            Each sequence in the batch reduced to an embedding of size hidden_size
        """
        concept_seqs = model_output['hidden_states'][-1]  # (batch, padding-len, hidden_size)
        if self.WORD_AGG_SCHEME == 'last':
            aggregated_vectors = torch.stack([concept_seqs[row, seq_len - 1, :]
                                              for row, seq_len in enumerate(model_input['seq_lens'])])  # (batch, hidden_size)
        elif self.WORD_AGG_SCHEME == 'mean':
            aggregated_vectors = ((concept_seqs * model_input['attention_mask'].unsqueeze(-1)).sum(dim=1)
                                  / model_input['seq_lens'].unsqueeze(-1))
        else:
            raise NotImplementedError
        return self._normalize(aggregated_vectors)  # (batch, hidden_size)

    def _embed_sample(self, sample: SegmentedSample) -> SegmentedSample:
        """Embed segments if present, into a single embedding.
        If choices are present, then embed each into a single vector."""
        if 'segments' in sample and sample['segments']:
            model_input = self._tok_batch_encode(sample['segments'])  # (#segments, padding_len)
            model_output = self.gpt2(input_ids=model_input['input_ids'],
                                     attention_mask=model_input['attention_mask'],
                                     output_hidden_states=True,
                                     return_dict=True)
            segment_embeddings = self._reduce_word_sequences(
                model_output, model_input)  # (#segments, hidden_size)
            segment_embeddings = self._normalize(segment_embeddings)
            if self.SEGMENT_AGG_SCHEME == 'mean':
                sample['context_embedding'] = segment_embeddings.mean(dim=0)  # (hidden_size,)
            else:
                raise NotImplementedError
        if 'choices' in sample and sample['choices']:
            model_input = self._tok_batch_encode(sample['choices'])  # (#choices, padding_len)
            model_output = self.gpt2(input_ids=model_input['input_ids'],
                                     attention_mask=model_input['attention_mask'],
                                     output_hidden_states=True,
                                     return_dict=True)
            sample['choices_embeddings'] = self._reduce_word_sequences(
                model_output, model_input)  # (#choices, hidden_size)
            sample['choices_embeddings'] = self._normalize(sample['choices_embeddings'])  # (#choices, hidden_size)
        return sample

    def _embed_context(self, examples: List[SegmentedSample]) -> torch.Tensor:
        """Embed a context (represented as a list of SegmentedSamples) into a single embedding vector"""
        examples = [self._embed_sample(example) for example in examples]
        if len(examples) > 1:
            if self.EXAMPLE_AGG_SCHEME == 'mean':
                context_embedding = torch.stack([example['context_embedding'] for example in examples]).mean(dim=0)
                context_embedding = self._normalize(context_embedding)
            else:
                raise NotImplementedError
        else:
            context_embedding = examples[0]['context_embedding']
        return context_embedding  # (hidden_size,)

    def distributed_encoding_similarity(self, requests_args) -> List[torch.Tensor]:
        """Compute similarity of choices.

        :param requests: list
            A list of pairs (context, doc)
            context: List of context SegmentedSamples: Optional[description] + [few-shot samples] + [doc]
            doc: The query doc SegmentedSample
        :return:
            A list of results, [torch.Tensor], one per request (doc)
            torch.Tensor: A list of similarity scores of a request (doc), one per choice [score,]
        """
        # Eschewing batching in favor of simplicity for now
        results = []
        self.gpt2.eval()
        with torch.no_grad():
            for context, doc in tqdm(requests_args):
                context_embedding = self._embed_context(context)  # (hidden_size,)
                doc = doc.copy()
                del doc['segments']
                choice_embeddings = self._embed_sample(doc)['choices_embeddings']  # (#choices, hidden_size)
                if self.SIMILARITY_FUNC == 'dot_product':
                    scores = torch.mm(choice_embeddings, context_embedding.unsqueeze(dim=1)).squeeze()  # (#choices,)
                elif self.SIMILARITY_FUNC == 'cosine_sim':
                    scores = torch.cosine_similarity(
                        choice_embeddings, context_embedding.unsqueeze(dim=0))  # (#choices,)
                else:
                    raise NotImplementedError
                results.append(scores)
        return results
