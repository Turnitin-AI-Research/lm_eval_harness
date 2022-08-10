"""Utilities for distributed encoding"""
from typing import Dict, List
import numpy as np
import torch
from lm_eval.base import rf


class SegmentedSample(Dict):
    """Segmented Sample class that enables empty instantiation and verification"""

    def __init__(self, *args, **kwargs):
        # self['query']: str  # Main input text
        # self['gold']: int  # Index of correct answer if there's only one.
        # self['gold_indices']: List[int]  # Multiple indices of correct answers if there are multiple
        # self['choices']: List[str]  # choice strings
        # self['hints']: List[str]  # formatting cues, hints
        # self['segments']: List[str]  # query segments to encode independently
        super().__init__(*args, **kwargs)


class MCSimilarityScores(List[float]):
    """Multiple Choice Match scores output by distributed_encoding_similarity"""
    pass


class DistEncTaskMixin:
    """
    Mixin for Distributed Encoding Task.
    Refer to new_multiple_choice_task.py for software design context.
    """

    def reorg_for_encoding(self, doc: SegmentedSample) -> SegmentedSample:
        """Reorganize doc segments based on encoding scheme"""
        if self.encoding_scheme in ['concat_all_examples', 'concat_each_example']:
            out_doc = doc.copy()
            out_doc['segments'] = self.SEGMENT_DELIMITER.join(doc['segments'])
        return out_doc

    def _reorg_for_fewshot(self, doc: SegmentedSample) -> SegmentedSample:
        """Reorganize a doc as a fewshot example"""
        if self.encoding_scheme in ['concat_all_examples', 'concat_each_example']:
            assert len(doc['segments']) == 1
            out_doc = SegmentedSample(
                segments=[doc['segments'][0] + self.ANSWER_DELIMITER + doc['choices'][doc['gold']]])
        else:
            out_doc = SegmentedSample(segments=doc['segments'] + [doc['choices'][doc['gold']]])
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
            Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
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

        # description = description + "\n\n" if description else ""
        description = None if not description else SegmentedSample(segments=[description])

        if num_fewshot == 0:
            # labeled_examples = ""
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

            context = [self._reorg_for_fewshot(description)] + [
                self._reorg_for_fewshot(example) for example in fewshotex] + [doc]
            if self.encoding_scheme == 'concat_all_examples':
                # Merge all samples into one
                context = [SegmentedSample(
                    segments=[segment for example in context for segment in example['segments']])]
            # labeled_examples = (
            #     "\n\n".join(
            #         [
            #             self.doc_to_text(doc) + self.doc_to_target(doc)
            #             for doc in fewshotex
            #         ]
            #     )
            #     + "\n\n"
            # )

        # example = self.doc_to_text(doc)
        # return description + labeled_examples + example
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

    def process_results(self, doc: SegmentedSample, results: MCSimilarityScores):
        gold = doc["gold_indices"]
        acc = 1.0 if np.argmax(results) in gold else 0.0

        return {
            "acc": acc
        }


class DistEncLMMixin:
    def tok_encode(self, string: str) -> torch.Tensor:
        return self.tokenizer.encode(string, add_special_tokens=False, return_tensors='pt')  # (N, T)

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
            'input_ids': torch.tensor(seqs, dtype=torch.long, device=self.device),
            'seq_lens': torch.tensor(seqs, dtype=torch.long, device=self.device),
            'attention_mask': torch.tensor([[1] * len + [0] * (max_len - len) for len in seq_lens],
                                           dtype=torch.long, device=self.device)
        }

    def _reduce_concept_sequences(self, model_output: Dict) -> torch.Tensor:
        """Aggregate a sequence of concept vectors into a single concept vector"""
        concept_seqs = model_output['hidden_states'][-1]  # (batch, padding-len, hidden_size)
        if self.reduction_scheme == 'last':
            aggregated_vectors = concept_seqs[:, model_output['seq_lens'] - 1, :]  # (batch, hidden_size)
        elif self.reduction_scheme == 'mean':
            aggregated_vectors = (concept_seqs * model_output['attention_mask']).sum() / model_output['seq_lens']
        else:
            raise NotImplementedError
        return aggregated_vectors  # (batch, hidden_size)

    def _embed_sample(self, sample: SegmentedSample, include_choices: bool = False) -> SegmentedSample:
        segments = self._tok_batch_encode(sample['segments'])  # (batch, padding_len)
        choices = self._tok_batch_encode(sample['choices'])  # (batch, padding_len)

        # Transformer encode
        segments = self.gpt2(input_ids=segments['input_ids'],
                             attention_mask=segments['attention_mask'],
                             output_hidden_states=True,
                             return_dict=True)
        segment_embeddings = self._reduce_concept_sequences(segments)  # (batch, hidden_size)
        sample['embedding'] = segment_embeddings.mean()
        if include_choices:
            choice = self.gpt2(input_ids=choices['input_ids'][[sample['gold']], :],
                               attention_mask=choices['attention_mask'],
                               output_hidden_states=True,
                               return_dict=True)
            segment_embeddings = segment_embeddings.cat(choice, dim=0)
        else:
            choices = self.gpt2(input_ids=choices['input_ids'][sample['gold_indices'], :],
                                attention_mask=choices['attention_mask'],
                                output_hidden_states=True,
                                return_dict=True)
            sample['choices_embeddings'] = self._reduce_concept_sequences(choices)
        return sample

    def distributed_encoding_similarity(self, requests_args):
        """Compute similarity of choices.

        :param requests: list
            A list of pairs (context, doc)
            context: List of context SegmentedSamples: Optional[description] + [few-shot samples] + [doc]
            doc: The query doc SegmentedSample
        :return:
            A list of results, [MCSimilarityScores], one per request (doc)
            MCSimilarityScores: A list of similarity scores of a request (doc), one per choice [score,]
        """
        # Eschewing batching in order to reduce complexity
        encoded_context
        for context, doc in requests_args:
            encoded_segments = [self.tok_encode(segment) for segment in request['segments']]
            encoded_choices = [self.tok_encode(choice) for choice in request['choices']]

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc = [self.eot_token_id]
            else:
                context_enc = self.tok_encode(context)

            continuation_enc = self.tok_encode(continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)
