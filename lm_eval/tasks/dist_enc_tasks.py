"""Task modifications for distributed encoding"""
from typing import List, Optional, Type, Dict
from collections import UserDict
import torch
from lm_eval.base import Task, rf
from lm_eval.metrics import mean
from . import hellaswag, webqs


class SegmentedSample(UserDict):
    """Segmented Sample class that enables empty instantiation and verification"""

    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, **kwargs)
        # self['segments']: List[str] = [] # query / question segments to encode independently
        # self['choices']: List[str] = [] # choice / answer strings
        # self['gold']: Optional[int] = None # Index of correct answer / choice if there's only one else None.
        # self['gold_indices']: List[int] = [] # Possibly multiple indices of correct answers / choices
        # self['question_hint']: Optional[str] = None  # question prompt
        # self['answer_hint']: Optional[str] = None  # Answer prompt
        self['task'] = task  # Pointer to task for passing in configuration to the model

    def copy(self):
        return self.__class__(**self)

    @property
    def task(self):
        return self['task']

    @task.setter
    def task(self, t):
        self['task'] = t

    def __eq__(self, __o: object) -> bool:
        return self.task.doc_to_decontamination_query(self) == self.task.doc_to_decontamination_query(__o)


class DistEncTaskMixin:
    """
    Mixin for Distributed Encoding Task.
    Refer to new_multiple_choice_task.py for software design context.
    """
    SEGMENT_DELIMITER: str = None
    ANSWER_DELIMITER: str = None
    ENCODING_SCHEME: str = None  # 'segment_each_example', 'concat_each_example', 'concat_all_examples',
    KWARGS: dict = None

    def __init__(self, *args, encoding_scheme: str = 'concat_all_examples', task_type: Optional[str] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ENCODING_SCHEME: str = encoding_scheme
        self.SEGMENT_DELIMITER: str = '\n'  # override this in subclass' constructor
        self.ANSWER_DELIMITER: str = ' '  # override this in subclass' constructor
        self.EXAMPLE_DELIMITER: str = '\n\n'  # override this in subclass' constructor
        self.TASK_TYPE = task_type
        self.KWARGS = kwargs

    def __repr__(self) -> str:
        return super().__repr__() + (f', {self.KWARGS}' if self.KWARGS else '')

    def verify_config(self):
        """Verify arguments collected from various mixins and objects"""
        assert self.SEGMENT_DELIMITER is not None
        assert self.ANSWER_DELIMITER is not None
        assert self.EXAMPLE_DELIMITER is not None
        assert self.ENCODING_SCHEME in ['concat_all_examples', 'concat_each_example', 'cross_encoding',
                                        'segment_each_example', 'merge_all_segments']
        assert self.TASK_TYPE in [None, 'gen']

    def config(self):
        return {
            'SEGMENT_DELIMITER': self.SEGMENT_DELIMITER,
            'ANSWER_DELIMITER': self.ANSWER_DELIMITER,
            'ENCODING_SCHEME': self.ENCODING_SCHEME,
            'EXAMPLE_DELIMITER': self.EXAMPLE_DELIMITER,
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

    def _answer_text(self, doc: Dict, *, choice: Optional[int] = None) -> str:
        """Given a choice number, return a formatted answer text along with segment-delimiter prefix"""
        if 'answer_hint' not in doc:  # sentence continuation
            answer = '' if choice is None else self.ANSWER_DELIMITER + doc['choices'][choice]
        else:  # Separate answer section
            answer = self.SEGMENT_DELIMITER + doc['answer_hint']
            if choice is not None:
                answer = (answer + self.ANSWER_DELIMITER + doc['choices'][choice])
        return answer

    def _answer_segment(self, doc: Dict, *, choice: Optional[int] = None) -> str:
        """Given a choice number, return a formatted answer segment without segment separator"""
        if 'answer_hint' not in doc:  # sentence continuation
            answer = '' if choice is None else doc['choices'][choice]
        else:  # Separate answer section
            answer = doc['answer_hint']
            if choice is not None:
                answer = (answer + self.ANSWER_DELIMITER + doc['choices'][choice])
        return answer

    def _make_fewshotex(self, doc: SegmentedSample,
                        *,
                        exclude_answer: bool = False) -> SegmentedSample:
        """
        * Reorganize the doc as a fewshot example.
        * Remove all unnecessary info.
        """
        # doc = self.process_segments(doc)
        if self.ENCODING_SCHEME in ['concat_all_examples', 'cross_encoding', 'concat_each_example']:
            # assert len(doc['segments']) == 1
            context = self.SEGMENT_DELIMITER.join(doc['segments'])
            # answer = '' if exclude_answer else self.ANSWER_DELIMITER + doc['choices'][doc['gold']]
            answer = self._answer_text(doc, choice=None if exclude_answer else doc['gold_indices'][0])
            out_doc = SegmentedSample(task=doc.task, segments=[context + answer])
        elif self.ENCODING_SCHEME in ['segment_each_example', 'merge_all_segments']:
            # answer = [] if exclude_answer else [doc['choices'][doc['gold']]]
            answer = [self._answer_segment(doc, choice=None if exclude_answer else doc['gold_indices'][0])]
            out_doc = SegmentedSample(task=doc.task, segments=doc['segments'] + answer)
        else:
            raise ValueError
        return out_doc

    def _merge_fewshotex(self, doc: SegmentedSample, examples: List[SegmentedSample]) -> SegmentedSample:
        """Processing on set of fewshot examples:
        if ENCODING_SCHEME == 'concat_all_examples' or 'cross_encoding':
            concatenate segments of all examples into one
        elif ENCODING_SCHEME == 'merge_all_segments':
            aggregate all segments into one list
        """
        for example in examples:
            assert len(example['segments']) == 1
        segments = [segment for example in examples for segment in example['segments']]
        if self.ENCODING_SCHEME in ['concat_all_examples']:
            return SegmentedSample(task=doc.task, segments=[self.EXAMPLE_DELIMITER.join(segments)])
        elif self.ENCODING_SCHEME == 'cross_encoding':
            return SegmentedSample(task=doc.task, segments=[self.EXAMPLE_DELIMITER.join(segments)],
                                   choices=[(self.ANSWER_DELIMITER + doc['choices'][i]) for i, _ in enumerate(doc['choices'])])
        elif self.ENCODING_SCHEME == 'merge_all_segments':
            return SegmentedSample(task=doc.task, segments=segments)
        else:
            raise ValueError

    @staticmethod
    def _remove_label(doc: SegmentedSample) -> SegmentedSample:
        out_doc = doc.copy()
        out_doc['choices'] = out_doc['gold'] = out_doc['gold_indices'] = None
        return out_doc

    def fewshot_context(
        self, doc: SegmentedSample, num_fewshot: int, provide_description: bool = None, rnd=None,
        description: str = None
    ) -> List[SegmentedSample]:
        """Returns a fewshot context list that is made up of a prepended description
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
            List of samples comprising the fewshot context. Every segment of each of these samples should be individually
            embedded and the resulting vectors combined per sample and the samples then combined thereafter to get
            the final embedding of the context.
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

        description_list = [] if not description else [
            self._make_fewshotex(SegmentedSample(task=doc.task, segments=[description]), exclude_answer=True)]

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
                num_sampled_docs = len(fewshotex)
                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]
                assert len(fewshotex) == min(num_fewshot, num_sampled_docs)

        context_list = description_list + [
            self._make_fewshotex(example) for example in fewshotex] + [
            self._make_fewshotex(self._remove_label(doc), exclude_answer=True)]

        if self.ENCODING_SCHEME in ['concat_all_examples', 'merge_all_segments', 'cross_encoding']:
            # Merge all samples into one
            context_list = [self._merge_fewshotex(doc, context_list)]
        return context_list

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
        if self.TASK_TYPE is None:
            factory_func = rf.distributed_encoding_similarity
        else:
            factory_func = rf.distributed_encoding_generation
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
        golds = doc["gold_indices"]
        # Framework adds yet another layer of aggregation on top of results. Unwind that.
        assert len(results) == 1
        results = results[0]
        scores = results['scores']
        acc = 1.0 if torch.argmax(scores) in golds else 0.0
        choice_len = torch.tensor([len(choice) for choice in doc['choices']], device=scores.device, dtype=torch.long)
        acc_norm = 1.0 if torch.argmax(scores / choice_len) in golds else 0.0
        ret_dict = {
            "acc": acc,
            "acc_norm": acc_norm,
            "rand_acc": 1. / len(scores)
        }
        if 'is_exact_match' in results:
            ret_dict['em'] = 1.0 if any(results['is_exact_match'][idx] for idx in doc['gold_indices']) else 0.0

        return ret_dict

    def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
            "rand_acc": False,
            "em": True
        }

    def aggregation(self):
        return {
            "acc": mean,
            "acc_norm": mean,
            "rand_acc": mean,
            "em": mean
        }

    def doc_to_target(self, doc):
        raise NotImplementedError('This method should not have been called')

    def doc_to_text(self, doc):
        raise NotImplementedError('This method should not have been called')


def make_gen_class(cls: Type[DistEncTaskMixin]) -> Type[Task]:
    """Convert a *Dist task to a *DistGen task"""
    class curryed_class(cls):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, task_type='gen', **kwargs)
    return curryed_class


class HellaSwagDist(DistEncTaskMixin, hellaswag.HellaSwag):
    def __init__(self, *args, **kwargs) -> None:
        # Super task classes are not passed any arguments by the harness but we do that here just for future proofing
        super().__init__(*args, **kwargs)
        # self.SEGMENT_DELIMITER: str = '\n'
        # self.ANSWER_DELIMITER: str = ' '
        # self.EXAMPLE_DELIMITER: str = '\n\n'
        self.verify_config()

    def _process_doc(self, doc):
        out_doc = SegmentedSample(super()._process_doc(doc), task=self)
        # Segments (including hints) so that they may be individually encoded (e.g 'Question: <question text>')
        out_doc['segments'] = [out_doc['query']]
        # Indices of one or more correct targets from out_doc['choices']
        out_doc['gold_indices'] = [out_doc['gold']]
        return self.process_segments(out_doc)


class WebQsDist(DistEncTaskMixin, webqs.WebQs):
    def __init__(self, *args, **kwargs) -> None:
        # Super task classes are not passed any arguments by the harness but we do that here just for future proofing
        super().__init__(*args, **kwargs)
        # self.SEGMENT_DELIMITER: str = '\n'
        # self.ANSWER_DELIMITER: str = ' '
        # self.EXAMPLE_DELIMITER: str = '\n\n'
        self.verify_config()

    def test_docs(self):
        return map(self._process_doc, super().test_docs())

    def training_docs(self):
        return map(self._process_doc, super().training_docs())

    def _process_doc(self, doc):
        out_doc = SegmentedSample(doc, task=self)
        # Extract all hints so that they may be optionally individually encoded without text
        out_doc['question_hint'] = 'Question:'
        out_doc['answer_hint'] = 'Answer:'
        # Segments (including hints) so that they may be individually encoded (e.g 'Question: <question text>')
        out_doc['segments'] = ['Question: ' + out_doc['question']]
        out_doc['choices'] = doc['answers']
        # All out_doc['choices'] are gold targets
        out_doc['gold_indices'] = list(range(len(out_doc['choices'])))
        out_doc['gold'] = None  # Indicates we have more than one possible targets
        return self.process_segments(out_doc)

    def process_results(self, doc: SegmentedSample, results: List[torch.Tensor]):
        """All choices in the test set are gold targets"""
        metrics = super().process_results(doc, results)
        # All choices are legitimate targets, therefore acc should be 1.0
        assert metrics['acc'] == metrics['acc_norm'] == 1.0
        return {
            # Exact match is the only accuracy metric for webqs.
            'acc': metrics['em']
        }
