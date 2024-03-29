"""Task modifications for distributed encoding"""
from typing import List, Optional, Type, Dict, Tuple
from collections import UserDict
import re
import numpy as np
import nltk
from lm_eval.base import Task, rf
from lm_eval.metrics import mean
from lm_eval.tasks import wsc273
from . import hellaswag, webqs

SENT_SEP = re.compile(r'(\.(?=\s+))')


def sent_tokenize(text: str) -> List[str]:
    """
    Sentence tokenizer that maintains leading whitespace in text. This is a poor man's version since Mr. Smith will
    be broken into two sentences 'Mr.' and ' Smith', which is not what we want.
    """
    ret_splits = None
    if text:
        splits = SENT_SEP.split(text)
        if len(splits) > 1:
            l = len(splits)
            ret_splits = [splits[2*i] + splits[2*i+1] for i in range(l // 2)]
            if l % 2:
                ret_splits.append(splits[-1])
        else:
            ret_splits = splits
    else:
        ret_splits = [text]

    return ret_splits


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
    # TODO: encoding_scheme default was mean. Need to retest with absent values
    def __init__(self, *args, encoding_scheme: str = None, task_type: Optional[str] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ENCODING_SCHEME: str = encoding_scheme  # passed in via config
        # Delimiter separating few-shot examples. Override this in subclass' constructor
        self.EXAMPLE_DELIMITER: str = '\n\n'
        # Delimiter between question and answer-hint only if answer-hint exists; set to empty string otherwise. Override in subclass.
        self.QA_DELIMITER: str = ''
        # Answer hint. Set in subclass if task has answer-hint. e.g. 'Answer:""
        self.ANSWER_HINT: str = ''
        # Answer hint split into sentences needed for sentence-level decoding. Override in subclass.
        self.ANSWER_HINT_SENTS: List[str] = []
        # Prefix to the choice / answer / completion. Override in subclass.
        self.HINT_ANSWER_DELIMITER: str = ' '
        # Task-type: 'gen' or None. Automatically set by factory function: make_gen_class
        self.TASK_TYPE = task_type
        self.KWARGS = kwargs

    def __repr__(self) -> str:
        return super().__repr__() + (f', {self.KWARGS}' if self.KWARGS else '')

    def verify_config(self):
        """Verify arguments collected from various mixins and objects"""
        assert self.EXAMPLE_DELIMITER is not None
        assert self.HINT_ANSWER_DELIMITER is not None
        assert self.ANSWER_HINT is not None
        assert self.ANSWER_HINT_SENTS is not None
        assert self.ENCODING_SCHEME in ['concat_all_examples', 'concat_each_example', 'cross_encoding',
                                        'segment_each_example', 'merge_all_segments', 'sentence_level_segmentation']
        assert self.TASK_TYPE in [None, 'gen']

    @property
    def config(self):
        rexp = re.compile(r'[A-Z_]+')
        return {k: v for k, v in vars(self).items() if rexp.fullmatch(k)}

    def _prepare_doc(self, doc: SegmentedSample) -> SegmentedSample:
        """Reorganize doc based on task parameters"""
        out_doc = doc.copy()
        self.answer_hint_w_prefix = self.QA_DELIMITER + self.ANSWER_HINT
        out_doc['choices'] = [self.HINT_ANSWER_DELIMITER + choice for choice in doc['choices']]
        if self.ENCODING_SCHEME in ['concat_all_examples', 'concat_each_example', 'cross_encoding'] and (len(doc['segments']) > 1):
            out_doc['segments'] = [''.join(doc['segments'])]
        elif self.ENCODING_SCHEME == 'sentence_level_segmentation':
            out_doc['segments'] = [sentence for segment in out_doc['segments'] for sentence in sent_tokenize(segment)]
            out_doc['choices_sents'] = [sent_tokenize(choice) for choice in out_doc['choices']]
            if self.ANSWER_HINT_SENTS:
                self.answer_hint_sents_w_prefix = (
                    [self.QA_DELIMITER + self.ANSWER_HINT_SENTS[0]] + self.ANSWER_HINT_SENTS[1:])
            else:
                self.answer_hint_sents_w_prefix = []
        return out_doc

    def _answer_text(self, doc: Dict, *, choice_idx: Optional[int] = None) -> str:
        """Given a choice_idx number, return a formatted answer text along with QA-delimiter prefix"""
        if choice_idx is None:
            return self.answer_hint_w_prefix
        else:
            return self.answer_hint_w_prefix + doc['choices'][choice_idx]

    def _answer_sents(self, doc: Dict, *, choice_idx: Optional[int] = None) -> str:
        """Given a choice_idx number, return list of answer segments without segment separator"""
        assert self.ENCODING_SCHEME == 'sentence_level_segmentation'
        return self.answer_hint_sents_w_prefix + ([] if choice_idx is None else doc['choices_sents'][choice_idx])

    def _make_fewshotex(self, doc: SegmentedSample, *, exclude_answer: bool = False) -> SegmentedSample:
        """
        * Reorganize the doc as one fewshot example.
        * Remove all unnecessary info.
        """
        choice_idx = None if exclude_answer else doc['gold_indices'][0]
        if self.ENCODING_SCHEME in ['concat_all_examples', 'cross_encoding', 'concat_each_example']:
            assert len(doc['segments']) == 1
            # context = ''.join(doc['segments']) if len(doc['segments']) > 1 else doc['segments'][0]
            question = doc['segments'][0]
            out_segments = [question + self._answer_text(doc, choice_idx=choice_idx)]
        elif self.ENCODING_SCHEME in ['segment_each_example', 'merge_all_segments']:
            out_segments = doc['segments'] + [self._answer_text(doc, choice_idx=choice_idx)]
        elif self.ENCODING_SCHEME == 'sentence_level_segmentation':
            out_segments = doc['segments'] + self._answer_sents(doc, choice_idx=choice_idx)
        else:
            raise ValueError(f'Invalid ENCODING_SCHEME: {self.ENCODING_SCHEME}')
        # Sometimes out_segments can be empty strings. Remove those.
        out_doc = SegmentedSample(task=doc.task, segments=[seg for seg in out_segments if seg])
        return out_doc

    def _make_fewshot_query(self, doc: SegmentedSample) -> SegmentedSample:
        """
        * Reorganize the doc as one fewshot example without the answer. This is meant for the query example only.
        * Remove all unnecessary info.
        """
        return self._make_fewshotex(doc, exclude_answer=True)

    def _merge_fewshotex(self, doc: SegmentedSample, examples: List[SegmentedSample]) -> SegmentedSample:
        """Process a set of fewshot examples:
        if ENCODING_SCHEME == 'concat_all_examples' or 'cross_encoding':
            concatenate segments of all examples into one
        elif ENCODING_SCHEME == 'merge_all_segments':
            aggregate all segments into one list
        """
        segments = [segment for example in examples for segment in example['segments']]
        if self.ENCODING_SCHEME in ['concat_all_examples', 'cross_encoding']:
            assert all(len(example['segments']) == 1 for example in examples)
            return SegmentedSample(task=doc.task, segments=[self.EXAMPLE_DELIMITER.join(segments)])
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
        invoked by evaluator.py

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

        description = None if not description else self._make_fewshotex(SegmentedSample(task=doc.task, segments=[description]),
                                                                        exclude_answer=True)

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

        return self._make_contextlist(doc, fewshotex, description)

    def _make_contextlist(self, doc: SegmentedSample, fewshotex: List[SegmentedSample], description: Optional[SegmentedSample]) -> List[SegmentedSample]:
        description_list = [] if description is None else [description]
        context_list = description_list + [
            self._make_fewshotex(example) for example in fewshotex] + [
            self._make_fewshot_query(self._remove_label(doc))]

        if self.ENCODING_SCHEME in ['concat_all_examples', 'merge_all_segments', 'cross_encoding']:
            # Merge all examples into one
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

    def process_results(self, doc: SegmentedSample, results: Tuple[Dict]):
        golds = doc["gold_indices"]
        # Framework adds yet another layer of aggregation on top of results. Unwind that.
        assert len(results) == 1
        results = results[0]
        scores = results['scores']
        acc = 1.0 if np.argmax(scores) in golds else 0.0
        ret_dict = {
            "acc": acc,
            "rand_acc": 1. / len(scores)
        }
        if self.ENCODING_SCHEME == 'cross_encoding' or self.TASK_TYPE == 'gen':
            choice_len = np.array([len(choice) for choice in doc['choices']])
            acc_norm = 1.0 if np.argmax(scores / choice_len) in golds else 0.0
            ret_dict["acc_norm"] = acc_norm
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
    """
    !!@@##@@!! -- Example 0 (5-shot)
    Making a sandwich: Several food items and dishes are laid out on a table. Meat product and other items are used to create a sandwich. Then a bento shaper is used to create an image in the sandwich.

    Throwing darts: A man in a black vest is standing in a room. He throws darts at a dart board on the wall. A woman stands next to him watching.

    Food and Entertaining: How to make money by having a house party. Find a couple friends to help plan the party. Planning a killer house party is a lot easier when you've got accomplices to spread the time and responsibilities out to. Talk to your friends, and see who is interested in helping. This will make things a lot easier on you. Consider the following : Who has what responsibilities.

    Home and Garden: How to trim trees. Wear safety goggles, helmet, and purchase a step ladder. Safety goggles and a helmet or hard hat will protect your head and eyes as you prune adult trees. You may also need a small step ladder to reach higher branches. However, if the branch is located high in the air and requires an extension ladder, consult a professional instead of trying to do it yourself
    . Purchase this equipment online or at a hardware store.

    Getting a haircut: The man working in the salon cuts off the woman's long hair and puts it in a bag. The man combs and cuts the woman's hair as she sits.

    Roof shingle removal: A man is sitting on a roof. He
    """
    def __init__(self, *args, **kwargs) -> None:
        # Super task classes are not passed any arguments by the harness but we do that here just for future proofing
        super().__init__(*args, **kwargs)
        self.EXAMPLE_DELIMITER: str = '\n\n'
        self.QA_DELIMITER = ''
        self.ANSWER_HINT = ''
        self.HINT_ANSWER_DELIMITER = ' '
        self.ANSWER_HINT_SENTS = []
        self.verify_config()

    def _process_doc(self, doc):
        out_doc = SegmentedSample(super()._process_doc(doc), task=self)
        # Segments (including hints) so that they may be individually encoded (e.g 'Question: <question text>')
        out_doc['segments'] = [out_doc['query']]
        # out_doc['choices'] is already set by superclass
        # Indices of one or more correct targets from out_doc['choices']
        out_doc['gold_indices'] = [out_doc['gold']]
        return self._prepare_doc(out_doc)


class WebQsDist(DistEncTaskMixin, webqs.WebQs):
    """
    !!@@##@@!! -- Example 0 (5-shot)
    Question: who is lamar odom married too?
    Answer: Khloé Kardashian

    Question: what do they speak in iran?
    Answer: Turkmen Language

    Question: what awards has louis sachar won?
    Answer: National Book Award for Young People's Literature

    Question: where do most of the people in egypt live?
    Answer: Cairo

    Question: who is the speaker of the house of representatives currently?
    Answer: Nancy Pelosi

    Question: what is the name of justin bieber brother?
    Answer:
    """
    def __init__(self, *args, **kwargs) -> None:
        # Super task classes are not passed any arguments by the harness but we do that here just for future proofing
        super().__init__(*args, **kwargs)
        self.EXAMPLE_DELIMITER: str = '\n\n'
        self.QA_DELIMITER = '\n'
        self.QUESTION_HINT = 'Question:'
        self.ANSWER_HINT = 'Answer:'
        self.HINT_QUESTION_DELIMITER = ' '
        self.HINT_ANSWER_DELIMITER = ' '
        self.ANSWER_HINT_SENTS = [self.ANSWER_HINT]
        self.QUESTION_HINT_SENTS = [self.QUESTION_HINT]
        self.verify_config()
        # Since all choices are gold targets in WebQ, only task-type == 'gen' makes sense here.
        assert self.TASK_TYPE == 'gen'

    def test_docs(self):
        return map(self._process_doc, super().test_docs())

    def training_docs(self):
        return map(self._process_doc, super().training_docs())

    def _process_doc(self, doc):
        out_doc = SegmentedSample(doc, task=self)
        # Extract all hints so that they may be optionally individually encoded without text
        out_doc['question_hint'] = self.QUESTION_HINT
        out_doc['answer_hint'] = self.ANSWER_HINT
        # Question / Context segments (including hints) so that they may be individually encoded (e.g 'Question: <question text>')
        question = self.QUESTION_HINT + self.HINT_QUESTION_DELIMITER + out_doc['question']
        out_doc['segments'] = [question]
        out_doc['choices'] = doc['answers']
        # All out_doc['choices'] are gold targets
        out_doc['gold_indices'] = list(range(len(out_doc['choices'])))
        out_doc['gold'] = None  # Indicates we have more than one possible targets
        return self._prepare_doc(out_doc)

    def process_results(self, doc: SegmentedSample, results):
        """All choices in the test set are gold targets"""
        metrics = super().process_results(doc, results)
        # All choices are legitimate targets, therefore acc should be 1.0
        assert metrics['acc'] == metrics['acc_norm'] == 1.0
        return {
            # Exact match is the only accuracy metric for webqs.
            'acc': metrics['em']
        }


# class DistEncTaskMixin2(DistEncTaskMixin):
#     """
#     Specializtion of DistEncTaskMixin for cases where there are multiple contexts instead of multiple targets.
#     This class is still under construction.
#     """

#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#     def _make_contextlist(self, doc: SegmentedSample, fewshotex: List[SegmentedSample],
#                           description: Optional[SegmentedSample]) -> List[List[SegmentedSample]]:
#         """Return multiple context lists instead of one"""
#         description_list = [] if description is None else [description]
#         contexts = []
#         for ctx_choice in doc['ctx_choices']:
#             _doc = doc.copy()
#             _doc['segments'] = [ctx_choice]
#             context_list = description_list + [
#                 self._make_fewshotex(example) for example in fewshotex] + [
#                 self._make_fewshot_query(self._remove_label(_doc))]

#             if self.ENCODING_SCHEME in ['concat_all_examples', 'merge_all_segments', 'cross_encoding']:
#                 # Merge all samples into one
#                 context_list = [self._merge_fewshotex(_doc, context_list)]
#             contexts.append(context_list)
#         return contexts


# class Wsc273Dist(DistEncTaskMixin2, wsc273.WinogradSchemaChallenge273):
#     """This class is still under construction."""
#     def __init__(self, *args, **kwargs) -> None:
#         # Super task classes are not passed any arguments by the harness but we do that here just for future proofing
#         super().__init__(*args, **kwargs)
#         self.verify_config()

#     def _process_doc(self, doc):
#         doc = SegmentedSample(super()._process_doc(doc), task=self)
#         doc['segments'] = [self.doc_to_text(doc)]
#         doc['choices'] = [self.partial_target(doc)]
#         doc['gold_indices'] = [0]
#         doc['ctx_choices'] = [self.partial_context(doc, option) for option in doc['options']]
#         doc['ctx_gold_indices'] = [doc['label']]
