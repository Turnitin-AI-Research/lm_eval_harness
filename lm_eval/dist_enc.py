"""Utilities for distributed encoding"""
from typing import Dict, List


class SegmentedSample(Dict):
    """Segmented Sample class that enables empty instantiation and verification"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key in ['hints', 'segments', 'gold_options', 'choices']:
            if key not in self:
                self[key] = []
            else:
                assert isinstance(self[key], List)
        if 'query' not in self:
            self['query'] = ''
        else:
            assert isinstance(self['query'], str)
        if 'gold' not in self:
            self['gold'] = None
        else:
            assert isinstance(self['gold'], int)


class DistEncMixin:
    """Mixin for Distributed Encoding Task"""
    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
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
        :returns: str
            The fewshot context.
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
        description = [] if not description else [SegmentedSample(hints=[description], segments=[description])]

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
        return description + fewshotex
