"""
HellaSwag: Can a Machine Really Finish Your Sentence?
https://arxiv.org/pdf/1905.07830.pdf

Hellaswag is a commonsense inference challenge dataset. Though its questions are
trivial for humans (>95% accuracy), state-of-the-art models struggle (<48%). This is
achieved via Adversarial Filtering (AF), a data collection paradigm wherein a
series of discriminators iteratively select an adversarial set of machine-generated
wrong answers. AF proves to be surprisingly robust. The key insight is to scale up
the length and complexity of the dataset examples towards a critical 'Goldilocks'
zone wherein generated text is ridiculous to humans, yet often misclassified by
state-of-the-art models.

Homepage: https://rowanzellers.com/hellaswag/
"""
import re
from lm_eval.base import MultipleChoiceTask
from lm_eval.dist_enc import DistEncTaskMixin, SegmentedSample


_CITATION = """
@inproceedings{zellers2019hellaswag,
    title={HellaSwag: Can a Machine Really Finish Your Sentence?},
    author={Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
    booktitle ={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
    year={2019}
}
"""


class HellaSwag(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "hellaswag"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": self.preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [self.preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    @classmethod
    def preprocess(cls, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]


class HellaSwagDist(DistEncTaskMixin, HellaSwag):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.ENCODING_SCHEME: str = kwargs.get('encoding_scheme', 'concat_all_examples')
        self.SEGMENT_DELIMITER: str = '\n'
        self.ANSWER_DELIMITER: str = ' '
        self.verify_config()
        self.task_args = kwargs

    def _process_doc(self, doc):
        out_doc = SegmentedSample(super()._process_doc(doc), task=self)
        # Extract all hints so that they may be optionally individually encoded without text
        out_doc['hints'] = []
        # Segments (including hints) so that they may be individually encoded (e.g 'Question: <question text>')
        out_doc['segments'] = [out_doc['query']]
        # Indices of one or more correct targets from out_doc['choices']
        out_doc['gold_indices'] = [out_doc['gold']]
        return self.process_segments(out_doc)

    def __repr__(self) -> str:
        return super().__repr__() + (f', {self.task_args}' if self.task_args else '')
