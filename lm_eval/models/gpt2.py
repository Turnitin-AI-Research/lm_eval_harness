from typing import List, Optional
import transformers
import torch
from lm_eval.base import BaseLM
from lm_eval.models.dist_enc_model import DistEncSimMixin, DistEncGenMixin

MAX_MAX_LEN = 10240


def str_to_bool(arg: Optional[str]) -> bool:
    """Convert parameter string to bool"""
    if arg is None:
        return False
    arg = arg.lower()
    assert arg in ['true', 'false']
    return arg == 'true'


class HFLM(BaseLM):
    def __init__(
        self,
        device="cuda",
        pretrained="gpt2",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        PARALLELIZE: Optional[str] = None
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)
        cache_dir = None

        self.PARALLELIZE: bool = str_to_bool(PARALLELIZE)
        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            if not self.PARALLELIZE:
                print(f"Using device '{device}'")
            else:
                print("setting device_map='auto'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        if self.PARALLELIZE:
            assert device in [0, '0', 'cuda:0'], f'Device ({device}) must be set to "0" with PARALLELIZE model-arg'

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        try:
            self.gpt2 = transformers.AutoModelForCausalLM.from_pretrained(
                pretrained,
                revision=revision + ("/" + subfolder if subfolder is not None else ""),
                device_map='auto' if self.PARALLELIZE else None,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
        except ValueError:
            self.gpt2 = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                pretrained,
                device_map='balanced' if self.PARALLELIZE else None,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            if self.PARALLELIZE:
                # self.gpt2.parallelize()
                self._device = 0
        if not self.PARALLELIZE:
            self.gpt2 = self.gpt2.to(self.device)
        self.gpt2.eval()

        # pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
        if subfolder is not None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                pretrained if tokenizer is None else tokenizer,
                revision=revision,
                subfolder=subfolder,
                cache_dir=cache_dir
            )
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                pretrained if tokenizer is None else tokenizer,
                revision=revision,
                cache_dir=cache_dir
            )

        assert isinstance(
            self.tokenizer,
            (
                transformers.GPT2Tokenizer,
                transformers.GPT2TokenizerFast,
                transformers.T5Tokenizer,
                transformers.T5TokenizerFast,
                transformers.BloomTokenizerFast,
                transformers.GPTNeoXTokenizerFast
            ),
        ), f"this tokenizer ({type(self.tokenizer)}) has not been checked for compatibility yet!"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vocab_size = self.tokenizer.vocab_size

        if isinstance(
            self.tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)
        ):
            assert self.tokenizer.encode("hello\n\nhello") == [
                31373,
                198,
                198,
                31373,
            ], self.tokenizer.encode("hello\n\nhello")

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        # TODO: fix multi-gpu
        # gpus = torch.cuda.device_count()
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx

        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.gpt2.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.gpt2(inps)[0][:, :, :50257]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.gpt2.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )


# for backwards compatibility
GPT2LM = HFLM


class DistributedSim(DistEncSimMixin, HFLM):
    """Wrapper around HFLM that perfoms distributed encoding instead of cross-encoding"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verify_config()


class DistributedGen(DistEncGenMixin, HFLM):
    """Wrapper around HFLM that perfoms distributed encoding instead of cross-encoding"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verify_config()
