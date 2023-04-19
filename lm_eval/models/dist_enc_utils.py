from typing import Union, Optional, Tuple
import collections
from functools import wraps
import torch
from torch import Tensor
from torch.profiler import profile, record_function, ProfilerActivity
import transformers


BaseModelType = Union[transformers.GPT2LMHeadModel,
                      transformers.GPTNeoForCausalLM,
                      transformers.GPTJForCausalLM,
                      transformers.GPTNeoXForCausalLM,
                      transformers.BloomForCausalLM,
                      transformers.MT5ForConditionalGeneration,
                      transformers.T5ForConditionalGeneration,
                      ]
TokenizerType = Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]


class PropertyDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


def trace_handler(p):
    """Called from Pytorch profiler, to print / save profile traces"""
    print(p.key_averages().table(sort_by="cuda_memory_usage"))
    print(p.key_averages(group_by_stack_n=5).table(sort_by="cuda_memory_usage"))
    print(p.key_averages(group_by_input_shape=True).table(sort_by="cuda_memory_usage"))
    print(p.key_averages(group_by_stack_n=5).table(sort_by="cuda_memory_usage", row_limit=10))
    # print(p.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_memory_usage"))
    # print(p.key_averages(group_by_stack_n=5).table(sort_by="cpu_memory_usage"))
    p.export_chrome_trace("./trace_log/pytorch_trace_" + str(p.step_num) + ".json")


def dummy_decorator(nameOrfunc):
    """Dummy decorator"""
    def echo(func):
        return func
    if callable(nameOrfunc):
        return nameOrfunc
    else:
        return echo


# instrument = dummy_decorator
def instrument(nameOrfunc):
    """Decorator that runs a func under record_function context"""
    name, func = None, None

    def decorator(func):
        @wraps(name or func.__name__)
        def wrapper(*args, **kwargs):
            with record_function(name or func.__name__):
                return func(*args, **kwargs)
        return wrapper
    if callable(nameOrfunc):
        func = nameOrfunc
        return decorator(func)
    else:
        name = nameOrfunc
        return decorator


def do_profile(func):
    """Decorator sets up pytorch profiling."""
    @wraps(func.__name__)
    def wrapper(*args, **kwargs):
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True, profile_memory=True, with_stack=True,
                     schedule=torch.profiler.schedule(wait=0, warmup=0, active=1000),
                     #  on_trace_ready=trace_handler
                     on_trace_ready=torch.profiler.tensorboard_trace_handler('./tb_log/1')
                     ) as prof:
            return func(*args, **kwargs, prof=prof)
    return wrapper


def get_max_length(model: BaseModelType,
                   tokenizer: TokenizerType) -> int:
    MAX_MAX_LEN = 10240
    if isinstance(model, transformers.GPT2LMHeadModel):
        return model.config.n_ctx
    elif isinstance(model, (transformers.GPTNeoForCausalLM,
                            transformers.GPTJForCausalLM,
                            transformers.GPTNeoXForCausalLM)):
        return model.config.max_position_embeddings
    elif isinstance(model, (transformers.MT5ForConditionalGeneration,
                            transformers.T5ForConditionalGeneration)):
        return MAX_MAX_LEN  # self.tokenizer.max_len_single_sentence
    elif isinstance(model, transformers.BloomForCausalLM):
        return min(tokenizer.model_max_length, MAX_MAX_LEN)


class ParameterlessAttentionDecoder(torch.nn.Module):
    """
    Parameterless attention implementation. Single head and no Wq, Wk, Wv and Wo matrices.
    """

    def __init__(self,
                 base_model: BaseModelType,
                 single_layer: bool = True) -> None:
        super().__init__()
        self.num_layers = 1 if single_layer else self.num_decoder_layers(base_model)
        self._is_enc_dec = self.is_enc_dec(base_model)
        self.input_embeddings = base_model.get_input_embeddings()
        # self.input_embeddings.requires_grad_(False)
        self.output_embeddings = base_model.get_output_embeddings()
        # self.output_embeddings.requires_grad_(False)
        if isinstance(base_model, (transformers.MT5ForConditionalGeneration,
                                   transformers.T5ForConditionalGeneration)):
            self._base_model = base_model

    @staticmethod
    def is_enc_dec(model: BaseModelType) -> bool:
        """Return True if this is an encoder-decoder model like T5."""
        if isinstance(model, (transformers.T5ForConditionalGeneration,
                              transformers.MT5ForConditionalGeneration)):
            return True
        elif isinstance(model, (transformers.GPT2LMHeadModel,
                                transformers.GPTNeoForCausalLM,
                                transformers.GPTJForCausalLM,
                                transformers.GPTNeoXForCausalLM,
                                transformers.BloomForCausalLM,
                                transformers.MT5ForConditionalGeneration,
                                transformers.T5ForConditionalGeneration,
                                )):
            return False
        else:
            raise NotImplementedError()

    @staticmethod
    def num_decoder_layers(base_model: BaseModelType):
        if hasattr(base_model.config, 'attention_layers'):
            return len(base_model.config.attention_layers)
        elif hasattr(base_model.config, 'num_decoder_layers'):
            return base_model.config.num_decoder_layers
        else:
            raise NotImplementedError()

    @instrument
    def forward(self, *,
                input_ids: Optional[torch.LongTensor] = None,  # compatible with HF GPT like base_model
                inputs_embeds: Optional[torch.FloatTensor] = None,  # compatible with HF GPT like base_model
                encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,  # compatible with HF T5 base_model
                attention_mask: torch.LongTensor,
                labels: Optional[torch.LongTensor] = None,  # compatible with HF T5 base_model
                decoder_attention_mask: Optional[torch.LongTensor] = None,  # compatible with HF T5 base_model
                output_hidden_states: bool = True,  # Only for compatibility with HF. ignored
                return_dict: bool = True):  # Only for compatibility with HF. ignored
        if labels is None:  # gpt like base_model
            assert all(var is None for var in (encoder_outputs, decoder_attention_mask))
            if input_ids is not None:
                assert inputs_embeds is None
                inputs_embeds = self.input_embeddings(input_ids)  # (N, S, D)
            else:
                assert inputs_embeds is not None  # (N, S, D)
            Q = inputs_embeds
            batchMaskQ = attention_mask  # (N, S)
            M = None
            batchMaskM = None
        else:  # encoder-decoder model
            assert all(var is None for var in (inputs_embeds, input_ids))
            assert decoder_attention_mask is not None
            assert self._is_enc_dec, f'Labels are only supported for T5 like models'
            M = encoder_outputs[0]  # type: ignore (N, Sm, D)
            batchMaskM = attention_mask
            decoder_input_ids = self._base_model.prepare_decoder_input_ids_from_labels(labels)
            Q = self.input_embeddings(decoder_input_ids)  # type: ignore
            batchMaskQ = decoder_attention_mask

        att_out = self.soft_cluster(Q=Q, M=M, batchMaskQ=batchMaskQ, batchMaskM=batchMaskM)
        out_embeds: torch.FloatTensor = self.output_embeddings.weight  # type: ignore # (V, D)
        logits = torch.matmul(att_out['hidden'], out_embeds.T)  # (N, Sq, V)
        return PropertyDict({'logits': logits,
                             #  'hidden_states': [att_out['hidden']],
                             #  'attention_weights': att_out['attention_weights']
                             })

    @ staticmethod
    @instrument
    def soft_cluster(*,
                     Q: torch.FloatTensor,
                     batchMaskQ: torch.LongTensor,
                     M: Optional[Tensor] = None,
                     batchMaskM: Optional[torch.LongTensor] = None):
        """
        Causal linear attention layer.
        :param Q
            decoder input vectors of shape (N, Sq, D) where D is the vector size. Causal self-attention
            is applied on this sequence.
        :param batchMaskQ
            batch mask for decoder input vectors of shape (N, Sq). value 1 implies True, 0 implies False
        :param M
            encoded memory vectors of shape (N, Sm, D) on which cross-attention is applied
        :param batchMaskM
            batch mask for encoded vectors of shape (N, Sm). value 1 implies True, 0 implies False
        :return Tensor[N, Sq, D]
            cross & self attention aggregated embeddings
        """
        N, Sq = Q.shape[0], Q.shape[1]
        maskQ = torch.ones((Sq, Sq), device=batchMaskQ.device).tril().unsqueeze(0) * \
            batchMaskQ.unsqueeze(1)  # (N, Sq, Sq)
        if M is not None:
            assert M.shape[0] == N
            Sm = M.shape[1]
            assert batchMaskM is not None
            maskM = torch.ones((1, Sq, Sm), device=batchMaskM.device) * batchMaskM.unsqueeze(1)  # (N, Sq, Sm)
            maskMQ = torch.cat((maskM, maskQ), dim=-1)  # (N, Sq, Sm+Sq)
            M2 = torch.cat((M, Q), dim=1)  # (N, Sm+Sq, D)
        else:
            maskMQ = maskQ  # (N, Sq, Sq)
            M2 = Q  # (N, Sq, D)
        maskMQ = maskMQ.to(dtype=torch.bool, device=Q.device)
        dot_product = torch.matmul(Q, M2.permute(0, 2, 1))  # (N, Sq, Sm+Sq)
        dot_product = torch.where(maskMQ, dot_product, -torch.inf)  # (N, Sq, Sm+Sq)
        attention_weights = torch.nn.functional.softmax(dot_product, dim=-1)  # (N, Sq, Sm+Sq)
        attention_weights = attention_weights * batchMaskQ.unsqueeze(-1)  # (N, Sq, Sm+Sq)
        hidden = torch.matmul(attention_weights, M2)  # (N, Sq, D)
        return PropertyDict(attention_weights=attention_weights,
                            # M2=M2, maskM=maskM, maskQ=maskQ, maskMQ=maskMQ,
                            hidden=hidden  # (N, Sq, D)
                            )

    @ staticmethod
    def _test_soft_cluster():
        """ Test soft-cluster function """
        D = 128
        N = 2
        Sq, Sm = 6, 12
        zeroedQ, zeroedM = 2, 4
        M = torch.rand(N, Sm, D)
        maskM = torch.tensor([[1] * Sm, [1] * (Sm - (zeroedM)) + [0] * zeroedM])
        Q = torch.rand(N, Sq, D)
        maskQ = torch.tensor([[1] * (Sq - zeroedQ) + [0] * zeroedQ, [1] * Sq])
        print(f'M.shape={M.shape}, Q.shape={Q.shape}')
        out = ParameterlessAttentionDecoder.soft_cluster(Q=Q, batchMaskQ=maskQ, M=M, batchMaskM=maskM)
        assert (N, Sq, D) == out['hidden'].shape
        assert (N, Sq, Sq + Sm) == out['attention_weights'].shape
        (out['attention_weights'].sum(dim=-1))
        assert torch.allclose(out['attention_weights'].sum(dim=-1), maskQ.to(dtype=torch.float32))
        # check for zero masked columns
        assert torch.where(torch.cat((maskM, maskQ), dim=1).unsqueeze(
            1).to(torch.bool), 0., out['attention_weights']).sum() == 0
        # check for zero masked rows
        assert torch.where(maskQ.unsqueeze(-1).to(torch.bool), 0., out['attention_weights']).sum() == 0.

    def _test_enc_dec(self):
        """ Test forward function in encoder-decoder mode. """
        V, D = self._base_model.get_input_embeddings().weight.shape
        N = 2
        Sq, Sm = 6, 12
        zeroedQ, zeroedM = 2, 4
        mem_ids = torch.randint(1, V, (N, Sm))
        mem_embeds = self._base_model.get_input_embeddings()(mem_ids)
        mem_attention_mask = torch.tensor([[1] * Sm, [1] * (Sm - (zeroedM)) + [0] * zeroedM])
        query_ids = torch.randint(1, V, (N, Sq))
        query_attention_mask = torch.tensor([[1] * (Sq - zeroedQ) + [0] * zeroedQ, [1] * Sq])
        print(f'mem_embeds.shape={mem_embeds.shape}, query_ids.shape={query_ids.shape}')
        with torch.no_grad():
            out = self(encoder_outputs=(mem_embeds,), attention_mask=mem_attention_mask,
                       labels=query_ids, decoder_attention_mask=query_attention_mask)
            # out = decoder( attention_mask=attention_mask, input_ids=labels)
        assert 1 == len(out['hidden_states'])
        assert (N, Sq, D) == out['hidden_states'][0].shape
        assert (N, Sq, V) == out['logits'].shape
        assert (N, Sq, Sq + Sm) == out['attention_weights'].shape
        # verify that sum-of-weights == 1, except at masked query positions where it should be 0
        assert torch.allclose(out['attention_weights'].sum(dim=-1), query_attention_mask.to(torch.float32))
        # check for zero masked columns
        assert torch.where(torch.cat((mem_attention_mask, query_attention_mask), dim=1).unsqueeze(1).to(torch.bool),
                           0., out['attention_weights']).sum() == 0.
        # check for zero masked columns
        assert torch.where(query_attention_mask.unsqueeze(-1).to(torch.bool),
                           0., out['attention_weights']).sum() == 0.
        # Check that masked positions get zero embedding vectors
        assert torch.where(query_attention_mask.unsqueeze(-1).to(torch.bool), 0., out['hidden_states'][0]).sum() == 0.
        # Zero embedding vectors should have zero logits too
        assert out['logits'][0, (Sq - 2):].sum() == 0.

    def _test_dec_1(self):
        V, D = self._base_model.get_input_embeddings().weight.shape
        N = 2
        Sq, Sm = 6, 12
        zeroedQ, zeroedM = 2, 4
        query_ids = torch.randint(1, V, (N, Sq))
        query_attention_mask = torch.tensor([[1] * (Sq - zeroedQ) + [0] * zeroedQ, [1] * Sq])
        print(f'query_ids.shape={query_ids.shape}')
        with torch.no_grad():
            out = self(input_ids=query_ids, attention_mask=query_attention_mask)
        assert 1 == len(out['hidden_states'])
        assert (N, Sq, D) == out['hidden_states'][0].shape
        assert (N, Sq, V) == out['logits'].shape
        assert (N, Sq, Sq) == out['attention_weights'].shape
        # verify that sum-of-weights == 1, except at masked query positions where it should be 0
        assert torch.allclose(out['attention_weights'].sum(dim=-1), query_attention_mask.to(torch.float32))
        assert torch.where(query_attention_mask.unsqueeze(-1).to(torch.bool),
                           0., out['attention_weights']).sum() == 0.
        # Check that masked positions get zero embedding vectors
        assert torch.where(query_attention_mask.unsqueeze(-1).to(torch.bool), 0., out['hidden_states'][0]).sum() == 0.
        # Zero embedding vectors should have zero logits too
        assert out['logits'][0, (Sq - 2):].sum() == 0.

    def _test_dec_2(self):
        V, D = self._base_model.get_input_embeddings().weight.shape
        N = 2
        Sq, Sm = 6, 12
        zeroedQ, zeroedM = 2, 4
        query_embeds = self._base_model.get_input_embeddings()(torch.randint(1, V, (N, Sq)))
        query_attention_mask = torch.tensor([[1] * (Sq - zeroedQ) + [0] * zeroedQ, [1] * Sq])
        print(f'query_embeds.shape={query_embeds.shape}')
        with torch.no_grad():
            out = self(inputs_embeds=query_embeds, attention_mask=query_attention_mask)
        assert 1 == len(out['hidden_states'])
        assert (N, Sq, D) == out['hidden_states'][0].shape
        assert (N, Sq, V) == out['logits'].shape
        assert (N, Sq, Sq) == out['attention_weights'].shape
        # verify that sum-of-weights == 1, except at masked query positions where it should be 0
        assert torch.allclose(out['attention_weights'].sum(dim=-1), query_attention_mask.to(torch.float32))
        assert torch.where(query_attention_mask.unsqueeze(-1).to(torch.bool),
                           0., out['attention_weights']).sum() == 0.
        # Check that masked positions get zero embedding vectors
        assert torch.where(query_attention_mask.unsqueeze(-1).to(torch.bool), 0., out['hidden_states'][0]).sum() == 0.
        # Zero embedding vectors should have zero logits too
        assert out['logits'][0, (Sq - 2):].sum() == 0.
