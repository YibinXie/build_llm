import inspect
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import torch

from lib.dataset.tokenizer import Tokenizer
from lib.layers.pos_enc import precompute_freqs_cis
from lib.infer.sample import *
from lib.layers.transformer import *


class Transformer(nn.Module):
    def __init__(self, params: Params):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = nn.ModuleList([Block(params) for _ in range(params.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # lm_head
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope
        )

    def forward_inference(self, tokens: torch.Tensor, start_pos: int):
        # for use during inference
        bs, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seq_len]

        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float('-inf'), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            """
            tensor([[0., -inf, -inf, -inf],
                    [0., 0., -inf, -inf],
                    [0., 0., 0., -inf],
                    [0., 0., 0., 0.]])
            when performing kv-cacheing, we compute the attention scores only for the new sequence.
            Thus, the matrix of scores is of size (seq_len, cache_len + seql_len), and the only masked
            entries are (i, j) for j > cache_len + i, since row i corresponds to token cache_len + i.
            """
            mask = torch.hstack(
                [torch.zeros((seq_len, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()  # float16 -> float32 for numerical stability
        return output

    def forward_loss(self, inputs: torch.Tensor, targets: torch.Tensor, ignore_index=-100):
        # for use during training
        bs, seq_len = inputs.shape
        h = self.tok_embeddings(inputs)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seq_len]

        mask = torch.full((seq_len, seq_len), float('-inf'), device=inputs.device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.type_as(h)

        start_pos = -1
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        logits = self.output(h).float()
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            targets.view(-1),
            reduction='mean',
            ignore_index=ignore_index
        )
        return loss

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None, start_pos: int = 0, ignore_index=-100):
        if targets is not None:
            return self.forward_loss(inputs, targets, ignore_index)
        else:
            return self.forward_inference(inputs, start_pos)

    def configure_optimizers(self,
                             learning_rate: float = 3e-4,
                             weight_decay: float = 0.0,
                             betas=(0.9, 0.97),
                             device_type: str = 'cuda',
                             master_process: bool = True
                             ):
        train_params = self.parameters()

        # train_params = []

        # finetune_type = 'all'
        # if finetune_type == 'rmsnorm':
        #     for name, param in self.named_parameters():
        #         if 'norm' in name:
        #             train_params.append(param)
        # elif finetune_type == 'all':
        #     for param in self.parameters():
        #         train_params.append(param)
        # elif finetune_type == 'all_no_pos_and_head':
        #     n, m = 0, 0
        #     for name, param in self.named_parameters():
        #         if name == 'output.weight':
        #             # do not include lm_head
        #             n += 1
        #             continue
        #         elif name == 'tok_emb.weight':
        #             # do not include tok_emb
        #             m += 1
        #             # Because token embeddings are in the very last part of the chain rule, we don't need
        #             # to calculate the gradients when we aren't updating them.
        #             param.requires_grad = False
        #         else:
        #             train_params.append(param)
        #     assert n == 1, 'did not find output.weight'
        #     assert m == 1, 'did not find tok_emb.weight'
        if master_process:
            print('number of parameters: ', sum(p.numel() for p in self.parameters()))
            # print('number of trainable parameters totally:', sum(p.numel() for p in train_params if p.requires_grad))
            # print('number of trainable parameters trained this time with grad:',
            #       sum(p.numel() for p in train_params if p.requires_grad))
            # print('number of trainable parameters trained this time all:', sum(p.numel() for p in train_params))

        use_fused = device_type == 'cuda'
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(train_params, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    # def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process):
    #     # start with all of the candidate parameters (that require grad)
    #     param_dict = {pn: p for pn, p in self.named_parameters()}
    #     param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    #     # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    #     # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    #     decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    #     nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    #     optim_groups = [
    #         {'params': decay_params, 'weight_decay': weight_decay},
    #         {'params': nodecay_params, 'weight_decay': 0.0}
    #     ]
    #     num_decay_params = sum(p.numel() for p in decay_params)
    #     num_nodecay_params = sum(p.numel() for p in nodecay_params)
    #     if master_process:
    #         print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    #         print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    #     # Create AdamW optimizer and use the fused version if it is available
    #     fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    #     use_fused = fused_available and device_type == "cuda"
    #     if master_process:
    #         print(f"using fused AdamW: {use_fused}")
    #     optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    #     return optimizer


class Llama:
    @staticmethod
    def build(params: Params) -> 'Llama':
        max_seq_len = params.max_seq_len
        ckpt_dir = params.ckpt_dir
        tokenizer_path = params.tokenizer_path
        seed = params.seed
        model_parallel_size = params.model_parallel_size

        assert 1 <= max_seq_len <= 8192, f'max_seq_len: {max_seq_len} should be in [1, 8192]'
        assert os.path.isdir(ckpt_dir), f'ckpt_dir: {ckpt_dir} does not exist'
        assert os.path.isfile(tokenizer_path), f'tokenizer_path: {tokenizer_path} does not exist'

        local_rank = 0
        # torch.cuda.set_device(local_rank)
        torch.manual_seed(seed)  # seed must be the same in all processes todo: why?

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob('*.pth'))
        assert len(checkpoints) > 0, f'no checkpoints found in {ckpt_dir}'
        assert model_parallel_size == len(
            checkpoints), f'ckpt_dir: {ckpt_dir} should have {model_parallel_size} checkpoints'
        ckpt_path = checkpoints[0]
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        print(f'Loaded checkpoint from {ckpt_path}')
        with open(Path(ckpt_dir) / 'params.json', 'r') as f:
            params_loaded = json.loads(f.read())

        params.update(
            **params_loaded
        )
        print(f'Loaded model args from {ckpt_dir}/params.json')

        tokenizer = Tokenizer(tokenizer_path)
        assert params.vocab_size == tokenizer.n_words, \
            f'vocab_size: {params.vocab_size} should be equal to tokenizer.n_words: {tokenizer.n_words}'
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            # torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
            torch.set_default_dtype(torch.bfloat16)
        else:
            # torch.set_default_tensor_type(torch.BFloat16TensorTensor)
            pass
        model = Transformer(params)
        model.load_state_dict(checkpoint, strict=True)
        print(f'Loaded model from {ckpt_path} in {time.time() - start_time:.2f} seconds')
        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
            self,
            prompt_tokens: List[List[int]],
            sample_rng: torch.Generator,
            max_gen_len: int,
            temperature: float = 0.6,
            top_p: float = 0.9,
            echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
        max_gen_len (int): Maximum length of the generated text sequence.
        temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
        top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
        logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
        echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.
        """
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        # install KV cache in all the Attention layers
        for block in self.model.layers:
            layer_dtype = block.attention.wq.weight.dtype
            layer_device = block.attention.wq.weight.device
            block.attention.cache = KVCache(
                batch_size=bsz,
                seq_length=total_len,
                n_kv_heads=params.n_kv_heads,
                head_dim=params.dim // params.n_heads,
                dtype=layer_dtype,
                device=layer_device,
            )

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cpu")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cpu")

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cpu")
        input_text_mask = tokens != pad_id

        if min_prompt_len == total_len:
            logits = self.model.forward_inference(tokens, prev_pos)

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens))

        for cur_pos in range(min_prompt_len, total_len):
            # get the logits for the next token in all the batch rows
            logits = self.model.forward_inference(tokens[:, prev_pos:cur_pos], prev_pos)
            # sample the next token
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p, sample_rng)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start: len(prompt_tokens[i]) + max_gen_len]
            # cut to after eos tok if any
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                except ValueError:
                    pass
            out_tokens.append(toks)

        # clean up the KV cache in all the layers
        for block in self.model.layers:
            block.attention.cache = None

        return out_tokens

    def text_completion(
            self,
            prompts: List[str],
            sample_rng: torch.Generator,
            temperature: float = 0.6,
            top_p: float = 0.9,
            max_gen_len: Optional[int] = None,
            echo: bool = False,
    ):
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        # encode the (string) prompts to tokens
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        # generate the completions in tokens space
        generation_tokens = self.generate(
            prompt_tokens=prompt_tokens,
            sample_rng=sample_rng,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            echo=echo,
        )
        # decode the completions back to strings
        completions = [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]
        return completions
