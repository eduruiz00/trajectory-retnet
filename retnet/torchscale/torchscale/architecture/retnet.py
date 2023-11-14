# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn import checkpoint_wrapper, wrap
from trajectory.models.ein import EinLinear
import pdb

from torchscale.architecture.utils import init_bert_params
from torchscale.component.droppath import DropPath
from torchscale.component.feedforward_network import make_experts
from torchscale.component.gate_linear_unit import GLU
from torchscale.component.multiscale_retention import MultiScaleRetention
from torchscale.component.xmoe.moe_layer import MOELayer
from torchscale.component.xmoe.routing import Top1Gate, Top2Gate
from torchscale.component.rms_norm import RMSNorm
    
    
class RetNetRelPos(nn.Module):
    def __init__(self, args):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, args.decoder_embed_dim // args.decoder_retention_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-5 - torch.arange(args.decoder_retention_heads, dtype=torch.float)))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        self.recurrent_chunk_size = args.recurrent_chunk_size
        
    def forward(self, slen, activate_recurrent=False, chunkwise_recurrent=False):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())
        elif chunkwise_recurrent:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])

            block_index = torch.arange(self.recurrent_chunk_size).to(self.decay)
            mask = torch.tril(torch.ones(self.recurrent_chunk_size, self.recurrent_chunk_size).to(self.decay))
            mask = torch.masked_fill(block_index[:, None] - block_index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            
            value_inner_decay = mask[:, -1] / mask[:, -1].sum(dim=-1, keepdim=True)
            value_inner_decay = value_inner_decay.unsqueeze(-1)
            scale = mask.sum(dim=-1, keepdim=True).sqrt()
            inner_mask = mask / scale

            cross_decay = torch.exp(self.decay * self.recurrent_chunk_size)
            query_inner_decay = torch.exp(self.decay[:, None] * (block_index + 1))
            query_inner_decay = query_inner_decay[:, :, None] / (scale / mask[:, -1].sum(dim=-1)[:, None, None])
            cross_decay = cross_decay[:, None, None]
            retention_rel_pos = ((sin, cos), (inner_mask, cross_decay, query_inner_decay, value_inner_decay))
        else:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])
            mask = torch.tril(torch.ones(slen, slen).to(self.decay))
            mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos

class DecoderLayer(nn.Module):
    def __init__(
        self,
        args,
        depth,
        is_moe_layer=False,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = torch.nn.Dropout(args.dropout)

        if args.drop_path_rate > 0:
            drop_path_prob = np.linspace(0, args.drop_path_rate, args.decoder_layers)[
                depth
            ]
            self.drop_path = DropPath(drop_path_prob)
        else:
            self.drop_path = None

        self.retention = self.build_retention(self.embed_dim, args)

        self.normalize_before = args.decoder_normalize_before

        self.retention_layer_norm = RMSNorm(self.embed_dim, eps=args.layernorm_eps)

        self.is_moe_layer = is_moe_layer
        self.ffn_dim = args.decoder_ffn_embed_dim

        if not self.is_moe_layer:
            self.ffn = self.build_ffn(
                self.embed_dim,
                self.args,
            )
        else:
            if args.moe_top1_expert:
                gate = Top1Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    use_fp32=args.moe_gating_use_fp32,
                    moe_eval_capacity_token_fraction=args.moe_eval_capacity_token_fraction,
                    use_xmoe=args.use_xmoe,
                )
            else:
                gate = Top2Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    args.moe_gating_use_fp32,
                    args.moe_second_expert_policy,
                    args.moe_normalize_gate_prob_before_dropping,
                    args.moe_eval_capacity_token_fraction,
                    use_xmoe=args.use_xmoe,
                )
            experts = make_experts(args, self.embed_dim, self.ffn_dim)
            self.moe_layer = MOELayer(gate, experts, args)

        self.final_layer_norm = RMSNorm(self.embed_dim, eps=args.layernorm_eps)

        if args.deepnorm:
            self.alpha = math.pow(2.0 * args.decoder_layers, 0.25)
        else:
            self.alpha = 1.0

    def build_ffn(self, embed_dim, args):
        return GLU(
            embed_dim,
            self.ffn_dim,
            args.activation_fn,
            args.dropout,
            args.activation_dropout,
        )

    def build_retention(self, embed_dim, args):
        return MultiScaleRetention(
            args,
            embed_dim,
            args.decoder_value_embed_dim,
            args.decoder_retention_heads,
        )

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(
        self,
        x,
        incremental_state=None,
        chunkwise_recurrent=False,
        retention_rel_pos=None,
    ):
        residual = x
        if self.normalize_before:
            x = self.retention_layer_norm(x)

        x = self.retention(
            x,
            incremental_state=incremental_state,
            rel_pos=retention_rel_pos,
            chunkwise_recurrent=chunkwise_recurrent,
        )
        x = self.dropout_module(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.retention_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        if not self.is_moe_layer:
            x = self.ffn(x)
            l_aux = None
        else:
            x, l_aux = self.moe_layer(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, l_aux


class RetNetDecoder(nn.Module):
    def __init__(
        self,
        config,
        # **kwargs
    ):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.stop_token = config.vocab_size * config.transition_dim
        self.block_size = config.block_size
        self.observation_dim = config.observation_dim

        self.action_dim = config.action_dim
        self.transition_dim = config.transition_dim
        self.action_weight = config.action_weight
        self.reward_weight = config.reward_weight
        self.value_weight = config.value_weight

        self.embed_dim = config.decoder_embed_dim

        self.dropout_module = torch.nn.Dropout(config.dropout)

        self.embed_scale = 1.0 if config.no_scale_embedding else math.sqrt(self.embed_dim)

        self.embed_tokens = nn.Embedding(config.vocab_size * config.transition_dim + 1, config.decoder_embed_dim)


        if (
            not config.no_output_layer
            and config.vocab_size > 0
        ):
            self.output_projection = self.build_output_projection(config)
        else:
            self.output_projection = None

        if config.layernorm_embedding:
            self.layernorm_embedding = RMSNorm(self.embed_dim, eps=config.layernorm_eps)
        else:
            self.layernorm_embedding = None

        self.layers = nn.ModuleList([])

        moe_freq = config.moe_freq
        for i in range(config.decoder_layers):
            is_moe_layer = moe_freq != 0 and (i + 1) % moe_freq == 0
            self.layers.append(
                self.build_decoder_layer(
                    config,
                    depth=i,
                    is_moe_layer=is_moe_layer,
                )
            )

        self.num_layers = len(self.layers)

        if config.decoder_normalize_before:
            self.layer_norm = RMSNorm(self.embed_dim, eps=config.layernorm_eps)
        else:
            self.layer_norm = None

        self.retnet_rel_pos = RetNetRelPos(config)
        self.chunkwise_recurrent = config.chunkwise_recurrent
        self.recurrent_chunk_size = config.recurrent_chunk_size
        

        if config.deepnorm:
            init_scale = math.pow(8.0 * config.decoder_layers, 0.25)
            for name, p in self.named_parameters():
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.div_(init_scale)

    def build_output_projection(
        self,
        config,
    ):
        output_projection = EinLinear(config.transition_dim, config.decoder_embed_dim, config.vocab_size + 1, bias=False)
        
        # if args.share_decoder_input_output_embed:
        #     output_projection = torch.nn.Linear(
        #         self.embed_tokens.weight.shape[1],
        #         self.embed_tokens.weight.shape[0],
        #         bias=False,
        #     )
        #     output_projection.weight = self.embed_tokens.weight
        # else:
        #     output_projection = torch.nn.Linear(
        #         args.decoder_embed_dim, args.vocab_size, bias=False
        #     )
        #     torch.nn.init.normal_(
        #         output_projection.weight, mean=0, std=args.decoder_embed_dim**-0.5
        #     )
        return output_projection

    def build_decoder_layer(
        self, args, depth, is_moe_layer=False
    ):
        layer = DecoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
        )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        if args.fsdp:
            layer = wrap(layer)
        return layer

    def forward_embedding(
        self,
        tokens,
        token_embedding=None,
        incremental_state=None,
    ):
        if incremental_state is not None and not self.is_first_step(incremental_state):
            tokens = tokens[:, -1:]

        if token_embedding is None:
            token_embedding = self.embed_tokens(tokens)

        x = embed = self.embed_scale * token_embedding

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        return x, embed

    def is_first_step(self, incremental_state):
        if incremental_state is None:
            return False
        return incremental_state.get("is_first_step", False)

    def offset_tokens(self, idx):
        _, t = idx.shape
        n_states = int(np.ceil(t / self.transition_dim))
        offsets = torch.arange(self.transition_dim) * self.vocab_size
        offsets = offsets.repeat(n_states).to(idx.device)
        offset_idx = idx + offsets[:t]
        offset_idx[idx == self.vocab_size] = self.stop_token
        return offset_idx

    def pad_to_full_observation(self, x, verify=False):
        b, t, _ = x.shape
        n_pad = (self.transition_dim - t % self.transition_dim) % self.transition_dim
        padding = torch.zeros(b, n_pad, self.embed_dim, device=x.device)
        ## [ B x T' x embedding_dim ]
        x_pad = torch.cat([x, padding], dim=1)
        ## [ (B * T' / transition_dim) x transition_dim x embedding_dim ]
        x_pad = x_pad.view(-1, self.transition_dim, self.embed_dim)
        if verify:
            self.verify(x, x_pad)
        return x_pad, n_pad

    def verify(self, x, x_pad):
        b, t, embedding_dim = x.shape
        n_states = int(np.ceil(t / self.transition_dim))
        inds = torch.arange(0, self.transition_dim).repeat(n_states)[:t]
        for i in range(self.transition_dim):
            x_ = x[:,inds == i]
            t_ = x_.shape[1]
            x_pad_ = x_pad[:,i].view(b, n_states, embedding_dim)[:,:t_]
            print(i, x_.shape, x_pad_.shape)
            try:
                assert (x_ == x_pad_).all()
            except:
                pdb.set_trace()

    def forward(
        self,
        input,
        targets=None,
        mask=None,
        incremental_state=None, # recurrent state (S_n in the paper)
        features_only=False,
        return_all_hiddens=False,
        token_embeddings=None,
        **kwargs
    ):
        b, t = input.size()
        offset_input = self.offset_tokens(input)
        # embed tokens
        x, _ = self.forward_embedding(
            offset_input, token_embeddings, incremental_state
        )
        is_first_step = self.is_first_step(incremental_state)

        if self.chunkwise_recurrent and input.size(1) % self.recurrent_chunk_size != 0:
            padding_len = self.recurrent_chunk_size - input.size(1) % self.recurrent_chunk_size
            slen = input.size(1) + padding_len
            x = F.pad(x, (0, 0, 0, padding_len))
        else:
            slen = input.size(1)
        # relative position
        retention_rel_pos = self.retnet_rel_pos(slen, incremental_state is not None and not is_first_step, chunkwise_recurrent=self.chunkwise_recurrent)
        # decoder layers
        inner_states = [x]

        l_aux = []

        for idx, layer in enumerate(self.layers):
            if incremental_state is None or is_first_step:
                if is_first_step and incremental_state is not None:
                    if idx not in incremental_state:
                        incremental_state[idx] = {}
            else:
                if idx not in incremental_state:
                    incremental_state[idx] = {}
                    
            x, l_aux_i = layer(
                x,
                incremental_state[idx] if incremental_state is not None else None,
                retention_rel_pos=retention_rel_pos,
                chunkwise_recurrent=self.chunkwise_recurrent,
            )
            l_aux.append(l_aux_i)
            inner_states.append(x)
            
        if self.chunkwise_recurrent and input.size(1) % self.recurrent_chunk_size != 0:
            x = x[:, :input.size(1), :]

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        x_pad, n_pad = self.pad_to_full_observation(x)

        if features_only:
            return x_pad
        
        logits = self.output_layer(x_pad)
        logits = logits.reshape(b, t + n_pad, self.vocab_size + 1)
        logits = logits[:,:t]

        # if we are given some desired targets also calculate the loss
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1), reduction='none')
            if self.action_weight != 1 or self.reward_weight != 1 or self.value_weight != 1:
                #### make weights
                n_states = int(np.ceil(t / self.transition_dim))
                weights = torch.cat([
                    torch.ones(self.observation_dim, device=input.device),
                    torch.ones(self.action_dim, device=input.device) * self.action_weight,
                    torch.ones(1, device=input.device) * self.reward_weight,
                    torch.ones(1, device=input.device) * self.value_weight,
                ])
                ## [ t + 1]
                weights = weights.repeat(n_states)
                ## [ b x t ]
                weights = weights[1:].repeat(b, 1)
                ####
                loss = loss * weights.view(-1)
            loss = (loss * mask.view(-1)).mean()
        else:
            loss = None

        return logits, loss

    def output_layer(self, features):
        return self.output_projection(features)
    
    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # TODO: Possible reason of failure: RMSNorm
        whitelist_weight_modules = (torch.nn.Linear, EinLinear, nn.ModuleList, RMSNorm)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
