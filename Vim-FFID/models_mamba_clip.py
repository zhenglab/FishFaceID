# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import Tensor
from typing import Optional, Union, List, Dict, Tuple, Any

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math
import sys
import types # For dynamic method binding

from collections import namedtuple

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

# Use relative import assuming rope.py is in the same directory
# from .rope import * # Original relative import
from rope import *     # Changed to direct import
import random

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# Add PromptBackboneInteraction class
class PromptBackboneInteraction(nn.Module):
    """
    Implements bidirectional interaction between backbone features and prompt features
    Using cross-attention and multi-layer cascade structure
    """
    def __init__(self, embed_dim, interaction_layers=2, interaction_strength=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.interaction_layers = interaction_layers
        self.interaction_strength = interaction_strength
        
        # Interaction projection layers
        self.img_proj = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(interaction_layers)
        ])
        self.prompt_proj = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(interaction_layers)
        ])
        
        # Layer normalization
        self.img_norm = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(interaction_layers)
        ])
        self.prompt_norm = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(interaction_layers)
        ])
        
    def forward(self, img_features, prompt_features):
        """
        Implements multi-layer cascaded bidirectional interaction
        img_features: [B, D]
        prompt_features: [C, P, D]
        """
        # Directly use input features to avoid cutting off gradient flow
        B, D = img_features.shape
        C, P, D = prompt_features.shape
        
        # Initialize current features
        curr_img = img_features
        curr_prompt = prompt_features
        
        for i in range(self.interaction_layers):
            # 1. Projection
            proj_img = self.img_proj[i](curr_img)  # [B, D]
            
            # Process prompts for each category
            proj_prompts = torch.zeros(C, P, D, device=prompt_features.device, dtype=prompt_features.dtype)
            for c in range(C):
                proj_prompts[c] = self.prompt_proj[i](curr_prompt[c])  # [P, D]
            
            # 2. Cross-attention
            # Attention of image to prompts
            flat_prompts = proj_prompts.reshape(-1, D)  # [C*P, D]
            img_prompt_attn = F.softmax(torch.matmul(proj_img, flat_prompts.t()) / math.sqrt(D), dim=-1)  # [B, C*P]
            
            # Attention of prompts to image
            prompt_img_attn = torch.zeros(C, P, B, device=img_features.device, dtype=img_features.dtype)
            for c in range(C):
                prompt_img_attn[c] = F.softmax(torch.matmul(proj_prompts[c], proj_img.t()) / math.sqrt(D), dim=-1)  # [P, B]
            
            # 3. Enhanced features - Create new tensors without modifying input
            img_context = torch.matmul(img_prompt_attn, flat_prompts)  # [B, D]
            # Use addition operator to create new tensor, avoid using clone()
            enhanced_img = curr_img + (self.interaction_strength * img_context)
            enhanced_img = self.img_norm[i](enhanced_img)
            
            enhanced_prompt = torch.zeros(C, P, D, device=prompt_features.device, dtype=prompt_features.dtype)
            for c in range(C):
                prompt_context = torch.matmul(prompt_img_attn[c], proj_img)  # [P, D]
                # Use addition operator to create new tensor, avoid using clone()
                enhanced_c = curr_prompt[c] + (self.interaction_strength * prompt_context)
                enhanced_prompt[c] = self.prompt_norm[i](enhanced_c)
            
            # Update current features - Directly use enhanced tensors
            curr_img = enhanced_img
            curr_prompt = enhanced_prompt
        
        # Ensure returned features are normalized
        normalized_img = F.normalize(curr_img, p=2, dim=1)
        normalized_prompt = F.normalize(curr_prompt, p=2, dim=2)
        
        return normalized_img, normalized_prompt


__all__ = [
    # Add new model names here later
    'vim_tiny_patch16_224_clip_prompts',
    'vim_small_patch16_224_clip_prompts',
    'vim_base_patch16_224_clip_prompts',
    'vim_base_patch16_stride8_224_clip_prompts',
    'vim_tiny_patch16_224_clip_prompts_fusion',
    'vim_tiny_patch16_224_clip_prototype_prompts',
    'vim_small_patch16_224_clip_prototype_prompts',
    'vim_base_patch16_224_clip_prototype_prompts',
]


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_devide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1" # Note: Or specify v2 based on requirement
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class VisionMamba(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 stride=16,
                 depth=24,
                 embed_dim=192,
                 channels=3,
                 num_classes=1000,
                 ssm_cfg=None,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 if_bidirectional=False,
                 final_pool_type='none', # Deprecated if use_class_aware_prompts=True
                 if_abs_pos_embed=False, # Often False when using prompts/fusion
                 if_rope=False,
                 if_rope_residual=False,
                 flip_img_sequences_ratio=-1.,
                 if_bimamba=False, # Recommended to use bimamba_type
                 bimamba_type="none", # e.g., "v2"
                 if_cls_token=False, # Keep CLS token option, exclude from interaction
                 if_devide_out=False,
                 init_layer_scale=None,
                 use_double_cls_token=False,
                 use_middle_cls_token=False,
                 # >>> Class Aware Prompt Params <<<
                 use_class_aware_prompts: bool = False,
                 # >>> Class Prototype Prompt Params <<<
                 use_class_prototype_prompts: bool = False,
                 prototype_fusion_alpha: float = 0.5,
                 prototype_momentum: float = 0.9,
                 # >>> Intermediate Fusion Params <<<
                 use_intermediate_fusion: bool = False,
                 intermediate_layer_idx: Union[int, List[int]] = None, # e.g. [depth//2] or [6, 12, 18]
                 # >>> Intermediate Auxiliary Loss Params <<<
                 use_intermediate_aux_loss: bool = False,
                 intermediate_aux_loss_layers: Union[int, List[int]] = None, # Layers for auxiliary loss
                 # >>> Multiple Prompts Per Class Params <<<
                 prompts_per_class: int = 1,
                 prompt_diversity_weight: float = 0.1,
                 # >>> Backbone-Prompt Interaction Params <<<
                 use_cross_interaction: bool = False,
                 interaction_layers: int = 2,
                 interaction_strength: float = 0.3,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)
        super().__init__()
        self.depth = depth # Store depth
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        # self.final_pool_type = final_pool_type # Replaced by class prompt logic
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_cls_tokens = 0
        if if_cls_token:
            self.num_cls_tokens = 2 if use_double_cls_token else 1

        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim
        
        # Store number of prompts per class and diversity weight
        self.prompts_per_class = prompts_per_class
        self.prompt_diversity_weight = prompt_diversity_weight

        # Add backbone-prompt interaction parameters
        self.use_cross_interaction = use_cross_interaction
        self.interaction_layers = interaction_layers 
        self.interaction_strength = interaction_strength
        
        # Create interaction module if enabled
        if self.use_cross_interaction:
            print(f"--- Using Backbone-Prompt Cross Interaction: {interaction_layers} layers, strength={interaction_strength} ---")
            self.interaction_module = PromptBackboneInteraction(
                embed_dim=embed_dim,
                interaction_layers=interaction_layers,
                interaction_strength=interaction_strength
            )

        # Ensure both prompt mechanisms are not enabled simultaneously
        assert not (use_class_aware_prompts and use_class_prototype_prompts), "Cannot use both class_aware_prompts and class_prototype_prompts"
        
        self.use_class_aware_prompts = use_class_aware_prompts
        self.use_class_prototype_prompts = use_class_prototype_prompts

        if self.use_class_aware_prompts:
            print(f"--- Using Class Aware Prompts with {prompts_per_class} prompts per class ---")
            assert num_classes > 0, "num_classes must be positive for class aware prompts"
            # Modify prompt_tokens shape to support multiple prompts per class
            self.prompt_tokens = nn.Parameter(torch.zeros(1, num_classes, prompts_per_class, embed_dim, **factory_kwargs))
            trunc_normal_(self.prompt_tokens, std=.02)
            # Disable incompatible options
            if self.if_abs_pos_embed:
                print("Warning: Disabling absolute positional embedding with class aware prompts.")
                self.if_abs_pos_embed = False
            self.final_pool_type = 'none' # Override pooling
        
        elif self.use_class_prototype_prompts:
            print(f"--- Using Class Prototype Prompts with {prompts_per_class} prompts per class ---")
            assert num_classes > 0, "num_classes must be positive for class prototype prompts"
            # Modify prompt_tokens shape to support multiple prompts per class
            self.prompt_tokens = nn.Parameter(torch.zeros(1, num_classes, prompts_per_class, embed_dim, **factory_kwargs))
            trunc_normal_(self.prompt_tokens, std=.02)
            
            # Class prototype storage - Use register_buffer instead of Parameter since we don't need gradient updates
            # Also support multiple prototypes per class
            self.register_buffer('class_prototypes', torch.zeros(num_classes, prompts_per_class, embed_dim, **factory_kwargs))
            self.register_buffer('class_counts', torch.zeros(num_classes, prompts_per_class, dtype=torch.int64, device=device))
            
            # Prototype fusion parameters
            self.prototype_fusion_alpha = prototype_fusion_alpha
            self.prototype_momentum = prototype_momentum
            self.prototype_initialized = False
            
            # Disable incompatible options
            if self.if_abs_pos_embed:
                print("Warning: Disabling absolute positional embedding with class prototype prompts.")
                self.if_abs_pos_embed = False
            self.final_pool_type = 'none' # Override pooling

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim, norm_layer=(nn.LayerNorm if not rms_norm else partial(RMSNorm, eps=norm_epsilon)))
        num_patches = self.patch_embed.num_patches

        # CLS Tokens (optional)
        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim, **factory_kwargs))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim, **factory_kwargs))
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim, **factory_kwargs))
                trunc_normal_(self.cls_token, std=.02)

        # Positional Embedding (optional, potentially disabled)
        if self.if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_cls_tokens, self.embed_dim, **factory_kwargs))
            self.pos_drop = nn.Dropout(p=drop_rate)
            trunc_normal_(self.pos_embed, std=.02)
        else:
            self.pos_embed = None
            self.pos_drop = nn.Identity()

        # RoPE (optional)
        if if_rope:
            half_head_dim = embed_dim // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len,
                device=device,
                dtype=dtype
            )
        else:
             self.rope = None

        # >>> Intermediate Layer Fusion and Auxiliary Loss Setup <<<
        self.use_intermediate_fusion = use_intermediate_fusion
        self.use_intermediate_aux_loss = use_intermediate_aux_loss
        
        # Process intermediate layer indices
        if isinstance(intermediate_layer_idx, int):
            intermediate_layer_idx = [intermediate_layer_idx]
        self.intermediate_layer_indices = [] if intermediate_layer_idx is None else intermediate_layer_idx
            
        if use_intermediate_aux_loss and intermediate_aux_loss_layers is None:
            # Default to using the same layers as fusion for auxiliary loss if not specified
            self.intermediate_aux_loss_layers = self.intermediate_layer_indices if intermediate_layer_idx is not None else [depth//2]
        else:
            self.intermediate_aux_loss_layers = [] if intermediate_aux_loss_layers is None else intermediate_aux_loss_layers
            
        # Ensure they are in list form
        if isinstance(self.intermediate_aux_loss_layers, int):
            self.intermediate_aux_loss_layers = [self.intermediate_aux_loss_layers]
            
        # If both fusion and auxiliary loss are enabled, ensure they use the same indices
        if self.use_intermediate_fusion and self.use_intermediate_aux_loss:
            # Merge the two lists and remove duplicates
            combined_indices = list(set(self.intermediate_layer_indices + self.intermediate_aux_loss_layers))
            self.intermediate_layer_indices = combined_indices
            self.intermediate_aux_loss_layers = combined_indices
        
        # Create normalization and projection layers for intermediate fusion
        if self.use_intermediate_fusion and self.intermediate_layer_indices:
            self.intermediate_norm = nn.ModuleDict()
            
            # Calculate total dimension of fused features: final layer + all intermediate layers
            fusion_dim = embed_dim * (1 + len(self.intermediate_layer_indices))
            
            # Create normalization layers for each intermediate layer
            for idx in self.intermediate_layer_indices:
                self.intermediate_norm[str(idx)] = nn.LayerNorm(embed_dim, eps=norm_epsilon)
            
            # Create fusion projection and normalization layers
            self.fusion_proj = nn.Linear(fusion_dim, embed_dim, bias=False)
            self.fusion_norm = nn.LayerNorm(embed_dim, eps=norm_epsilon)
            
            # Initialize weights
            self.apply(segm_init_weights)
            
        # Create separate projection and normalization layers for auxiliary loss
        if self.use_intermediate_aux_loss:
            self.aux_loss_projectors = nn.ModuleDict()
            self.aux_loss_norms = nn.ModuleDict()
            
            for idx in self.intermediate_aux_loss_layers:
                # Create separate projection layer for each layer, output dimension matches backbone
                self.aux_loss_projectors[str(idx)] = nn.Linear(embed_dim, embed_dim, bias=False)
                self.aux_loss_norms[str(idx)] = nn.LayerNorm(embed_dim, eps=norm_epsilon)
                
            # Initialize weights
            self.apply(segm_init_weights)

        # Mamba Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        # Final Norm
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # Head (becomes Identity if using class prompts)
        if self.use_class_aware_prompts:
            self.head = nn.Identity()
        else:
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
            self.head.apply(segm_init_weights)


        # Weight init (skip head if identity)
        self.patch_embed.apply(segm_init_weights)
        # Mamba init needs to be applied after all layers are defined
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        exclude = {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}
        if self.use_class_aware_prompts:
             exclude.add("prompt_tokens")
        # Exclude fusion layers
        if self.use_intermediate_fusion:
             if hasattr(self, 'fusion_proj'):
                  exclude.add("fusion_proj.weight")
             if hasattr(self, 'fusion_norm'):
                  exclude.add("fusion_norm.weight")
                  exclude.add("fusion_norm.bias")
             if hasattr(self, 'intermediate_norm'):
                  for name, param in self.intermediate_norm.named_parameters():
                       exclude.add(f"intermediate_norm.{name}")
        return exclude

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        # --- Input Processing ---
        x = self.patch_embed(x)
        B, M_patch, D = x.shape

        # Prepend CLS token(s) if enabled
        token_position = None # Track CLS token index/indices
        if self.if_cls_token:
            if self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                token_position = [0, M_patch + 1] # Indices after concat
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
            else:
                cls_token = self.cls_token.expand(B, -1, -1)
                if self.use_middle_cls_token:
                    token_position = M_patch // 2
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                elif if_random_cls_token_position:
                    token_position = random.randint(0, M_patch)
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                    # print("token_position (CLS - random): ", token_position)
                else: # Prepend
                    token_position = 0
                    x = torch.cat((cls_token, x), dim=1)
        M = x.shape[1] # Update total sequence length

        # Add Absolute Positional Embedding (if enabled and exists)
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, :M] # Use only needed pos embeddings
            x = self.pos_drop(x)
        else:
             x = self.pos_drop(x) # Apply dropout even without pos_embed


        # --- Optional Shuffle/Flip ---
        if if_random_token_rank:
            # print("Warning: Random token rank shuffling.")
            shuffle_indices = torch.randperm(M, device=x.device)
            x = x[:, shuffle_indices, :]
            # Update CLS token positions if they were used
            if token_position is not None:
                shuffle_indices_list = shuffle_indices.tolist()
                if isinstance(token_position, list):
                    token_position = [shuffle_indices_list.index(tp) for tp in token_position]
                else:
                    token_position = shuffle_indices_list.index(token_position)
                # print("Updated token_position after shuffle: ", token_position)

        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and random.random() < self.flip_img_sequences_ratio:
            # print("Warning: Flipping sequence.")
            x = x.flip([1])
            if_flip_img_sequences = True

        # --- Mamba Blocks ---
        residual = None
        hidden_states = x
        intermediate_states_dict = {} # Store intermediate states for fusion or auxiliary loss

        # Forward pass through layers
        if not self.if_bidirectional:
            for i, layer in enumerate(self.layers):
                # Store state *before* the layer if its index is needed for fusion or auxiliary loss
                if (self.use_intermediate_fusion or self.use_intermediate_aux_loss) and \
                   (i in self.intermediate_layer_indices or i in self.intermediate_aux_loss_layers):
                     intermediate_states_dict[i] = hidden_states.clone()

                # Apply RoPE if enabled (potentially flip-aware)
                if self.rope is not None:
                    if if_flip_img_sequences: hidden_states = hidden_states.flip([1])
                    hidden_states = self.rope(hidden_states)
                    if if_flip_img_sequences: hidden_states = hidden_states.flip([1])

                    if residual is not None and self.if_rope_residual:
                         if if_flip_img_sequences: residual = residual.flip([1])
                         residual = self.rope(residual)
                         if if_flip_img_sequences: residual = residual.flip([1])

                # Pass through Mamba block
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
        else:
             # Bidirectional pass (only stores forward states for fusion for simplicity)
             # TODO: Potentially fuse backward states too? Requires careful handling.
             intermediate_states_f_dict = {} # Store only forward intermediate states
             for i in range(self.depth // 2):
                 layer_idx_f = i * 2
                 layer_idx_b = i * 2 + 1

                 # Apply RoPE before both directions
                 if self.rope is not None:
                     hidden_states_rope = self.rope(hidden_states)
                     residual_rope = self.rope(residual) if residual is not None and self.if_rope_residual else residual
                 else:
                     hidden_states_rope = hidden_states
                     residual_rope = residual

                 # --- Forward Pass ---
                 # Store state before forward layer if needed
                 if (self.use_intermediate_fusion or self.use_intermediate_aux_loss) and \
                    (layer_idx_f in self.intermediate_layer_indices or layer_idx_f in self.intermediate_aux_loss_layers):
                      intermediate_states_f_dict[layer_idx_f] = hidden_states_rope.clone() # Store RoPE'd state

                 hidden_states_f, residual_f = self.layers[layer_idx_f](
                     hidden_states_rope, residual_rope, inference_params=inference_params
                 )

                 # --- Backward Pass ---
                 hidden_states_flipped = hidden_states_rope.flip([1])
                 residual_flipped = residual_rope.flip([1]) if residual_rope is not None else None
                 hidden_states_b, residual_b = self.layers[layer_idx_b](
                     hidden_states_flipped, residual_flipped, inference_params=inference_params
                 )

                 # Combine bidirectional outputs
                 hidden_states = hidden_states_f + hidden_states_b.flip([1])
                 residual = residual_f + residual_b.flip([1])

             intermediate_states_dict = intermediate_states_f_dict # Use collected forward states
             if (self.use_intermediate_fusion or self.use_intermediate_aux_loss) and \
                not intermediate_states_dict and \
                (self.intermediate_layer_indices or self.intermediate_aux_loss_layers):
                 print(f"Warning: No specified intermediate layers hit in forward pass of bidirectional. Fusion/auxiliary loss skipped.")


        # --- Intermediate Fusion ---
        backbone_output_sequence = hidden_states # Start with final layer's output
        if self.use_intermediate_fusion and intermediate_states_dict:
             # Check if necessary fusion components exist
             if not hasattr(self, 'intermediate_norm') or not hasattr(self, 'fusion_proj') or not hasattr(self, 'fusion_norm'):
                 print("Warning: Intermediate fusion is enabled but required components are missing. Skipping fusion.")
             else:
                 features_to_fuse = [backbone_output_sequence] # Start with final output
                 # Normalize and collect intermediate features in specified order
                 for idx in self.intermediate_layer_indices:
                     if idx in intermediate_states_dict:
                          norm_layer = self.intermediate_norm[str(idx)]
                          features_to_fuse.append(norm_layer(intermediate_states_dict[idx]))
                     else:
                          print(f"Warning: Intermediate state for layer {idx} was expected but not found.")

                 if len(features_to_fuse) > 1: # Ensure we have multiple features to fuse
                      # print(f"Fusing features from final layer + layers {list(intermediate_states_dict.keys())}")
                      backbone_output_sequence = torch.cat(features_to_fuse, dim=-1)
                      backbone_output_sequence = self.fusion_proj(backbone_output_sequence)
                      backbone_output_sequence = self.fusion_norm(backbone_output_sequence)
                 # else:
                      # print("Warning: Fusion intended but only final features available.")

        # --- Process Intermediate Features for Auxiliary Loss ---
        aux_loss_features = {}
        if self.use_intermediate_aux_loss and intermediate_states_dict:
            for idx in self.intermediate_aux_loss_layers:
                if idx in intermediate_states_dict:
                    # Extract intermediate features
                    intermediate_feat = intermediate_states_dict[idx]
                    # Apply projection and normalization
                    aux_norm_layer = self.aux_loss_norms[str(idx)]
                    aux_proj_layer = self.aux_loss_projectors[str(idx)]
                    
                    # Normalize and project
                    aux_feat = aux_norm_layer(intermediate_feat)
                    aux_feat = aux_proj_layer(aux_feat)
                    
                    # Store processed features
                    aux_loss_features[idx] = aux_feat

        # --- Final Normalization ---
        # Apply final norm to the potentially fused sequence
        if not self.fused_add_norm:
            # Note: Residual connection after fusion might be complex.
            # Option 1: Add the *original* pre-fusion residual (if available)
            # Option 2: Treat the fused output as the new main path, apply norm directly.
            # Let's use Option 1 for now, similar to original non-fusion path.
            if residual is None: # Should generally not happen if fusion occurs after some layers
                residual = backbone_output_sequence
            else:
                residual = residual + self.drop_path(backbone_output_sequence)
            normalized_sequence = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Use fused add norm function. Apply to the potentially fused sequence.
            # Residual handling is tricky here too. If fusion happens, adding the
            # last non-fused residual might be incorrect. Let's apply the norm
            # directly to the backbone_output_sequence, assuming fusion creates the
            # new state that needs normalization, potentially skipping the last residual add.
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            # Apply norm without adding the *last* residual if fusion occurred.
            # Pass the *original* residual if no fusion? This gets complicated.
            # Simpler: Apply norm directly to the output sequence, potentially dropping last residual add.
            normalized_sequence = fused_add_norm_fn(
                self.drop_path(backbone_output_sequence), # Apply dropout before norm
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual if not self.use_intermediate_fusion else None, # Only add residual if NOT fused? Or always add? Let's try adding if available.
                prenorm=False, # Applying norm *after* the block/fusion
                residual_in_fp32=self.residual_in_fp32,
            )

        return normalized_sequence, token_position, aux_loss_features if self.use_intermediate_aux_loss else None


    def forward(self, x, return_features=False, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False, labels=None):
        # Get features and auxiliary loss features
        normalized_sequence, token_position, aux_loss_features = self.forward_features(
            x, inference_params,
            if_random_cls_token_position=if_random_cls_token_position,
            if_random_token_rank=if_random_token_rank
        )
        B, M, D = normalized_sequence.shape # Get dimensions

        # --- Output Logic: Contrastive Features or Standard Head ---
        if self.use_class_aware_prompts or self.use_class_prototype_prompts:
            # --- Contrastive Feature Extraction Logic ---
            # 1. Prepare image tokens (exclude CLS tokens)
            image_tokens = normalized_sequence
            if self.if_cls_token and token_position is not None:
                mask = torch.ones(M, dtype=torch.bool, device=x.device)
                if isinstance(token_position, list): # Double CLS tokens
                    for pos in token_position:
                         if 0 <= pos < M: mask[pos] = False
                else: # Single CLS token
                    if 0 <= token_position < M: mask[token_position] = False
                image_tokens = normalized_sequence[:, mask, :] # Shape (B, N, D) where N <= M

            # Handle case where only CLS tokens remain
            if image_tokens.shape[1] == 0:
                 print("Warning: No image tokens left after excluding CLS tokens. Cannot extract features for contrastive loss.")
                 # Return dummy features of expected shape to avoid crashes downstream
                 dummy_img_feat = torch.zeros(B, D, device=x.device, dtype=x.dtype) # Shape (B, D)
                 num_classes = self.prompt_tokens.shape[1]
                 # Create dummy features for multiple prompts per class
                 dummy_pmp_feat = torch.zeros(num_classes, self.prompts_per_class, D, device=x.device, dtype=x.dtype)
                 return dummy_img_feat, dummy_pmp_feat # Return pair of features

            # 2. Pool image tokens to get a single image feature vector per image
            image_features = image_tokens.mean(dim=1) # Shape (B, D)
            
            # Update prototypes during training if labels are provided
            if self.use_class_prototype_prompts and self.training and labels is not None:
                with torch.no_grad():
                    self._update_class_prototypes(image_features, labels)
            
            # 3. Get prompt features and apply diversity loss
            diversity_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
            if self.use_class_aware_prompts:
                # Get multiple prompt tokens per class, shape [C, P, D]
                prompt_features = self.prompt_tokens.squeeze(0)
                
                # Compute and add prompt diversity loss during training
                if self.training and self.prompt_diversity_weight > 0 and self.prompts_per_class > 1:
                    # Directly compute diversity loss, retain gradient connection
                    diversity_loss = self._compute_prompt_diversity_loss(prompt_features)
                
            elif self.use_class_prototype_prompts:
                # Get base multiple prompt tokens, shape [C, P, D]
                base_prompts = self.prompt_tokens.squeeze(0)
                
                # Fuse prototypes with prompts (during both training and inference)
                if self.prototype_initialized:
                    # Ensure features are normalized for better fusion
                    normalized_prototypes = F.normalize(self.class_prototypes, p=2, dim=-1)
                    normalized_prompts = F.normalize(base_prompts, p=2, dim=-1)
                    
                    # Fuse prototypes and prompts (weighted sum followed by normalization)
                    alpha = self.prototype_fusion_alpha
                    fused_prompts = (1-alpha) * normalized_prompts + alpha * normalized_prototypes
                    prompt_features = F.normalize(fused_prompts, p=2, dim=-1)
                else:
                    # Use base prompts if prototypes are not initialized
                    prompt_features = F.normalize(base_prompts, p=2, dim=-1)
                
                # Compute and add prompt diversity loss during training
                if self.training and self.prompt_diversity_weight > 0 and self.prompts_per_class > 1:
                    # Directly compute diversity loss, retain gradient connection
                    diversity_loss = self._compute_prompt_diversity_loss(prompt_features)
            else:
                # Should not reach here
                raise ValueError("Neither class_aware_prompts nor class_prototype_prompts is enabled")

            # 4. Return feature pair for contrastive loss
            # Ensure image_features are normalized for cosine similarity computation with prompt_features
            image_features = F.normalize(image_features, p=2, dim=1)
            
            # Add backbone-prompt interaction step
            if self.use_cross_interaction and hasattr(self, 'interaction_module'):
                # Record pre-interaction features for analysis (optional)
                pre_interaction_img_features = None
                pre_interaction_prompt_features = None
                if random.random() < 0.01:  # Save only when analysis is needed
                    pre_interaction_img_features = image_features.detach()
                    pre_interaction_prompt_features = prompt_features.detach()
                
                # Apply interaction module to enhance features - Directly pass features without using clone()
                enhanced_img_features, enhanced_prompt_features = self.interaction_module(
                    image_features, 
                    prompt_features
                )
                
                # Use enhanced features
                image_features = enhanced_img_features  # Already normalized within interaction module
                prompt_features = enhanced_prompt_features  # Already normalized within interaction module
                
                # Compute diversity loss using enhanced prompts during training
                if self.training and self.prompt_diversity_weight > 0 and self.prompts_per_class > 1:
                    # Directly compute diversity loss using enhanced prompt_features, retain gradient connection
                    diversity_loss = self._compute_prompt_diversity_loss(prompt_features)
                
                # Print interaction effect analysis (optional, for debugging only)
                if pre_interaction_img_features is not None and pre_interaction_prompt_features is not None:
                    with torch.no_grad():
                        # Compute change in image features before and after interaction
                        img_change = F.cosine_similarity(pre_interaction_img_features, image_features, dim=1).mean().item()
                        # Compute average change in prompt features
                        prompt_change = 0.0
                        for c in range(prompt_features.shape[0]):
                            for p in range(prompt_features.shape[1]):
                                prompt_change += F.cosine_similarity(
                                    pre_interaction_prompt_features[c, p:p+1], 
                                    prompt_features[c, p:p+1], 
                                    dim=1
                                ).item()
                        prompt_change /= (prompt_features.shape[0] * prompt_features.shape[1])
                        print(f"Interaction effect - Img change: {1-img_change:.4f}, Prompt change: {1-prompt_change:.4f}")
            
            # Return prompt_features with shape [C, P, D] for contrastive loss to handle multiple prompts per class
            # Also return computed diversity_loss
            return image_features, prompt_features, diversity_loss

        else:
            # --- Standard Head Logic (No prompts) ---
            # Select feature based on CLS token or pooling (existing logic)
            if self.if_cls_token:
                 if self.use_double_cls_token:
                     # Average the two CLS tokens
                     if token_position is not None and isinstance(token_position, list) and len(token_position) == 2:
                          idx0, idx1 = token_position
                          if 0 <= idx0 < M and 0 <= idx1 < M:
                               features = (normalized_sequence[:, idx0, :] + normalized_sequence[:, idx1, :]) / 2
                          else: # Handle invalid indices
                               print("Warning: Invalid double CLS token indices. Using mean pool.")
                               features = normalized_sequence.mean(dim=1)
                     else: # Should not happen
                          print("Warning: Double CLS token position invalid. Using mean pool.")
                          features = normalized_sequence.mean(dim=1)

                 else: # Single CLS token
                     if token_position is not None and isinstance(token_position, int):
                          if 0 <= token_position < M:
                               features = normalized_sequence[:, token_position, :]
                          else: # Handle invalid index
                               print(f"Warning: Invalid single CLS token index ({token_position}). Using mean pool.")
                               features = normalized_sequence.mean(dim=1)
                     else: # Should not happen
                          print(f"Warning: Single CLS token position invalid ({token_position}). Using mean pool.")
                          features = normalized_sequence.mean(dim=1)

            # --- Original Pooling Logic (if no CLS token) ---
            elif hasattr(self, 'final_pool_type') and self.final_pool_type == 'mean':
                 features = normalized_sequence.mean(dim=1)
            elif hasattr(self, 'final_pool_type') and self.final_pool_type == 'max':
                 features = normalized_sequence.max(dim=1)[0]
            elif hasattr(self, 'final_pool_type') and self.final_pool_type == 'none':
                 features = normalized_sequence[:, -1, :] # Use last token
            else:
                 # Default or fallback if no CLS and no specific pooling defined
                 print("Warning: No CLS token and final_pool_type not defined or unsupported. Defaulting to mean pooling.")
                 features = normalized_sequence.mean(dim=1)
            # --- End Original Pooling Logic ---


            # Pass features through the standard head (which should exist if not using prompts)
            logits = self.head(features) # self.head should be nn.Linear here

            # --- Return based on return_features flag ---
            if return_features:
                if self.use_intermediate_aux_loss and aux_loss_features:
                    return logits, features, aux_loss_features
                else:
                    return logits, features
            else:
                if self.use_intermediate_aux_loss and aux_loss_features:
                    return logits, aux_loss_features
                else:
                    return logits

    def _compute_prompt_diversity_loss(self, prompt_features):
        """
        Compute prompt diversity loss to encourage diversity among prompts within the same class
        Add orthogonality constraint and cosine similarity penalty to make prompts within the same class as different as possible
        Args:
            prompt_features: Shape [num_classes, prompts_per_class, embed_dim]
        Returns:
            Computed diversity loss value
        """
        if self.prompts_per_class <= 1:
            # If there is only one prompt per class, no need to compute diversity loss
            return torch.zeros(1, device=prompt_features.device, requires_grad=True)
        
        # Normalize input without using clone() to keep gradient flow
        normalized_prompts = F.normalize(prompt_features, p=2, dim=-1)
        
        # Initialize loss to 0
        total_loss = 0.0
        batch_size = normalized_prompts.shape[0]  # Number of classes
        
        for c in range(batch_size):
            # Get all prompts for the current class
            class_prompts = normalized_prompts[c]  # [prompts_per_class, embed_dim]
            
            # Compute cosine similarity matrix among prompts within the class
            similarity_matrix = torch.matmul(class_prompts, class_prompts.transpose(0, 1))  # [prompts_per_class, prompts_per_class]
            
            # Create mask to exclude diagonal elements (similarity of self with self)
            mask = torch.ones_like(similarity_matrix) - torch.eye(self.prompts_per_class, device=similarity_matrix.device)
            
            # Compute cosine similarity loss - higher similarity results in higher loss
            masked_sim = similarity_matrix * mask
            cosine_loss = masked_sim.sum() / (self.prompts_per_class * (self.prompts_per_class - 1))
            
            # Compute orthogonality loss - if prompts are more orthogonal (dot product close to 0), loss is smaller
            # Target is to make non-diagonal elements as close to 0 as possible
            orthogonal_target = torch.zeros_like(similarity_matrix)
            orthogonal_loss = F.mse_loss(masked_sim, orthogonal_target * mask)
            
            # Combine the two losses
            class_diversity_loss = cosine_loss + orthogonal_loss
            
            # If there are more prompts, add extra "dispersion" loss to avoid prompts clustering in one region of feature space
            if self.prompts_per_class > 2:
                # Compute centroid of prompts within the class
                centroid = class_prompts.mean(dim=0, keepdim=True)
                # Compute distance of each prompt to the centroid - larger distance is better
                dist_to_centroid = 1.0 - F.cosine_similarity(class_prompts, centroid.expand_as(class_prompts), dim=1)
                # Encourage prompts to move away from the centroid (smaller distances result in higher loss)
                dispersion_loss = torch.mean(torch.exp(-5.0 * dist_to_centroid))
                class_diversity_loss = class_diversity_loss + dispersion_loss
            
            # Merge losses using addition operator
            total_loss = total_loss + class_diversity_loss
        
        # Take average across all classes
        avg_loss = total_loss / batch_size
        
        # Return weighted loss value
        return self.prompt_diversity_weight * avg_loss

    def get_diversity_loss(self):
        """
        Return zero loss since we no longer store diversity_loss as an instance attribute
        """
        return torch.tensor(0.0, device=self.prompt_tokens.device)

    def _update_class_prototypes(self, image_features, labels):
        """Update class prototype vectors using improved strategy to ensure capturing diverse features within the class"""
        # Use no_grad context to prevent prototype updates from affecting backpropagation
        with torch.no_grad():
            # Normalize features, ensure not modifying original input
            image_features_normalized = F.normalize(image_features.detach(), p=2, dim=1)
            
            # Iterate over each sample in the batch, update corresponding class prototype
            for i, label in enumerate(labels):
                if label >= self.num_classes:
                    continue  # Ignore labels exceeding number of classes
                
                # Get current sample feature
                feat = image_features_normalized[i]
                
                # To support multiple prompts per class, decide which prompt to update
                if not self.prototype_initialized:
                    # If first time updating this class prototype, use different strategy to initialize diverse prototypes
                    
                    # Check if this class has empty prototype slots
                    empty_slots = (self.class_counts[label] == 0).nonzero().squeeze(-1)
                    
                    if len(empty_slots) > 0:
                        # There are empty slots, directly fill them
                        prompt_idx = empty_slots[0].item()
                        self.class_prototypes[label, prompt_idx] = feat
                        self.class_counts[label, prompt_idx] = 1
                    else:
                        # All slots already have samples, find the slot with the least samples to update
                        prompt_idx = self.class_counts[label].argmin().item()
                        count = self.class_counts[label, prompt_idx].item()
                        current_prototype = self.class_prototypes[label, prompt_idx]
                        new_prototype = (current_prototype * count + feat) / (count + 1)
                        self.class_prototypes[label, prompt_idx] = new_prototype
                        self.class_counts[label, prompt_idx] += 1
                    
                    # Check if all prompts for all classes have at least one sample
                    if torch.all(self.class_counts > 0):
                        self.prototype_initialized = True
                        print("All class prototypes initialized")
                        
                        # After initialization, apply a simple clustering algorithm to separate prototypes
                        if self.prompts_per_class > 1:
                            self._diversify_prototypes()
                else:
                    # Update strategy:
                    # 1. Compute similarity of sample with all prototypes of the class
                    # 2. Assign sample to the most similar prototype (80% of the time) or second most similar prototype (20% of the time) to increase diversity
                    # 3. Update the selected prototype using momentum
                    
                    # Compute similarity of feature with all prototypes of the current class
                    normalized_prototypes = F.normalize(self.class_prototypes[label], p=2, dim=-1)
                    similarities = torch.matmul(feat, normalized_prototypes.t())
                    
                    # Get similarity ranking
                    sorted_indices = torch.argsort(similarities, descending=True)
                    
                    # 80% probability to update the most similar prototype, 20% probability to update the second most similar prototype (if exists)
                    if self.prompts_per_class > 1 and torch.rand(1).item() < 0.2 and len(sorted_indices) > 1:
                        update_idx = sorted_indices[1].item()  # Choose second most similar prototype
                    else:
                        update_idx = sorted_indices[0].item()  # Choose most similar prototype
                    
                    # Update using momentum
                    momentum = self.prototype_momentum
                    current_prototype = self.class_prototypes[label, update_idx]
                    new_prototype = momentum * current_prototype + (1 - momentum) * feat
                    self.class_prototypes[label, update_idx] = new_prototype
                    
                    # Re-normalize after update
                    normalized_prototype = F.normalize(new_prototype, p=2, dim=0)
                    self.class_prototypes[label, update_idx] = normalized_prototype
    
    def _diversify_prototypes(self):
        """Apply simple separation strategy after initialization to ensure prototype diversity"""
        print("Diversifying class prototypes...")
        
        # Ensure execution within no_grad context
        with torch.no_grad():
            for c in range(self.num_classes):
                # Check if this class has enough samples to optimize prototypes
                if torch.all(self.class_counts[c] > 0):
                    # Get current class prototypes
                    prototypes = self.class_prototypes[c]  # [prompts_per_class, embed_dim]
                    new_prototypes = torch.zeros_like(prototypes)  # Create new tensor to store updated prototypes
                    
                    # For each prototype, add a bit of random perturbation and move away from other prototypes
                    for i in range(self.prompts_per_class):
                        # Compute average direction of current prototype with other prototypes
                        other_indices = [j for j in range(self.prompts_per_class) if j != i]
                        if len(other_indices) > 0:  # Ensure there are other prototypes
                            # Collect other prototypes
                            others = torch.stack([prototypes[j] for j in other_indices])
                            mean_others = others.mean(dim=0)
                            
                            # Move away from other prototypes
                            direction = prototypes[i] - mean_others
                            
                            # Normalize and add movement
                            direction = F.normalize(direction, p=2, dim=0)
                            
                            # Small step size, 0.1, and slight random perturbation
                            perturbed = prototypes[i] + 0.1 * direction + 0.05 * torch.randn_like(prototypes[i])
                            new_prototypes[i] = F.normalize(perturbed, p=2, dim=0)
                    
                    # Update all prototypes at once
                    self.class_prototypes[c] = new_prototypes


# --- Registration for New Models ---

@register_model
def vim_tiny_patch16_224_clip_prompts(pretrained=False, num_classes=1000, prompts_per_class=1, prompt_diversity_weight=0.1, **kwargs):
    """ Vim-Tiny with Patch16, 224x224, using Class Aware Prompts and Cross-Interaction """
    model = VisionMamba(
        patch_size=16, embed_dim=192, depth=24, num_classes=num_classes,
        rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        if_cls_token=True, use_middle_cls_token=True, # Keep CLS but exclude from interaction
        bimamba_type="v2", # Assuming v2 is desired
        if_devide_out=True,
        use_class_aware_prompts=True, # Enable prompts
        use_intermediate_fusion=False, # Disable fusion by default for this variant
        if_abs_pos_embed=False, # Usually disabled with prompts
        prompts_per_class=prompts_per_class, # Set number of prompts per class
        prompt_diversity_weight=prompt_diversity_weight, # Set diversity loss weight
        **kwargs)
    model.default_cfg = _cfg()
    # Add pretrained loading logic if available for this specific variant
    # if pretrained: ...
    return model

@register_model
def vim_small_patch16_224_clip_prompts(pretrained=False, num_classes=1000, prompts_per_class=1, prompt_diversity_weight=0.1, **kwargs):
    """ Vim-Small with Patch16, 224x224, using Class Aware Prompts and Cross-Interaction 
        Relies on kwargs to control features like fusion.
    """
    # Default use_intermediate_fusion to False if not provided in kwargs
    # but allow kwargs to override it.
    # We explicitly set use_class_aware_prompts=True as that's the core of this variant.
    model_args = {
        'patch_size': 16, 'embed_dim': 384, 'depth': 24, 'num_classes': num_classes,
        'rms_norm': True, 'residual_in_fp32': True, 'fused_add_norm': True,
        'if_cls_token': True, 'use_middle_cls_token': True, 
        'bimamba_type': "v2", 
        'if_devide_out': True,
        'use_class_aware_prompts': True, 
        'if_abs_pos_embed': False, 
        'prompts_per_class': prompts_per_class, # Set number of prompts per class
        'prompt_diversity_weight': prompt_diversity_weight, # Set diversity loss weight
        # Let kwargs determine fusion settings
        # 'use_intermediate_fusion': False, # Removed hardcoded value
    }
    # Update default args with any values passed via kwargs
    model_args.update(kwargs)
    
    model = VisionMamba(**model_args)
    
    model.default_cfg = _cfg()
    # Add pretrained loading logic if available for this specific variant
    # if pretrained: ...
    return model

@register_model
def vim_tiny_patch16_224_clip_prompts_fusion(pretrained=False, num_classes=1000, intermediate_layer_idx=None, prompts_per_class=1, prompt_diversity_weight=0.1, **kwargs):
    """ Vim-Tiny with Patch16, 224x224, Class Prompts, Cross-Interaction, and Intermediate Fusion """
    if intermediate_layer_idx is None:
         intermediate_layer_idx = [12] # Default fusion layer for tiny (depth 24)
    model = VisionMamba(
        patch_size=16, embed_dim=192, depth=24, num_classes=num_classes,
        rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        if_cls_token=True, use_middle_cls_token=True, # Keep CLS but exclude from interaction
        bimamba_type="v2",
        if_devide_out=True,
        use_class_aware_prompts=True, # Enable prompts
        use_intermediate_fusion=True, # Enable fusion
        intermediate_layer_idx=intermediate_layer_idx,
        if_abs_pos_embed=False, # Usually disabled with prompts/fusion
        prompts_per_class=prompts_per_class, # Set number of prompts per class
        prompt_diversity_weight=prompt_diversity_weight, # Set diversity loss weight
        **kwargs)
    model.default_cfg = _cfg()
    # if pretrained: ...
    return model

# You can add more variants combining different options (stride, size, fusion layers, etc.) 

# --- Added Base Model Definition ---
@register_model
def vim_base_patch16_224_clip_prompts(pretrained=False, num_classes=1000, prompts_per_class=1, prompt_diversity_weight=0.1, use_cross_interaction=False, interaction_layers=2, interaction_strength=0.3, **kwargs):
    """ Vim-Base with Patch16, 224x224, using Class Aware Prompts and Cross-Interaction
        Relies on kwargs to control features like fusion.
        Uses embed_dim=768, depth=24 as base configuration assumption.
    """
    model_args = {
        'patch_size': 16, 'embed_dim': 768, 'depth': 24, 'num_classes': num_classes, # Base config: embed_dim=768, depth=24
        'rms_norm': True, 'residual_in_fp32': True, 'fused_add_norm': True,
        'if_cls_token': True, 'use_middle_cls_token': True, # Consistent with small
        'bimamba_type': "v2", # Consistent with small
        'if_devide_out': True, # Consistent with small
        'use_class_aware_prompts': True, # Core feature
        'if_abs_pos_embed': False, # Standard for prompts
        'prompts_per_class': prompts_per_class, # Set number of prompts per class
        'prompt_diversity_weight': prompt_diversity_weight, # Set diversity loss weight
        # Add backbone-prompt interaction parameters
        'use_cross_interaction': use_cross_interaction,
        'interaction_layers': interaction_layers,
        'interaction_strength': interaction_strength,
        # Let kwargs determine fusion settings
    }
    # Update default args with any values passed via kwargs
    model_args.update(kwargs)

    model = VisionMamba(**model_args)

    model.default_cfg = _cfg()
    # Add pretrained loading logic if available for this specific variant
    # if pretrained: ...
    return model
# --- End Added Base Model Definition ---

# --- Added Class Prototype Prompt Models ---
@register_model
def vim_tiny_patch16_224_clip_prototype_prompts(pretrained=False, num_classes=1000, prompts_per_class=1, prompt_diversity_weight=0.1, **kwargs):
    """ Vim-Tiny with Patch16, 224x224, using Class Prototype Prompts """
    model = VisionMamba(
        patch_size=16, embed_dim=192, depth=24, num_classes=num_classes,
        rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        if_cls_token=True, use_middle_cls_token=True, # Keep CLS but exclude from interaction
        bimamba_type="v2", # Assuming v2 is desired
        if_devide_out=True,
        use_class_prototype_prompts=True, # Enable prototype prompts instead of standard prompts
        if_abs_pos_embed=False, # Usually disabled with prompts
        prompts_per_class=prompts_per_class, # Set number of prompts per class
        prompt_diversity_weight=prompt_diversity_weight, # Set diversity loss weight
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vim_small_patch16_224_clip_prototype_prompts(pretrained=False, num_classes=1000, prompts_per_class=1, prompt_diversity_weight=0.1, **kwargs):
    """ Vim-Small with Patch16, 224x224, using Class Prototype Prompts """
    model_args = {
        'patch_size': 16, 'embed_dim': 384, 'depth': 24, 'num_classes': num_classes,
        'rms_norm': True, 'residual_in_fp32': True, 'fused_add_norm': True,
        'if_cls_token': True, 'use_middle_cls_token': True, 
        'bimamba_type': "v2", 
        'if_devide_out': True,
        'use_class_prototype_prompts': True, # Enable prototype prompts
        'if_abs_pos_embed': False, 
        'prompts_per_class': prompts_per_class, # Set number of prompts per class
        'prompt_diversity_weight': prompt_diversity_weight, # Set diversity loss weight
    }
    model_args.update(kwargs)
    
    model = VisionMamba(**model_args)
    model.default_cfg = _cfg()
    return model

@register_model
def vim_base_patch16_224_clip_prototype_prompts(pretrained=False, num_classes=1000, prompts_per_class=1, prompt_diversity_weight=0.1, **kwargs):
    """ Vim-Base with Patch16, 224x224, using Class Prototype Prompts """
    model_args = {
        'patch_size': 16, 'embed_dim': 768, 'depth': 24, 'num_classes': num_classes,
        'rms_norm': True, 'residual_in_fp32': True, 'fused_add_norm': True,
        'if_cls_token': True, 'use_middle_cls_token': True,
        'bimamba_type': "v2",
        'if_devide_out': True,
        'use_class_prototype_prompts': True, # Enable prototype prompts
        'if_abs_pos_embed': False,
        'prompts_per_class': prompts_per_class, # Set number of prompts per class
        'prompt_diversity_weight': prompt_diversity_weight, # Set diversity loss weight
    }
    model_args.update(kwargs)
    
    model = VisionMamba(**model_args)
    model.default_cfg = _cfg()
    return model
# --- End Added Class Prototype Prompt Models --- 

@register_model
def vim_base_patch16_stride8_224_clip_prompts(pretrained=False, num_classes=1000, prompts_per_class=1, prompt_diversity_weight=0.1, **kwargs):
    """ Vim-Base with Patch16 but Stride8, 224x224, using Class Aware Prompts and Cross-Interaction
        Uses patch_size=16 but stride=8 for higher resolution feature maps
        Uses embed_dim=768, depth=24 as base configuration assumption.
    """
    model_args = {
        'patch_size': 16, 'stride': 8, 'embed_dim': 768, 'depth': 24, 'num_classes': num_classes,
        'rms_norm': True, 'residual_in_fp32': True, 'fused_add_norm': True,
        'if_cls_token': True, 'use_middle_cls_token': True,
        'bimamba_type': "v2",
        'if_devide_out': True,
        'use_class_aware_prompts': True, # Core feature
        'if_abs_pos_embed': False, # Standard for prompts
        'prompts_per_class': prompts_per_class, # Set number of prompts per class
        'prompt_diversity_weight': prompt_diversity_weight, # Set diversity loss weight
        # Let kwargs determine fusion settings
    }
    # Update default args with any values passed via kwargs
    model_args.update(kwargs)

    model = VisionMamba(**model_args)

    model.default_cfg = _cfg()
    # Add pretrained loading logic if available for this specific variant
    # if pretrained: ...
    return model 

@register_model
def vim_tiny_patch16_224_ce_infonce_multiprompt(pretrained=False, num_classes=1000, prompts_per_class=3, prompt_diversity_weight=0.2, **kwargs):
    """ Vim-Tiny model supporting CE+InfoNCE combination mode with multiple prompts per class """
    model = VisionMamba(
        patch_size=16, embed_dim=192, depth=24, num_classes=num_classes,
        rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        if_cls_token=True, use_middle_cls_token=True,
        bimamba_type="v2",
        if_devide_out=True,
        use_class_aware_prompts=True, # Enable prompts
        if_abs_pos_embed=False,
        prompts_per_class=prompts_per_class, # Multiple prompts per class
        prompt_diversity_weight=prompt_diversity_weight, # Increase diversity loss weight
        **kwargs)
    model.default_cfg = _cfg()
    
    # Add a standard classification head for CE loss
    model.head = nn.Linear(model.embed_dim, num_classes)
    nn.init.zeros_(model.head.weight)
    nn.init.zeros_(model.head.bias)
    
    return model

@register_model
def vim_small_patch16_224_ce_infonce_multiprompt(pretrained=False, num_classes=1000, prompts_per_class=3, prompt_diversity_weight=0.2, **kwargs):
    """ Vim-Small model supporting CE+InfoNCE combination mode with multiple prompts per class """
    model_args = {
        'patch_size': 16, 'embed_dim': 384, 'depth': 24, 'num_classes': num_classes,
        'rms_norm': True, 'residual_in_fp32': True, 'fused_add_norm': True,
        'if_cls_token': True, 'use_middle_cls_token': True, 
        'bimamba_type': "v2", 
        'if_devide_out': True,
        'use_class_aware_prompts': True, 
        'if_abs_pos_embed': False, 
        'prompts_per_class': prompts_per_class,
        'prompt_diversity_weight': prompt_diversity_weight,
    }
    model_args.update(kwargs)
    
    model = VisionMamba(**model_args)
    
    # Add a standard classification head for CE loss
    model.head = nn.Linear(model.embed_dim, num_classes)
    nn.init.zeros_(model.head.weight)
    nn.init.zeros_(model.head.bias)
    
    model.default_cfg = _cfg()
    return model

# Modify forward method to support CE+InfoNCE combination mode
def forward_with_ce_infonce(self, x, return_features=False, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False, labels=None):
    """Forward method supporting CE+InfoNCE combination mode"""
    # Get feature sequence
    normalized_sequence, token_position, aux_loss_features = self.forward_features(
        x, inference_params,
        if_random_cls_token_position=if_random_cls_token_position,
        if_random_token_rank=if_random_token_rank
    )
    B, M, D = normalized_sequence.shape
    
    # 1. Process features for CE loss (usually CLS token or pooled features)
    if self.if_cls_token:
        if self.use_double_cls_token:
            # Average two CLS tokens
            if token_position is not None and isinstance(token_position, list) and len(token_position) == 2:
                idx0, idx1 = token_position
                if 0 <= idx0 < M and 0 <= idx1 < M:
                    ce_features = (normalized_sequence[:, idx0, :] + normalized_sequence[:, idx1, :]) / 2
                else:
                    ce_features = normalized_sequence.mean(dim=1)
            else:
                ce_features = normalized_sequence.mean(dim=1)
        else:  # Single CLS token
            if token_position is not None and isinstance(token_position, int):
                if 0 <= token_position < M:
                    ce_features = normalized_sequence[:, token_position, :]
                else:
                    ce_features = normalized_sequence.mean(dim=1)
            else:
                ce_features = normalized_sequence.mean(dim=1)
    else:
        # Use mean pooling if no CLS token
        ce_features = normalized_sequence.mean(dim=1)
    
    # Compute logits through classification head
    logits = self.head(ce_features)
    
    # 2. Process features for InfoNCE
    # Prepare image tokens (exclude CLS tokens)
    image_tokens = normalized_sequence
    if self.if_cls_token and token_position is not None:
        mask = torch.ones(M, dtype=torch.bool, device=x.device)
        if isinstance(token_position, list):
            for pos in token_position:
                if 0 <= pos < M: mask[pos] = False
        else:
            if 0 <= token_position < M: mask[token_position] = False
        image_tokens = normalized_sequence[:, mask, :]
    
    # Pool image tokens to get features for InfoNCE
    if image_tokens.shape[1] > 0:
        infonce_features = image_tokens.mean(dim=1)
    else:
        # Use CE features if no valid tokens
        infonce_features = ce_features
    
    # Normalize InfoNCE features
    infonce_features = F.normalize(infonce_features, p=2, dim=1)
    
    # 3. Get prompt features
    diversity_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
    if self.use_class_aware_prompts:
        # Get multiple prompts per class
        prompt_features = self.prompt_tokens.squeeze(0)
        
        # Compute diversity loss during training
        if self.training and self.prompt_diversity_weight > 0 and self.prompts_per_class > 1:
            diversity_loss = self._compute_prompt_diversity_loss(prompt_features)
    
    elif self.use_class_prototype_prompts:
        # Get base prompts
        base_prompts = self.prompt_tokens.squeeze(0)
        
        # Fuse prototypes with prompts
        if self.prototype_initialized:
            normalized_prototypes = F.normalize(self.class_prototypes, p=2, dim=-1)
            normalized_prompts = F.normalize(base_prompts, p=2, dim=-1)
            
            alpha = self.prototype_fusion_alpha
            fused_prompts = (1-alpha) * normalized_prompts + alpha * normalized_prototypes
            prompt_features = F.normalize(fused_prompts, p=2, dim=-1)
        else:
            prompt_features = F.normalize(base_prompts, p=2, dim=-1)
        
        # Compute diversity loss
        if self.training and self.prompt_diversity_weight > 0 and self.prompts_per_class > 1:
            diversity_loss = self._compute_prompt_diversity_loss(prompt_features)
            
        # Update prototypes during training if labels are provided
        if self.training and labels is not None:
            with torch.no_grad():
                self._update_class_prototypes(infonce_features, labels)
    else:
        raise ValueError("CE+InfoNCE mode requires class_aware_prompts or class_prototype_prompts")
    
    # 4. Return quadruple: (logits, image_features, prompt_features, diversity_loss)
    return logits, infonce_features, prompt_features, diversity_loss

# To avoid modifying existing class definition, we use monkey patching to register this new method
# Need to manually call this function after model initialization to set forward_with_ce_infonce method
def register_ce_infonce_forward(model):
    """Register CE+InfoNCE combination mode forward method"""
    model.forward_with_ce_infonce = types.MethodType(forward_with_ce_infonce, model)
    print("Registered CE+InfoNCE forward method to model")