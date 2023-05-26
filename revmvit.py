# MViTv2 implementation taken from detectron2
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/mvit.py

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath, Mlp, trunc_normal_

from rev import RevBackProp, RevViT
from utils import (
    PatchEmbed,
    add_decomposed_rel_pos,
    get_abs_pos,
    window_partition,
    window_unpartition,
)


def attention_pool(x, pool, norm=None):
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H1, W1) -> (B, H1, W1, C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    """Multiscale Multi-head Attention block."""

    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        pool_kernel=(3, 3),
        stride_q=1,
        stride_kv=1,
        residual_pooling=True,
        window_size=0,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            pool_kernel (tuple): kernel size for qkv pooling layers.
            stride_q (int): stride size for q pooling layer.
            stride_kv (int): stride size for kv pooling layer.
            residual_pooling (bool): If true, enable residual pooling.
            use_rel_pos (bool): If True, add relative postional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out)

        # qkv pooling
        pool_padding = [k // 2 for k in pool_kernel]
        dim_conv = dim_out // num_heads
        self.pool_q = nn.Conv2d(
            dim_conv,
            dim_conv,
            pool_kernel,
            stride=stride_q,
            padding=pool_padding,
            groups=dim_conv,
            bias=False,
        )
        self.norm_q = norm_layer(dim_conv)
        self.pool_k = nn.Conv2d(
            dim_conv,
            dim_conv,
            pool_kernel,
            stride=stride_kv,
            padding=pool_padding,
            groups=dim_conv,
            bias=False,
        )
        self.norm_k = norm_layer(dim_conv)
        self.pool_v = nn.Conv2d(
            dim_conv,
            dim_conv,
            pool_kernel,
            stride=stride_kv,
            padding=pool_padding,
            groups=dim_conv,
            bias=False,
        )
        self.norm_v = norm_layer(dim_conv)

        self.window_size = window_size
        if window_size:
            self.q_win_size = window_size // stride_q
            self.kv_win_size = window_size // stride_kv
        self.residual_pooling = residual_pooling

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            assert input_size[0] == input_size[1]
            size = input_size[0]
            rel_dim = 2 * max(size // stride_q, size // stride_kv) - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_dim, head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H, W, C)
        qkv = (
            self.qkv(x)
            .reshape(B, H, W, 3, self.num_heads, -1)
            .permute(3, 0, 4, 1, 2, 5)
        )
        # q, k, v with shape (B * nHead, H, W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H, W, -1).unbind(0)

        q = attention_pool(q, self.pool_q, self.norm_q)
        k = attention_pool(k, self.pool_k, self.norm_k)
        v = attention_pool(v, self.pool_v, self.norm_v)

        ori_q = q
        if self.window_size:
            q, q_hw_pad = window_partition(q, self.q_win_size)
            k, kv_hw_pad = window_partition(k, self.kv_win_size)
            v, _ = window_partition(v, self.kv_win_size)
            q_hw = (self.q_win_size, self.q_win_size)
            kv_hw = (self.kv_win_size, self.kv_win_size)
        else:
            q_hw = q.shape[1:3]
            kv_hw = k.shape[1:3]

        q = q.view(q.shape[0], np.prod(q_hw), -1)
        k = k.view(k.shape[0], np.prod(kv_hw), -1)
        v = v.view(v.shape[0], np.prod(kv_hw), -1)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, q_hw, kv_hw
            )

        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.view(x.shape[0], q_hw[0], q_hw[1], -1)

        if self.window_size:
            x = window_unpartition(
                x, self.q_win_size, q_hw_pad, ori_q.shape[1:3]
            )

        if self.residual_pooling:
            x += ori_q

        H, W = x.shape[1], x.shape[2]
        x = (
            x.view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    """Multiscale Transformer block, specifically for Stage Transitions."""

    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        qkv_pool_kernel=(3, 3),
        stride_q=1,
        stride_kv=1,
        residual_pooling=True,
        window_size=0,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
        enable_amp=False,
    ):
        """
        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            num_heads (int): Number of attention heads in the MViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            qkv_pool_kernel (tuple): kernel size for qkv pooling layers.
            stride_q (int): stride size for q pooling layer.
            stride_kv (int): stride size for kv pooling layer.
            residual_pooling (bool): If true, enable residual pooling.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_rel_pos (bool): If True, add relative postional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution.
            enable_amp (bool): If True, enable mixed precision training.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            pool_kernel=qkv_pool_kernel,
            stride_q=stride_q,
            stride_kv=stride_kv,
            residual_pooling=residual_pooling,
            window_size=window_size,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size,
        )

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=int(dim_out * mlp_ratio),
            out_features=dim_out,
            act_layer=act_layer,
        )
        self.enable_amp = enable_amp

        # For Stage-Transition
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        if stride_q > 1:
            kernel_skip = stride_q + 1
            padding_skip = int(kernel_skip // 2)
            self.pool_skip = nn.MaxPool2d(
                kernel_skip, stride_q, padding_skip, ceil_mode=False
            )

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            x_norm = self.norm1(x)
            x_block = self.attn(x_norm)

            if hasattr(self, "proj"):
                x = self.proj(x_norm)
            if hasattr(self, "pool_skip"):
                x = attention_pool(x, self.pool_skip)

            x = x + self.drop_path(x_block)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class ReversibleMultiScaleBlock(nn.Module):
    """Reversible Multiscale Transformer block, no pool residual or projection."""

    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        qkv_pool_kernel=(3, 3),
        stride_q=1,
        stride_kv=1,
        residual_pooling=True,
        window_size=0,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
        enable_amp=False,
    ):
        """
        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            num_heads (int): Number of attention heads in the MViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            qkv_pool_kernel (tuple): kernel size for qkv pooling layers.
            stride_q (int): stride size for q pooling layer.
            stride_kv (int): stride size for kv pooling layer.
            residual_pooling (bool): If true, enable residual pooling.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_rel_pos (bool): If True, add relative postional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution.
            enable_amp (bool): If True, enable mixed precision training.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            pool_kernel=qkv_pool_kernel,
            stride_q=stride_q,
            stride_kv=stride_kv,
            residual_pooling=residual_pooling,
            window_size=window_size,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size,
        )

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=int(dim_out * mlp_ratio),
            out_features=dim_out,
            act_layer=act_layer,
        )
        self.enable_amp = enable_amp

        self.seeds = {}

        if stride_q > 1:
            raise ValueError(
                "stride_q > 1 is not supported for ReversibleMultiScaleBlock."
            )

    def seed_cuda(self, key):
        """
        Fix seeds to allow for stochastic elements such as
        dropout to be reproduced exactly in activation
        recomputation in the backward pass.

        From RevViT.
        """

        # randomize seeds
        # use cuda generator if available
        if (
            hasattr(torch.cuda, "default_generators")
            and len(torch.cuda.default_generators) > 0
        ):
            # GPU
            device_idx = torch.cuda.current_device()
            seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            seed = int(torch.seed() % sys.maxsize)

        self.seeds[key] = seed
        torch.manual_seed(self.seeds[key])

    def F(self, x):
        """Attention forward pass"""
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            x_out = self.attn(self.norm1(x))
        return x_out

    def G(self, x):
        """MLP forward pass"""
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            x_out = self.mlp(self.norm2(x))
        return x_out

    def forward(self, X_1, X_2):
        assert X_1.shape == X_2.shape, "Input shapes are different."

        self.seed_cuda("attn")
        f_X_2 = self.F(X_1)

        self.seed_cuda("droppath")
        Y_1 = X_1 + self.drop_path(f_X_2)

        # free memory
        del X_1

        self.seed_cuda("mlp")
        g_Y_1 = self.G(Y_1)

        torch.manual_seed(self.seeds["droppath"])
        Y_2 = X_2 + self.drop_path(g_Y_1)

        del X_2

        return Y_1, Y_2

    def backward_pass(self, Y_1, Y_2, dY_1, dY_2):
        """
        equations for recovering activations:
        X2 = Y2 - MLP(Y1)
        X1 = Y1 - Attn(X2)
        """

        # temporarily record intermediate activation for G
        # and use them for gradient calculcation of G
        with torch.enable_grad():
            Y_1.requires_grad = True

            torch.manual_seed(self.seeds["mlp"])
            g_Y_1 = self.G(Y_1)

            torch.manual_seed(self.seeds["droppath"])
            g_Y_1 = self.drop_path(g_Y_1)

            g_Y_1.backward(dY_2, retain_graph=True)

        # activation recomputation is by design and not part of
        # the computation graph in forward pass.
        with torch.no_grad():
            X_2 = Y_2 - g_Y_1
            del g_Y_1

            dY_1 = dY_1 + Y_1.grad
            Y_1.grad = None

        # record F activations and calc gradients on F
        with torch.enable_grad():
            X_2.requires_grad = True

            torch.manual_seed(self.seeds["attn"])
            f_X_2 = self.F(X_2)

            torch.manual_seed(self.seeds["droppath"])
            f_X_2 = self.drop_path(f_X_2)

            f_X_2.backward(dY_1, retain_graph=True)

        # propagate reverse computed acitvations at the start of
        # the previou block for backprop.s
        with torch.no_grad():
            X_1 = Y_1 - f_X_2

            del f_X_2, Y_1
            dY_2 = dY_2 + X_2.grad

            X_2.grad = None
            X_2 = X_2.detach()

        return X_1, X_2, dY_1, dY_2

    def backward_pass_recover(self, Y_1, Y_2):
        """
        Use equations to recover activations and return them.
        Used for parallelizing the backward pass.
        """
        with torch.enable_grad():
            Y_1.requires_grad = True

            torch.manual_seed(self.seeds["mlp"])
            g_Y_1 = self.G(Y_1)

            torch.manual_seed(self.seeds["droppath"])
            g_Y_1 = self.drop_path(g_Y_1)

        with torch.no_grad():
            X_2 = Y_2 - g_Y_1

        with torch.enable_grad():
            X_2.requires_grad = True

            torch.manual_seed(self.seeds["attn"])
            f_X_2 = self.F(X_2)

            torch.manual_seed(self.seeds["droppath"])
            f_X_2 = self.drop_path(f_X_2)

        with torch.no_grad():
            X_1 = Y_1 - f_X_2

        # Keep tensors around to do backprop on the graph.
        ctx = [X_1, X_2, Y_1, g_Y_1, f_X_2]
        return ctx

    def backward_pass_grads(self, X_1, X_2, Y_1, g_Y_1, f_X_2, dY_1, dY_2):
        """
        Receive intermediate activations and inputs to backprop through.
        """

        with torch.enable_grad():
            g_Y_1.backward(dY_2)

        with torch.no_grad():
            dY_1 = dY_1 + Y_1.grad
            Y_1.grad = None

        with torch.enable_grad():
            f_X_2.backward(dY_1)

        with torch.no_grad():
            dY_2 = dY_2 + X_2.grad
            X_2.grad = None
            X_2.detach()

        return dY_1, dY_2


class ReversibleMViT(nn.Module):
    """
    This module adds reversibility on top of Multiscale Vision Transformer (MViT) from :paper:'mvitv2'.
    """

    def __init__(
        self,
        img_size=224,
        patch_kernel=(7, 7),
        patch_stride=(4, 4),
        patch_padding=(3, 3),
        in_chans=3,
        embed_dim=96,
        depth=16,
        num_heads=1,
        last_block_indexes=(0, 2, 13, 15),
        qkv_pool_kernel=(3, 3),
        adaptive_kv_stride=4,
        adaptive_window_size=56,
        residual_pooling=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        num_classes=1000,
        use_abs_pos=False,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        fast_backprop=False,
        enable_amp=False,
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_kernel (tuple): kernel size for patch embedding.
            patch_stride (tuple): stride size for patch embedding.
            patch_padding (tuple): padding size for patch embedding.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of MViT.
            num_heads (int): Number of base attention heads in each MViT block.
            last_block_indexes (tuple): Block indexes for last blocks in each stage.
            qkv_pool_kernel (tuple): kernel size for qkv pooling layers.
            adaptive_kv_stride (int): adaptive stride size for kv pooling.
            adaptive_window_size (int): adaptive window size for window attention blocks.
            residual_pooling (bool): If true, enable residual pooling.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative postional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            fast_backprop (bool): If True, use fast backprop, i.e. PaReprop.
            enable_amp (bool): If True, enable automatic mixed precision.
        """
        super().__init__()
        self.patch_embed = PatchEmbed(
            kernel_size=patch_kernel,
            stride=patch_stride,
            padding=patch_padding,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        img_size = img_size[0]

        if use_abs_pos:
            # Initialize absoluate positional embedding with pretrain image size.
            num_patches = (img_size // patch_stride[0]) * (
                img_size // patch_stride[1]
            )
            num_positions = num_patches
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_positions, embed_dim)
            )
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        dim_out = embed_dim
        stride_kv = adaptive_kv_stride
        window_size = adaptive_window_size
        input_size = (img_size // patch_stride[0], img_size // patch_stride[1])
        stage = 2
        stride = patch_stride[0]
        # self._out_feature_strides = {}
        # self._out_feature_channels = {}
        self.blocks = nn.ModuleList()
        for i in range(depth):
            # Multiply stride_kv by 2 if it's the last block of stage2 and stage3.
            # Here however, we modify it so that we only look at stage2 (since we only have 3 stages for a smaller CIFAR model)
            # if i == last_block_indexes[1] or i == last_block_indexes[2]:
            if i == last_block_indexes[1]:
                stride_kv_ = stride_kv * 2
            else:
                stride_kv_ = stride_kv
            # hybrid window attention: global attention in last three stages.
            window_size_ = 0 if i in last_block_indexes[1:] else window_size
            block_type = (
                MultiScaleBlock
                if i - 1 in last_block_indexes
                else ReversibleMultiScaleBlock
            )
            block = block_type(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                qkv_pool_kernel=qkv_pool_kernel,
                stride_q=2 if i - 1 in last_block_indexes else 1,
                stride_kv=stride_kv_,
                residual_pooling=residual_pooling,
                window_size=window_size_,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                input_size=input_size,
                enable_amp=enable_amp,
            )
            self.blocks.append(block)

            embed_dim = dim_out
            if i in last_block_indexes:
                embed_dim *= 2
                dim_out *= 2
                num_heads *= 2
                stride_kv = max(stride_kv // 2, 1)
                stride *= 2
                stage += 1
            if i - 1 in last_block_indexes:
                window_size = window_size // 2
                input_size = [s // 2 for s in input_size]

        self.head = nn.Linear(embed_dim, num_classes, bias=True)
        self.norm = nn.LayerNorm(embed_dim)

        # self._out_features = out_features
        self._last_block_indexes = last_block_indexes
        self.use_fast_backprop = fast_backprop

        if self.use_fast_backprop:
            # Initialize streams globally
            global s1, s2
            s1 = torch.cuda.default_stream(device=torch.cuda.current_device())
            # s1 = torch.cuda.Stream(device=torch.cuda.current_device())
            s2 = torch.cuda.Stream(device=torch.cuda.current_device())

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, False, x.shape[1:3])

        # process layers in reversible and irreversible stacks
        stack = []
        for l_i in range(len(self.blocks)):
            if isinstance(self.blocks[l_i], MultiScaleBlock):
                stack.append(("Stage Transition", l_i))
            else:
                if len(stack) == 0 or stack[-1][0] == "Stage Transition":
                    stack.append(("Reversible", []))
                stack[-1][1].append(l_i)

        for i, substack in enumerate(stack):
            if substack[0] == "Stage Transition":
                x = self.blocks[substack[1]](x)
            else:
                # first concat two copies of x for the two streams
                x = torch.cat([x, x], dim=-1)

                if not self.training or self.no_custom_backward:
                    executing_fn = RevViT.vanilla_backward
                elif self.use_fast_backprop:
                    executing_fn = FastRevBackProp.apply
                else:
                    executing_fn = RevBackProp.apply

                x = executing_fn(
                    x, self.blocks[substack[1][0] : substack[1][-1] + 1]
                )

        x = x.reshape(x.shape[0], -1, x.shape[-1]).permute(0, 2, 1)

        x = x.mean(2)

        x = self.norm(x)

        x = self.head(x)

        return x


class FastRevBackProp(RevBackProp):

    """
    Fast backpropagation inheriting from standard reversible backpropagation.
    By parallelizing the backward pass, we can achieve significant speedups
    using a minor increase in memory usage.
    """

    @staticmethod
    def backward(ctx, dx):
        """
        Key differences are separating the logic into two functions:
        (1) backward_pass_recover: which recomputes the activations
        (2) backward_pass_grad: which updates the gradients
        We can perform these two steps in parallel if we stagger which
        layers they are performed on. Hence, we start with the last layer,
        and then run (2) on the current layer and (1) on the next layer
        simultaneously.
        """
        # obtaining gradients dX_1 and dX_2 from the concatenated input
        dX_1, dX_2 = torch.chunk(dx, 2, dim=-1)

        # retrieve the last saved activations, to start rev recomputation
        X_1, X_2 = ctx.saved_tensors
        # layer weights
        layers = ctx.layers

        # Keep a dictionary of events to synchronize on
        # Each is for the completion of a recalculation (f) or gradient calculation (b)
        events = {}
        for i in range(len(layers)):
            events[f"f{i}"] = torch.cuda.Event()
            events[f"b{i}"] = torch.cuda.Event()

        # Run backward staggered on two streams, which were defined globally in __init__
        # Initial pass to recover activations for the "first" layer (i.e. the last layer).
        # prev contains all the info needed to compute the gradients
        with torch.cuda.stream(s1):
            layer = layers[-1]
            prev = layer.backward_pass_recover(Y_1=X_1, Y_2=X_2)
            events["f0"].record(s1)

        # Now stagger streams based on iteration
        for i, (this_layer, next_layer) in enumerate(
            zip(layers[1:][::-1], layers[:-1][::-1])
        ):
            if i % 2 == 0:
                stream1 = s1
                stream2 = s2
            else:
                stream1 = s2
                stream2 = s1

            # This is the gradient update using the previous activations
            with torch.cuda.stream(stream1):
                # b{i} waits on b{i-1}
                if i > 0:
                    events[f"b{i-1}"].synchronize()

                if i % 2 == 0:
                    dY_1, dY_2 = this_layer.backward_pass_grads(
                        *prev, dX_1, dX_2
                    )
                else:
                    dX_1, dX_2 = this_layer.backward_pass_grads(
                        *prev, dY_1, dY_2
                    )

                events[f"b{i}"].record(stream1)

            # This is the activation recomputation
            with torch.cuda.stream(stream2):
                # f{i} waits on f{i-1}
                events[f"f{i}"].synchronize()
                prev = next_layer.backward_pass_recover(
                    Y_1=prev[0], Y_2=prev[1]
                )
                events[f"f{i+1}"].record(stream2)

        # Last iteration
        if len(layers) - 1 % 2 == 0:
            stream2 = s1
        else:
            stream2 = s2
        next_layer = layers[0]

        with torch.cuda.stream(stream2):
            if len(layers) > 1:
                events[f"b{len(layers)-2}"].synchronize()
            if len(layers) - 1 % 2 == 0:
                dY_1, dY_2 = next_layer.backward_pass_grads(*prev, dX_1, dX_2)
                dx = torch.cat([dY_1, dY_2], dim=-1)
            else:
                dX_1, dX_2 = next_layer.backward_pass_grads(*prev, dY_1, dY_2)
                dx = torch.cat([dX_1, dX_2], dim=-1)
            events[f"b{len(layers)-1}"].record(stream2)

        # Synchronize, for PyTorch 1.9
        torch.cuda.current_stream().wait_stream(s1)
        torch.cuda.current_stream().wait_stream(s2)
        torch.cuda.synchronize()
        events[f"b{len(layers)-1}"].synchronize()

        del dX_1, dX_2, dY_1, dY_2, X_1, X_2, prev[:]
        return dx, None, None
