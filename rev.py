import sys
import torch
from torch import nn

# Needed to implement custom backward pass
from torch.autograd import Function as Function
# We use the standard pytorch multi-head attention module
from torch.nn import MultiheadAttention as MHA 

class RevViT(nn.Module):

    def __init__(self, 
                embed_dim = 768, 
                n_head = 8, 
                depth = 8, 
                patch_size = (2, 2), 
                image_size = (32, 32), 
                num_classes = 10,
                ):
    
        super().__init__()

        self.embed_dim = embed_dim 
        self.n_head = n_head
        self.depth = depth
        self.patch_size = patch_size

        num_patches = (image_size[0] // self.patch_size[0]) * (image_size[1] // self.patch_size[1])
        
        self.layers = nn.ModuleList([
                    ReversibleBlock(dim=self.embed_dim, num_heads=self.n_head)
                    for _ in range(self.depth)])
                    
        self.no_custom_backward = False

        self.patch_embed = nn.Conv2d(3, 
                                    self.embed_dim, 
                                    kernel_size=patch_size, 
                                    stride=patch_size)

        self.pos_embeddings = nn.Parameter(
                    torch.zeros(
                        1, num_patches, self.embed_dim
                    ))

        self.head = nn.Linear(2 * self.embed_dim, num_classes, bias=True)
        self.norm = nn.LayerNorm(2 * self.embed_dim)

    @staticmethod
    def vanilla_backward(h, layers):
        """
        Using rev layers without rev backpropagation. Debugging purposes only.
        Activated with self.no_custom_backward.
        """
        # split into hidden states (h) and attention_output (a)
        h, a = torch.chunk(h, 2, dim=-1)
        for _, layer in enumerate(layers):
            a, h = layer(a, h)

        return torch.cat([a, h], dim=-1)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x += self.pos_embeddings

        x = torch.cat([x, x], dim=-1)

        # no need for custom backprop in eval/model stat log
        if not self.training or self.no_custom_backward:
            executing_fn = RevViT.vanilla_backward
        else:
            executing_fn = RevBackProp.apply

        x = executing_fn(
            x,
            self.layers,
        )

        # aggregate across sequence length  
        x = x.mean(1)
        
        # head pre-norm  
        x = self.norm(x)

        # pre-softmax logits 
        x = self.head(x)

        # return pre-softmax logits
        return x

class RevBackProp(Function):
    """
    Custom Backpropagation function to allow (A) flusing memory in foward
    and (B) activation recomputation reversibly in backward for gradient calculation.
    Inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
    """

    @staticmethod
    def forward(
        ctx,
        x,
        layers,
    ):
        """
        Reversible Forward pass. Any intermediate activations from `buffer_layers` are
        cached in ctx for forward pass. This is not necessary for standard usecases.
        Each reversible layer implements its own forward pass logic.
        """


        X_1, X_2 = torch.chunk(x, 2, dim=-1)

        intermediate = []

        for layer in layers:
            X_1, X_2 = layer(X_1, X_2)
            all_tensors = [X_1.detach(), X_2.detach()]

        ctx.save_for_backward(*all_tensors)
        ctx.layers = layers

        return torch.cat([X_1, X_2], dim=-1)

    @staticmethod
    def backward(ctx, dx):
        """
        Reversible Backward pass. Any intermediate activations from `buffer_layers` are
        recovered from ctx. Each layer implements its own loic for backward pass (both
        activation recomputation and grad calculation).
        """
        dX_1, dX_2 = torch.chunk(dx, 2, dim=-1)

        # retrieve params from ctx for backward
        X_1, X_2, *int_tensors = ctx.saved_tensors

        layers = ctx.layers

        for _, layer in enumerate(layers[::-1]):

            X_1, X_2, dX_1, dX_2 = layer.backward_pass(
                Y_1=X_1,
                Y_2=X_2,
                dY_1=dX_1,
                dY_2=dX_2,
            )

        dx = torch.cat([dX_1, dX_2], dim=-1)

        del int_tensors
        del dX_1, dX_2, X_1, X_2

        return dx, None, None

class ReversibleBlock(nn.Module):
    """
    Reversible Blocks for Reversible Vision Transformer and also
    for state-preserving blocks in Reversible MViT. See Section
    3.3.2 in paper for details.
    """

    def __init__(
        self,
        dim,
        num_heads,
    ):
        """
        Block is composed entirely of function F (Attention
        sub-block) and G (MLP sub-block) including layernorm.
        """
        super().__init__()

        self.F = AttentionSubBlock(dim=dim, num_heads=num_heads)

        self.G = MLPSubblock(dim=dim)

        self.seeds = {}

    def forward(self, X_1, X_2):
        """
        forward pass equations:
        Y_1 = X_1 + Attention(X_2), F = Attention
        Y_2 = X_2 + MLP(Y_1), G = MLP
        """

        # Y_1 : attn_output
        f_X_2 = self.F(X_2)

        # Y_1 = X_1 + f(X_2)
        Y_1 = X_1 + f_X_2

        # free memory
        del X_1

        g_Y_1 = self.G(Y_1)

        # Y_2 = X_2 + g(Y_1)
        Y_2 = X_2 + g_Y_1

        del X_2

        return Y_1, Y_2

    def backward_pass(
        self,
        Y_1,
        Y_2,
        dY_1,
        dY_2,
    ):
        """
        equation for activation recomputation:
        X_2 = Y_2 - G(Y_1), G = MLP
        X_1 = Y_1 - F(X_2), F = Attention
        """

        # temporarily record intermediate activation for G
        # and use them for gradient calculcation of G
        with torch.enable_grad():

            Y_1.requires_grad = True

            g_Y_1 = self.G(Y_1)

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

            f_X_2 = self.F(X_2)

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


class MLPSubblock(nn.Module):
    """
    This creates the function G such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim,
        mlp_ratio = 4,
    ):

        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
            )         
        

    def forward(self, x):
        return self.mlp(self.norm(x))


class AttentionSubBlock(nn.Module):
    """
    This creates the function F such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim,
        num_heads,
    ):

        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)

        # This will be set externally during init
        self.thw = None

        # the actual attention details are the same as Multiscale
        # attention for MViTv2 (with channel up=projection inside block)
        # can also implement no upprojection attention for vanilla ViT
        self.attn = MHA(dim, num_heads)
        
    def forward(self, x):
        x = self.norm(x)
        out, _ = self.attn(x, x , x)
        return out

# model = RevViT()
# x = torch.rand((1, 3, 32, 32)).cuda()
# model = model.cuda()
# output = model(x)
# loss = output.norm(dim = 1)
# loss.backward(retain_graph=True)

# rev_grad = model.patch_embed.weight.grad.clone()

# for param in model.parameters():
#     param.grad = None

# model.no_custom_backward = True
# output = model(x)
# loss = output.norm(dim = 1)
# loss.backward()
# vanilla_grad = model.patch_embed.weight.grad.clone()
# print((rev_grad - vanilla_grad).abs().max())