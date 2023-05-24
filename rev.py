import torch
from torch import nn

# Needed to implement custom backward pass
from torch.autograd import Function as Function

# We use the standard pytorch multi-head attention module
from torch.nn import MultiheadAttention as MHA


class RevViT(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        n_head=8,
        depth=8,
        patch_size=(
            2,
            2,
        ),  # this patch size is used for CIFAR-10
        # --> (32 // 2)**2 = 256 sequence length
        image_size=(32, 32),  # CIFAR-10 image size
        num_classes=10,
        enable_amp=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_head = n_head
        self.depth = depth
        self.patch_size = patch_size

        num_patches = (image_size[0] // self.patch_size[0]) * (
            image_size[1] // self.patch_size[1]
        )

        # Reversible blocks can be treated same as vanilla blocks,
        # any special treatment needed for reversible bacpropagation
        # is contrained inside the block code and not exposed.
        self.layers = nn.ModuleList(
            [
                ReversibleBlock(
                    dim=self.embed_dim,
                    num_heads=self.n_head,
                    enable_amp=enable_amp,
                )
                for _ in range(self.depth)
            ]
        )

        # Boolean to switch between vanilla backprop and
        # rev backprop. See, ``--vanilla_bp`` flag in main.py
        self.no_custom_backward = False

        # Standard Patchification and absolute positional embeddings as in ViT
        self.patch_embed = nn.Conv2d(
            3, self.embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.pos_embeddings = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dim)
        )

        # The two streams are concatenated and passed through a linear
        # layer for final projection. This is the only part of RevViT
        # that uses different parameters/FLOPs than a standard ViT model.
        # Note that fusion can be done in several ways, including
        # more expressive methods like in paper, but we use
        # linear layer + LN for simplicity.
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
        # patchification using conv and flattening
        # + abolsute positional embeddings
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x += self.pos_embeddings

        # the two streams X_1 and X_2 are initialized identically with x and
        # concatenated along the last dimension to pass into the reversible blocks
        x = torch.cat([x, x], dim=-1)

        # no need for custom backprop in eval/inference phase
        if not self.training or self.no_custom_backward:
            executing_fn = RevViT.vanilla_backward
        else:
            executing_fn = RevBackProp.apply

        # This takes care of switching between vanilla backprop and rev backprop
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
    and (B) activation recomputation reversibly in backward for gradient
    calculation. Inspired by
    https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
    """

    @staticmethod
    def forward(
        ctx,
        x,
        layers,
    ):
        """
        Reversible Forward pass.
        Each reversible layer implements its own forward pass pass logic.
        """

        # obtaining X_1 and X_2 from the concatenated input
        X_1, X_2 = torch.chunk(x, 2, dim=-1)

        for layer in layers:
            X_1, X_2 = layer(X_1, X_2)
            all_tensors = [X_1.detach(), X_2.detach()]

        # saving only the final activations of the last reversible block
        # for backward pass, no intermediate activations are needed.
        ctx.save_for_backward(*all_tensors)
        ctx.layers = layers

        return torch.cat([X_1, X_2], dim=-1)

    @staticmethod
    def backward(ctx, dx):
        """
        Reversible Backward pass.
        Each layer implements its own logic for backward pass (both
        activation recomputation and grad calculation).
        """
        # obtaining gradients dX_1 and dX_2 from the concatenated input
        dX_1, dX_2 = torch.chunk(dx, 2, dim=-1)

        # retrieve the last saved activations, to start rev recomputation
        X_1, X_2 = ctx.saved_tensors
        # layer weights
        layers = ctx.layers

        for _, layer in enumerate(layers[::-1]):
            # this is recomputing both the activations and the gradients wrt
            # those activations.
            X_1, X_2, dX_1, dX_2 = layer.backward_pass(
                Y_1=X_1,
                Y_2=X_2,
                dY_1=dX_1,
                dY_2=dX_2,
            )
        # final input gradient to be passed backward to the patchification layer
        dx = torch.cat([dX_1, dX_2], dim=-1)

        del dX_1, dX_2, X_1, X_2

        return dx, None, None


class ReversibleBlock(nn.Module):
    """
    Reversible Blocks for Reversible Vision Transformer.
    See Section 3.3.2 in paper for details.
    """

    def __init__(self, dim, num_heads, enable_amp):
        """
        Block is composed entirely of function F (Attention
        sub-block) and G (MLP sub-block) including layernorm.
        """
        super().__init__()
        # F and G can be arbitrary functions, here we use
        # simple attwntion and MLP sub-blocks using vanilla attention.
        self.F = AttentionSubBlock(
            dim=dim, num_heads=num_heads, enable_amp=enable_amp
        )

        self.G = MLPSubblock(dim=dim, enable_amp=enable_amp)

        # note that since all functions are deterministic, and we are
        # not using any stochastic elements such as dropout, we do
        # not need to control seeds for the random number generator.
        # To see usage with controlled seeds and dropout, see pyslowfast.

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

        # free memory since X_1 is now not needed
        del X_1

        g_Y_1 = self.G(Y_1)

        # Y_2 = X_2 + g(Y_1)
        Y_2 = X_2 + g_Y_1

        # free memory since X_2 is now not needed
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

        And we use pytorch native logic carefully to
        calculate gradients on F and G.
        """

        # temporarily record intermediate activation for G
        # and use them for gradient calculcation of G
        with torch.enable_grad():
            Y_1.requires_grad = True

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            g_Y_1 = self.G(Y_1)

            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            g_Y_1.backward(dY_2, retain_graph=True)

        # activation recomputation is by design and not part of
        # the computation graph in forward pass. Hence we do not
        # need to record it in the computation graph.
        with torch.no_grad():
            # recomputing X_2 from the rev equation
            X_2 = Y_2 - g_Y_1

            # free memory since g_Y_1 is now not needed
            del g_Y_1

            # the gradients for the previous block
            # note that it is called dY_1 but it in fact dX_1 in math.
            # reusing same variable to save memory
            dY_1 = dY_1 + Y_1.grad

            # free memory since Y_1.grad is now not needed
            Y_1.grad = None

        # record F activations and calc gradients on F
        with torch.enable_grad():
            X_2.requires_grad = True

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            f_X_2 = self.F(X_2)

            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            f_X_2.backward(dY_1, retain_graph=True)

        # propagate reverse computed acitvations at the start of
        # the previou block for backprop.s
        with torch.no_grad():
            # recomputing X_1 from the rev equation
            X_1 = Y_1 - f_X_2

            del f_X_2, Y_1
            # the gradients for the previous block
            # note that it is called dY_2 but it in fact dX_2 in math.
            # reusing same variable to save memory
            dY_2 = dY_2 + X_2.grad

            # free memory since X_2.grad is now not needed
            X_2.grad = None

            X_2 = X_2.detach()

        # et voila~
        return X_1, X_2, dY_1, dY_2


class MLPSubblock(nn.Module):
    """
    This creates the function G such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim,
        mlp_ratio=4,
        enable_amp=False,  # standard for ViTs
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.enable_amp = enable_amp

    def forward(self, x):
        # The reason for implementing autocast inside forward loop instead
        # in the main training logic is the implicit forward pass during
        # memory efficient gradient backpropagation. In backward pass, the
        # activations need to be recomputed, and if the forward has happened
        # with mixed precision, the recomputation must also be so. This cannot
        # be handled with the autocast setup in main training logic.
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
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
        enable_amp=False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)

        # using vanilla attention for simplicity. To support adanced attention
        # module see pyslowfast.
        # Note that the complexity of the attention module is not a concern
        # since it is used blackbox as F block in the reversible logic and
        # can be arbitrary.
        self.attn = MHA(dim, num_heads, batch_first=True)
        self.enable_amp = enable_amp

    def forward(self, x):
        # See MLP fwd pass for explanation.
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            x = self.norm(x)
            out, _ = self.attn(x, x, x)
            return out


def main():
    """
    This is a simple test to check if the recomputation is correct
    by computing gradients of the first learnable parameters twice --
    once with the custom backward and once with the vanilla backward.

    The difference should be ~zero.
    """

    # insitantiating and fixing the model.
    model = RevViT()

    # random input, instaintiate and fixing.
    # no need for GPU for unit test, runs fine on CPU.
    x = torch.rand((1, 3, 32, 32))
    model = model

    # output of the model under reversible backward logic
    output = model(x)
    # loss is just the norm of the output
    loss = output.norm(dim=1)

    # computatin gradients with reversible backward logic
    # using retain_graph=True to keep the computation graph.
    loss.backward(retain_graph=True)

    # gradient of the patchification layer under custom bwd logic
    rev_grad = model.patch_embed.weight.grad.clone()

    # resetting the computation graph
    for param in model.parameters():
        param.grad = None

    # switching model mode to use vanilla backward logic
    model.no_custom_backward = True

    # computing forward with the same input and model.
    output = model(x)
    # same loss
    loss = output.norm(dim=1)

    # backward but with vanilla logic, does not need retain_graph=True
    loss.backward()

    # looking at the gradient of the patchification layer again
    vanilla_grad = model.patch_embed.weight.grad.clone()

    # difference between the two gradients is small enough.
    assert (rev_grad - vanilla_grad).abs().max() < 1e-6


if __name__ == "__main__":
    main()
