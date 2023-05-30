import torch
from torch import nn

# Inherit mostly from base RevViT
from rev import RevBackProp, ReversibleBlock, RevViT


class FastRevViT(RevViT):
    def __init__(self, enable_amp=False, **kwargs):
        super().__init__(**kwargs)

        # For Fast parallel revprop
        # Initialize global streams on current device
        global s1, s2
        s1 = torch.cuda.default_stream(device=torch.cuda.current_device())
        s2 = torch.cuda.Stream(device=torch.cuda.current_device())

        # Then override the reversible blocks with finer functions
        self.layers = nn.ModuleList(
            [
                FineReversibleBlock(
                    dim=self.embed_dim,
                    num_heads=self.n_head,
                    enable_amp=enable_amp,
                )
                for _ in range(self.depth)
            ]
        )

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
            executing_fn = FastRevBackProp.apply

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


class FastRevBackProp(RevBackProp):

    """
    Fast backpropagation inheriting from standard reversible backpropagation.
    By parallelizing the backward pass, we can achieve significant speedups
    using a minor increase in memory usage.
    Simplified version of original.
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
        events[f"b{len(layers)-1}"].synchronize()

        del dX_1, dX_2, dY_1, dY_2, X_1, X_2, prev[:]
        return dx, None, None


class FineReversibleBlock(ReversibleBlock):
    """
    Reversible Block with fine-grained backwards functions.
    Specifically, backward is now two functions:
        (1) backward_pass_recover: which recomputes the activations
        (2) backward_pass_grads: which updates the gradients
    See PaReprop paper for more details.
    """

    def backward_pass_recover(self, Y_1, Y_2):
        """
        Activation recomputation for recovering activations only.
        """
        with torch.enable_grad():
            Y_1.requires_grad = True
            g_Y_1 = self.G(Y_1)

        with torch.no_grad():
            X_2 = Y_2 - g_Y_1

        with torch.enable_grad():
            X_2.requires_grad = True
            f_X_2 = self.F(X_2)

        with torch.no_grad():
            X_1 = Y_1 - f_X_2

        # Keep tensors around to do backprop on the graph.
        ctx = [X_1, X_2, Y_1, g_Y_1, f_X_2]
        return ctx

    def backward_pass_grads(self, X_1, X_2, Y_1, g_Y_1, f_X_2, dY_1, dY_2):
        """
        Receive intermediate activations and inputs to backprop through
        and update gradients.
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
