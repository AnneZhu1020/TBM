from typing import Any
import torch as th
from torch import nn
from torch.nn import functional as F

from torch.autograd import Function

from minirl.common.policy import Extractor
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence


class StateEmbeddingNet(Extractor):
    def __init__(self, input_shape, embedding_size) -> None:
        super().__init__(
            extractor_fn="cnn",
            extractor_kwargs={
                "input_shape": input_shape,
                "conv_kwargs": (
                    {
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                    {
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                    {
                        "out_channels": embedding_size,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                ),
                "hiddens": (),
                "activation": nn.ELU,
            },
        )


class ForwardDynamicNet(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.forward_dynamics = nn.Sequential(
            nn.Linear(128 + self.num_actions, 256),
            nn.ReLU(),
        )
        self.fd_out = nn.Linear(256, 128)

    def forward(self, state_embedding, action):
        # Embedding shape: T x B x C
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        inputs = th.cat((state_embedding, action_one_hot), dim=2)
        next_state_emb = self.fd_out(self.forward_dynamics(inputs))
        return next_state_emb

class ForwardDynamicUncertaintyNet(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.forward_dynamics_mean = nn.Sequential(
            nn.Linear(128 + self.num_actions, 256),
            nn.ReLU(),
        )
        self.fd_out_mean = nn.Linear(256, 128)
        self.forward_dynamics_std = nn.Sequential(
            nn.Linear(128 + self.num_actions, 256),
            nn.ReLU(),
        )
        self.fd_out_std = nn.Linear(256, 128)

    @staticmethod
    def normalize(x):
        return x / th.sqrt(th.pow(x, 2).sum(dim=-1, keepdim=True))

    def forward(self, state_embedding, action):
        # Embedding shape: T x B x C
        # state_embedding = self.normalize(state_embedding)

        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        inputs = th.cat((state_embedding, action_one_hot), dim=2)
        next_state_emb_mean = self.fd_out_mean(self.forward_dynamics_mean(inputs))
        next_state_emb_std = self.fd_out_std(self.forward_dynamics_std(inputs))

        return next_state_emb_mean, next_state_emb_std


class InverseDynamicNet(nn.Module):
    def __init__(self, num_actions, embedding_size):
        super().__init__()
        self.num_actions = num_actions
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(2 * embedding_size, 256),
            nn.ReLU(),
        )
        self.id_out = nn.Linear(256, num_actions)

    @staticmethod
    def normalize(x):
        return x / th.sqrt(th.pow(x, 2).sum(dim=-1, keepdim=True))

    def forward(self, state_embedding, next_state_embedding):
        # Embedding shape: T x B x C
        # state_embedding = self.normalize(state_embedding)
        # next_state_embedding = self.normalize(next_state_embedding)

        inputs = th.cat((state_embedding, next_state_embedding), dim=2)
        action_logits = self.id_out(self.inverse_dynamics(inputs))
        return action_logits

class SR_rep(Extractor):
    def __init__(self, input_shape, num_actions, embedding_dim) -> None:
        super().__init__(
            extractor_fn="cnn",
            extractor_kwargs={
                "input_shape": input_shape,
                "conv_kwargs": (
                    {
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                    {
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                    {
                        "out_channels": embedding_dim * num_actions,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                ),
                "hiddens": (),
                "activation": nn.ELU,
            },
        )
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim

    def forward(self, obs, firsts):
        emb, _ = self.extract_features(obs, firsts)
        return emb.reshape(-1, self.num_actions, self.embedding_dim)


class VectorQuantization(Function):
    @staticmethod
    def forward(ctx: Any, inputs, codebook):
        with th.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = th.sum(codebook ** 2, dim=1)
            inputs_sqr = th.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Euclidean distance
            distances = th.addmm(codebook_sqr + inputs_sqr, inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = th.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            
            ctx.mark_non_differentiable(indices)

            return indices
        
    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
                           '`VectorQuantization`. The function `VectorQuantization` '
                           'is not differentiable. Use `VectorQuantizationStraightThrough` '
                           'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx: Any, inputs, codebook):
        indices = vector_quantization(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = th.index_select(codebook, dim=0, index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)


vector_quantization = VectorQuantization.apply
vector_quantization_st = VectorQuantizationStraightThrough.apply
__all__ = [vector_quantization, vector_quantization_st]


class VQEncoder(nn.Module):
    def __init__(self, num_actions, embedding_size, code_dim):
        super().__init__()
        self.num_actions = num_actions
        self.forward_dynamics = nn.Sequential(
            nn.Linear(2 * embedding_size + self.num_actions, 64),
            nn.ReLU(),
        )
        self.fd_out = nn.Linear(64, code_dim)

    def forward(self, state_embedding, action, next_state_embedding):
        # Embedding shape: T x B x C
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        inputs = th.cat((state_embedding, action_one_hot, next_state_embedding), dim=2)
        latent = self.fd_out(self.forward_dynamics(inputs))
        return latent

class VQDecoder(nn.Module):
    def __init__(self, embedding_size, code_dim):
        super().__init__()
        self.forward_dynamics = nn.Sequential(
            nn.Linear(code_dim, 64),
            nn.ReLU(),
        )
        self.fd_out = nn.Linear(64, embedding_size)

    def forward(self, zqx):
        # Embedding shape: T x B x C
        x_decoder = self.fd_out(self.forward_dynamics(zqx))
        return x_decoder


class BackDynamicNet(nn.Module):
    def __init__(self, num_actions, embedding_dim):
        super().__init__()
        self.num_actions = num_actions
        self.backward_dynamics = nn.Sequential(
            nn.Linear(embedding_dim + self.num_actions, 256),
            nn.ReLU(),
        )
        self.fd_out = nn.Linear(256, embedding_dim)

    def forward(self, next_state_embedding, action):
        # Embedding shape: T x B x C
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        inputs = th.cat((next_state_embedding, action_one_hot), dim=2)
        pred_state_emb = self.fd_out(self.backward_dynamics(inputs))
        return pred_state_emb
    

class ResidualBlock(nn.Module):
    """Residual block."""
    def __init__(self, num_channels, filter_size):
        self.num_channels = num_channels
        self.filter_size = filter_size
        self.conv1 = nn.Conv2d(self.filter_size, self.filter_size,
                               kernel_size=[3, 3], 
                               stride=(1, 1),
                               padding='SAME')
        self.conv2 = nn.Conv2d(self.filter_size, self.filter_size,
                               kernel_size=[3, 3], 
                               stride=(1, 1),
                               padding='SAME')
    
    def forward(self, x):
        y = F.relu(x)
        y = self.conv1(y)
        y = F.relu(y)
        y = self.conv2(y)
        return y + x


# From https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / th.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class FiLM(nn.Module):
    def __init__(self, num_actions, out_features=128):
        super().__init__()
        self.num_actions = num_actions
        self.out_features = out_features

        self.weight = nn.Linear(self.num_actions, self.out_features)
        self.bias = nn.Linear(self.num_actions, self.out_features)

        self.apply(initialize_parameters)

    def forward(self, state_emb, action):
        """
        x: state embedding, [bs, 128]
        y: one_hot action
        """
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float() # size([128, 16, 7])

        weight = self.weight(action_one_hot) # size([128, 16, 128])
        bias = self.bias(action_one_hot)
        out = state_emb * weight + bias # size([128, 16, 128])
        return F.relu(out)


class DetectorHead(nn.Module):
    def __init__(self, input_channel=128, max_length=128):
        super(DetectorHead, self).__init__()
        self.max_length = max_length
        self.rnn = nn.LSTM(128, 64, bidirectional=True)
        self.linear1 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        h0 = th.zeros(2, x.size(1), 64).to(x.device)
        c0 = th.zeros(2, x.size(1), 64).to(x.device)

        output, (hn, cn) = self.rnn(x, (h0, c0)) # output: torch.Size([128, 16, 128]); hn, cn: torch.Size([2, 16, 64])
        mask = self.sigmoid(self.linear1(x)) # mask: torch.Size([128, 16, 1])
        return output, mask, hn
        

class TrajectoryTransformer(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_heads=4):
        super(TrajectoryTransformer, self).__init__()

        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )
    
    def forward(self, embedded_sequence):
        # Assuming input_sequence shape is (batch_size, timestep, embedding_size)
        
        # Transformer expects input in the shape (timestep, batch_size, embedding_size)
        embedded_sequence = embedded_sequence.permute(1, 0, 2)
        
        # Self-attention mechanism
        output = self.transformer(embedded_sequence, embedded_sequence)

        # You can further process the output as needed
        return output

# # Example usage
# batch_size = 16
# timestep = 10
# embedding_size = 64
# hidden_size = 128
# num_layers = 2
# num_heads = 4

# # Create an instance of the TrajectoryTransformer
# model = TrajectoryTransformer(embedding_size, hidden_size, num_layers, num_heads)

# # Generate random input data
# input_sequence = torch.randn(batch_size, timestep, embedding_size)

# # Forward pass through the model
# output = model(input_sequence)

