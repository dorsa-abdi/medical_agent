"""Compact Delphi-compatible generative transformer.

Implements the core published design: disease-token embeddings, continuous age
encoding, causal attention, a competing-event head and exponential time-to-event
likelihood. It does not contain the restricted UK Biobank-trained weights.
"""
from dataclasses import asdict, dataclass
import math


@dataclass
class DelphiConfig:
    vocab_size: int
    block_size: int = 96
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 120
    dropout: float = 0.0
    t_min_days: float = 0.1
    pad_token_id: int = 0

    def as_dict(self):
        return asdict(self)


def build_model(config: DelphiConfig):
    """Construct lazily so non-Delphi code can run without PyTorch installed."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class AgeEncoding(nn.Module):
        def __init__(self):
            super().__init__()
            div = torch.exp(torch.arange(0, config.n_embd, 2) * (-math.log(10000) / config.n_embd))
            self.register_buffer("div_term", div)
            self.projection = nn.Linear(config.n_embd, config.n_embd, bias=False)

        def forward(self, age_days):
            angle = age_days.unsqueeze(-1) / 365.25 * self.div_term
            encoded = torch.zeros(*angle.shape[:-1], config.n_embd, device=age_days.device)
            encoded[..., 0::2], encoded[..., 1::2] = torch.sin(angle), torch.cos(angle)
            return self.projection(encoded)

    class DelphiModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=config.pad_token_id)
            self.age_embedding = AgeEncoding()
            layer = nn.TransformerEncoderLayer(
                d_model=config.n_embd, nhead=config.n_head,
                dim_feedforward=4 * config.n_embd, dropout=config.dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.blocks = nn.TransformerEncoder(layer, config.n_layer)
            self.norm = nn.LayerNorm(config.n_embd)
            self.event_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.event_head.weight = self.token_embedding.weight

        def forward(self, token_ids, age_days, targets=None, target_age_days=None):
            length = token_ids.shape[1]
            if length > config.block_size:
                raise ValueError(f"Sequence length {length} exceeds {config.block_size}")
            causal = torch.triu(torch.ones(length, length, device=token_ids.device, dtype=torch.bool), diagonal=1)
            padding = token_ids.eq(config.pad_token_id)
            hidden = self.token_embedding(token_ids) + self.age_embedding(age_days.float())
            hidden = self.norm(self.blocks(hidden, mask=causal, src_key_padding_mask=padding))
            logits = self.event_head(hidden)
            if targets is None:
                return {"event_logits": logits}
            valid = targets.ne(-1) & targets.ne(config.pad_token_id)
            event_loss = F.cross_entropy(logits[valid], targets[valid])
            dt = (target_age_days - age_days).clamp_min(config.t_min_days)
            log_total_rate = (
                torch.logsumexp(logits, dim=-1)
                - math.log(365.25 * config.vocab_size)
            ).clamp(-12, 12)
            total_rate = log_total_rate.exp()
            time_nll = -(log_total_rate - total_rate * dt)[valid].mean()
            return {"event_logits": logits,
                    "event_loss": event_loss, "time_loss": time_nll,
                    "loss": event_loss + time_nll}

        @torch.no_grad()
        def horizon_probabilities(self, token_ids, age_days, horizon_days):
            output = self(token_ids, age_days)
            log_rates = (
                output["event_logits"][:, -1] - math.log(365.25 * config.vocab_size)
            ).clamp(-12, 12)
            rates = log_rates.exp()
            total_rate = rates.sum(-1, keepdim=True).clamp_min(1e-12)
            any_event = 1 - torch.exp(-total_rate * horizon_days)
            return rates / total_rate * any_event

    return DelphiModel()
