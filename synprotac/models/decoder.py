from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import nn

from ..data.synthesis_route import ActionType
from .transformer.positional_encoding import PositionalEncoding


def _SimpleMLP(dim_in: int, dim_out: int, dim_hidden: int) -> Callable[[torch.Tensor], torch.Tensor]:
    return nn.Sequential(
        nn.Linear(dim_in, dim_hidden),
        nn.ReLU(),
        nn.Linear(dim_hidden, dim_hidden),
        nn.ReLU(),
        nn.Linear(dim_hidden, dim_out),
    )


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 512,
        num_layers: int = 6,
        pe_max_len: int = 32,
        output_norm: bool = False,
        dim_fp_embed_hidden: int = 512,
        num_reaction_classes: int = 91,
        num_reagent_classes: int = 483,
    ):
        super().__init__()
        self.d_model = d_model

        self.in_token = nn.Embedding(max(ActionType) + 1, d_model)
        self.in_reaction = nn.Embedding(num_reaction_classes, d_model)
        self.in_reagent = nn.Embedding(num_reagent_classes, d_model)

        self.pe_dec = PositionalEncoding(d_model=d_model, max_len=pe_max_len)
        self.dec = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model) if output_norm else None,
        )

    def get_empty_code(
        self,
        batch_size: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        code = torch.zeros([batch_size, 0, self.model_dim], device=device).float()
        code_padding_mask = torch.zeros([batch_size, 0], dtype=torch.bool, device=device)

        return code, code_padding_mask

    def embed(
        self,
        token_types: torch.Tensor,
        rxn_indices: torch.Tensor,
        reagent_indices: torch.Tensor,
    ) -> torch.Tensor:

        emb_token = self.in_token(token_types)
        emb_rxn = self.in_reaction(rxn_indices)
        emb_reagent = self.in_reagent(reagent_indices)
        token_types_expand = token_types.unsqueeze(-1).expand([token_types.size(0), token_types.size(1), self.d_model])
        emb_token = torch.where(token_types_expand == ActionType.REACTION, emb_rxn, emb_token)
        emb_token = torch.where(token_types_expand == ActionType.BUILDING_BLOCK, emb_reagent, emb_token)
        emb_token = torch.where(token_types_expand == ActionType.E3_CONNECTION, emb_reagent, emb_token)
        emb_token = self.pe_dec(emb_token)

        return emb_token

    def forward(
        self,
        code: torch.Tensor | None,
        code_padding_mask: torch.Tensor | None,
        token_types: torch.Tensor,
        rxn_indices: torch.Tensor,
        reagent_indices: torch.Tensor,
        token_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:

        bsz, seqlen = token_types.size()
        if code is None:
            code, code_padding_mask = self.get_empty_code(bsz, device=reagent_indices.device)

        x = self.embed(token_types, rxn_indices, reagent_indices)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=x.size(1),
            dtype=x.dtype,
            device=x.device,
        )
        tgt_key_padding_mask = (
            torch.zeros(
                [bsz, seqlen],
                dtype=causal_mask.dtype,
                device=causal_mask.device,
            ).masked_fill_(token_padding_mask, -torch.finfo(causal_mask.dtype).max)
            if token_padding_mask is not None
            else None
        )
        y: torch.Tensor = self.dec(
            tgt=x,
            memory=code,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=code_padding_mask,
        )  # (bsz, seq_len, d_model)

        return y

    if TYPE_CHECKING:

        def __call__(
            self,
            code: torch.Tensor | None,
            code_padding_mask: torch.Tensor | None,
            token_types: torch.Tensor,
            rxn_indices: torch.Tensor,
            reagent_indices: torch.Tensor,
            token_padding_mask: torch.Tensor | None,
        ) -> torch.Tensor: ...
