import dataclasses
from typing import Dict, List, Optional, Any

import torch
from torch import nn
from tqdm.auto import tqdm

from .decoder import Decoder
from .encoder import GraphEncoder
from ..comparm import GP
from ..data.synthesis_route import ActionType

from .output_head import (
    BaseFingerprintHead,
    ClassifierHead,
    MultiFingerprintHead,
    ReactantRetrievalResult,
)

class SynthesisGraphTransformer(nn.Module):
    """PROTAC synthesis route generation model with dual graph encoders."""
    
    def __init__(self, 
                 num_atom_classes=12, 
                 num_bond_classes=5, 
                 num_reaction_classes=91,
                 num_reagent_classes=483+1,
                 num_action_types=6,
                 hidden_dim=512, 
                 encoder_depth=8, 
                 decoder_depth=6, 
                 n_heads=8, 
                 dim_head=64, 
                 edge_dim=128, 
                 num_out_fps=1, 
                 fp_dim=256, 
                 warmup_prob=0.01):

        super().__init__()
        
        # Dual graph encoders for warhead and E3 ligand
        self.graph_encoder = GraphEncoder(
            num_atom_classes,
            num_bond_classes,
            dim=hidden_dim,
            depth=encoder_depth,
            dim_head=dim_head,
            edge_dim=edge_dim,
            heads=n_heads,
            rel_pos_emb=False,
            output_norm=False,
        )

        self.merge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Decoder for synthesis route generation
        self.decoder = Decoder(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=512,
            num_layers=decoder_depth,
            pe_max_len=32,
            output_norm=False,
            dim_fp_embed_hidden=256,
            num_reaction_classes=num_reaction_classes,
            num_reagent_classes=num_reagent_classes,
        )

        self.d_model = hidden_dim

        # Output heads for different action types
        self.action_type_head = ClassifierHead(
            self.d_model,
            num_action_types,
            dim_hidden=hidden_dim,
        )

        self.reaction_head = ClassifierHead(
            self.d_model,
            num_reaction_classes,
            dim_hidden=hidden_dim,
        )

        self.reagent_head = ClassifierHead(
            self.d_model,
            num_reagent_classes,  # Placeholder, not used directly
            dim_hidden=hidden_dim, 
        )

    def encode_molecules(self, batch: Dict[str, torch.Tensor]):
        """Encode warhead and E3 ligand molecules."""
        # Encode warhead
        warhead_batch = {
            "atoms": batch["warhead_atom_ids"],
            "bonds": batch["warhead_adjs"], 
            "atom_padding_mask": batch["warhead_atom_mask"].bool()  
        }
        warhead_code, warhead_mask = self.graph_encoder(warhead_batch)
        #print ('encode warhead', warhead_code.shape, warhead_mask.shape)
        # Encode E3 ligand
        e3_batch = {
            "atoms": batch["e3_ligand_atom_ids"],
            "bonds": batch["e3_ligand_adjs"],
            "atom_padding_mask": batch["e3_ligand_atom_mask"].bool()
        }

        e3_code, e3_mask = self.graph_encoder(e3_batch)
        #print ('encode e3_ligand', e3_code.shape, e3_mask.shape)

        # Extract reaction site representations
        warhead_site_mask=batch["warhead_reaction_site_mask"].bool()
        warhead_sites = warhead_code[warhead_site_mask]
        #print ('warhead_sites', warhead_sites.shape)
        e3_site_mask = batch["e3_ligand_reaction_site_mask"].bool()
        e3_sites = e3_code[e3_site_mask]

        B, N, D = warhead_code.shape
        warhead_sites = warhead_sites.unsqueeze(1).expand(-1,N,-1)
        e3_sites = e3_sites.unsqueeze(1).expand(-1,N,-1)
        #print ('warhead_code & warhead_sites',warhead_code.shape,warhead_sites.shape)
        warhead_context = torch.cat([warhead_code,warhead_sites], dim=-1)
        e3_context = torch.cat([e3_code,e3_sites], dim=-1)

        # Combine all molecular representations
        molecular_context = torch.cat([warhead_context, e3_context], dim=1)
        context_compressed = self.merge_head(molecular_context)

        context_mask = torch.cat([
            warhead_mask, e3_mask
        ], dim=1)
        
        return context_compressed, context_mask 

    def get_loss(
        self,
        code: torch.Tensor | None,
        code_padding_mask: torch.Tensor | None,
        action_types: torch.Tensor,
        reaction_indices: torch.Tensor,
        reagent_indices: torch.Tensor,
        token_padding_mask: torch.Tensor,
        **options,
    ):

        h = self.decoder(
            code=code,
            code_padding_mask=code_padding_mask,
            token_types=action_types,
            rxn_indices=reaction_indices,
            reagent_indices=reagent_indices,
            token_padding_mask=token_padding_mask,
        )[:, :-1]

        action_types_gt = action_types[:, 1:].contiguous()
        reaction_indices_gt = reaction_indices[:, 1:].contiguous()
        reagent_indices_gt = reagent_indices[:, 1:].contiguous()

        loss_dict: dict[str, torch.Tensor] = {}

        # Action type prediction loss
        loss_dict["action_type"] = self.action_type_head.get_loss(h, action_types_gt, None)
        
        # Reaction prediction loss (only for REACTION and E3_CONNECTION actions)
        reaction_mask = (action_types_gt == ActionType.REACTION) | (action_types_gt == ActionType.E3_CONNECTION)
        loss_dict["reaction"] = self.reaction_head.get_loss(h, reaction_indices_gt, reaction_mask)

        # Fingerprint prediction loss (only for BUILDING_BLOCK and E3_CONNECTION actions)
        reagent_mask = (action_types_gt == ActionType.BUILDING_BLOCK)
        loss_dict["reagent"] = self.reagent_head.get_loss(h, reagent_indices_gt, reagent_mask)

        return loss_dict

    def get_loss_shortcut(self, batch: Dict[str, torch.Tensor], **options):
        code, code_padding_mask = self.encode_molecules(batch)

        return self.get_loss(
            code=code,
            code_padding_mask=code_padding_mask,
            action_types=batch["action_types"],
            reaction_indices=batch["reaction_indices"],
            reagent_indices=batch["reagent_indices"],
            token_padding_mask=batch["seq_mask"],
            **options,
        )

    def predict(
        self,
        code: torch.Tensor | None,
        code_padding_mask: torch.Tensor | None,
        action_types: torch.Tensor,
        reaction_indices: torch.Tensor,
        reagent_indices: torch.Tensor,
        token_padding_mask: torch.Tensor,
        select_mask: torch.Tensor,
    ):
        h = self.decoder(
                code=code,
                code_padding_mask=code_padding_mask,
                token_types=action_types,
                rxn_indices=reaction_indices,
                reagent_indices=reagent_indices,
                token_padding_mask=token_padding_mask,
        )

        h_next = h[select_mask]  # (bsz, h_dim)
        action_type_logits = self.action_type_head.predict(h_next)
        reaction_logits = self.reaction_head.predict(h_next)
        reagent_logits = self.reagent_head.predict(h_next)
        return action_type_logits,reaction_logits,reagent_logits

