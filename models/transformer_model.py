"""
Unified transformer model that supports both case-level and proceeding-level prediction.
Switch between modes using the prediction_level parameter.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Dict, List, Optional

from models.embeddings import (
    StreamingCachedBertEmbedding, 
    DefendantRoleEmbedding, 
    CaseInfoEmbedding, 
    MagistrateInfoEmbedding
)
from models.transformer_helpers import (
    discretize_bond_amount, 
    encode_month_year_features, 
    bond_one_hot
)
from config import get_model_config


class CaseTransformerModel(nn.Module):
    """
    Transformer model that can perform either case-level or proceeding-level prediction.
    """
    
    def __init__(
        self,
        charge_vocab,
        mag_vocab,
        bert_cache_dir: Union[str, Path, None] = None,
        num_races: int = 6,
        num_genders: int = 2,
        max_age: int = 100,
        bert_out_dim: int = 768,
        hidden_dim: int = 896,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.2,
        max_len: int = 200,
        num_labels: int = None,
        bond_dim: int = None,
        temporal_dim: int = None,
        enable_interpretability: bool = False,
        cache_manager = None
    ):
        super().__init__()
        
        model_configuration = get_model_config()
        
        # Set default values from config if not provided
        if temporal_dim is None:
            temporal_dim = model_configuration.temporal_dim
        if bond_dim is None:
            bond_dim = model_configuration.NUM_BOND_CLASSES
        if num_labels is None:
            num_labels = model_configuration.num_labels
        
        # Store vocabularies
        self.charge_vocab = charge_vocab
        self.mag_vocab = mag_vocab
        self.num_labels = num_labels
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.enable_interpretability = enable_interpretability

        
        # Use cached BERT embeddings instead of live BERT
        self.text_emb = StreamingCachedBertEmbedding(bert_cache_dir, bert_out_dim, cache_manager)
        
        # Other embeddings
        self.def_emb = DefendantRoleEmbedding(
            num_races, num_genders + 1, max_age,
            embed_dim=8,
            out_dim=32
        )
        self.case_emb = CaseInfoEmbedding(
            vocab=charge_vocab, embedding_dim=192
        )
        self.mag_emb = MagistrateInfoEmbedding(
            vocab=mag_vocab, embedding_dim=32
        )
        
        # Store dimensions
        self.temporal_dim = temporal_dim
        self.bond_dim = bond_dim
        self.bert_out_dim = bert_out_dim
        
        # Embed teacher-forced labels
        self.prior_label_emb = nn.Linear(num_labels, 96)
        nn.init.xavier_uniform_(self.prior_label_emb.weight)
        nn.init.zeros_(self.prior_label_emb.bias)

        self.feature_dims = {
            "Defendant_Embedding": self.def_emb.out_dim,           # 32
            "Case_Embedding": self.case_emb.out_dim,               # 192
            "BERT_Text_Embedding": bert_out_dim,                   # 768
            "Magistrate_Embedding": self.mag_emb.out_dim,          # 32
            "Prior_Label_Embedding": self.prior_label_emb.out_features,  # 96
            "Prior_Charge_Embedding": self.case_emb.out_dim,       # 192 (reuses case_emb)
            "Court_Type": 1,
            "Original_Bond": bond_dim,
            "Sum_Of_Bonds": bond_dim,
            "Prior_Case_Count": 1,
            "Temporal_Features": temporal_dim,
        }
        
        # Calculate total input dimension from feature_dims
        hidden_dim_totals = (
            self.feature_dims["Defendant_Embedding"] +
            self.feature_dims["Case_Embedding"] +
            self.feature_dims["BERT_Text_Embedding"] +
            self.feature_dims["Magistrate_Embedding"] +
            self.feature_dims["Prior_Label_Embedding"] +
            self.feature_dims["Prior_Charge_Embedding"] +
            self.feature_dims["Court_Type"] +
            self.feature_dims["Original_Bond"] +
            self.feature_dims["Sum_Of_Bonds"] +
            self.feature_dims["Prior_Case_Count"] +
            self.feature_dims["Temporal_Features"]
        )

        
        
        # Fuse into single token
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim_totals, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        nn.init.normal_(self.pos, mean=0, std=0.02)
        
        # Transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        
        # Initialize encoder weights
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Classifier head
        self.classifier = nn.Linear(hidden_dim, num_labels) if num_labels else None

    def encode_proceeding(self, pdict: dict, proceeding_idx: int, hidden_features: int) -> torch.Tensor:
        """
        Encode a single proceeding into embedding vector
        
        Args:
            pdict: Dictionary containing proceeding information
            proceeding_idx: Index of this proceeding in the sequence
            hidden_features: Number of hidden features (unused but kept for compatibility)
            
        Returns:
            Encoded proceeding tensor of shape (1, hidden_dim)
        """
        current_device = next(self.parameters()).device
        model_configuration = get_model_config()
        
        # Get embeddings
        de = self.def_emb(
            pdict.get("race_id", 5),
            pdict.get("gender_id", 2),
            pdict.get("age", 25)
        ).to(current_device)
        
        ce = self.case_emb(pdict.get("charges", {})).to(current_device)
        
        # Cached BERT embedding
        proceeding_id = pdict.get("proceeding_id")
        te = self.text_emb(proceeding_id).to(current_device)
        
        # Court type feature
        court_type = pdict.get("type", "mag")
        court_type_feature = torch.tensor(
            [[1.0 if court_type == "cdc" else 0.0]],
            dtype=torch.float32,
            device=current_device
        )
        
        # Magistrate embedding
        me = self.mag_emb(pdict.get("magistrate_section", "unknown")).to(current_device)
        
        # Bond features with special handling for initial state
        if proceeding_id and proceeding_id.startswith('initial_'):
            # Initial state - use special "not_set_or_missing" category
            orig_bin = model_configuration.BOND_BINS["not_set_or_missing"]
            sum_bin = model_configuration.BOND_BINS["not_set_or_missing"]
        elif proceeding_idx == 0:
            # First real proceeding but no bond info yet
            orig_bin = model_configuration.BOND_BINS["not_set_or_missing"]
            sum_bin = model_configuration.BOND_BINS["not_set_or_missing"]
        else:
            # Normal processing
            orig_bin = discretize_bond_amount(pdict.get("original_bond"))
            sum_bin = discretize_bond_amount(pdict.get("sum_of_bonds"))
        
        # Sanity check: bond_dim should equal number of classes for one-hot
        assert self.bond_dim == model_configuration.NUM_BOND_CLASSES, (
            f"bond_dim ({self.bond_dim}) must equal NUM_BOND_CLASSES "
            f"({model_configuration.NUM_BOND_CLASSES}) for one-hot encoding."
        )
        
        orig_bond_feat = bond_one_hot(orig_bin, current_device)  # (1, bond_dim)
        sum_bond_feat = bond_one_hot(sum_bin, current_device)    # (1, bond_dim)
        
        # Prior case count
        prior_case_count = float(pdict.get("prior_case_count", 0))
        prior_case_feat = torch.tensor(
            [[prior_case_count]],
            dtype=torch.float32,
            device=current_device
        )
        
        # Prior charges embedding
        prior_charges = pdict.get("all_prior_charges", {})
        prior_charge_emb = self.case_emb(prior_charges).to(current_device)
        
        # Prior labels embedding
        if "true_prior_labels" in pdict:
            prev_labels = pdict["true_prior_labels"]
            prev_tensor = torch.tensor(
                prev_labels,
                dtype=torch.float32,
                device=current_device
            ).unsqueeze(0)
            prior_label_feat = self.prior_label_emb(prev_tensor)
        else:
            prior_label_feat = torch.zeros(
                1, self.prior_label_emb.out_features,
                device=current_device
            )
        
        # Temporal features
        proceeding_date = pdict.get("date")
        temporal_feat = encode_month_year_features(proceeding_date, current_device)
        
        # Combine court_type + bonds + prior_case_count
        feature_block = torch.cat(
            [court_type_feature, orig_bond_feat, sum_bond_feat, prior_case_feat],
            dim=1
        )
        
        # Combine all features
        tensors_to_cat = [
            de.to(current_device),
            ce.to(current_device),
            te.to(current_device),
            me.to(current_device),
            prior_label_feat.to(current_device),
            prior_charge_emb.to(current_device),
            feature_block.to(current_device),
            temporal_feat.to(current_device)
        ]
        
        combined = torch.cat(tensors_to_cat, dim=1)
        return self.fuse(combined)
    def encode_proceeding_unfused(self, pdict: dict, proceeding_idx: int) -> torch.Tensor:
        """
        Encode proceeding WITHOUT the fusion layer - for interpretability.
        Returns all feature components concatenated but not fused.
        
        Returns:
            Tensor of shape (1, total_feature_dim) with all components preserved
        """
        current_device = next(self.parameters()).device
        model_configuration = get_model_config()
        
        # Get embeddings (copy from encode_proceeding)
        de = self.def_emb(
            pdict.get("race_id", 5),
            pdict.get("gender_id", 2),
            pdict.get("age", 25)
        ).to(current_device)
        
        ce = self.case_emb(pdict.get("charges", {})).to(current_device)
        
        proceeding_id = pdict.get("proceeding_id")
        te = self.text_emb(proceeding_id).to(current_device)
        
        court_type = pdict.get("type", "mag")
        court_type_feature = torch.tensor(
            [[1.0 if court_type == "cdc" else 0.0]],
            dtype=torch.float32,
            device=current_device
        )
        
        me = self.mag_emb(pdict.get("magistrate_section", "unknown")).to(current_device)
        
        # Bond features
        if proceeding_id and proceeding_id.startswith('initial_'):
            orig_bin = model_configuration.BOND_BINS["not_set_or_missing"]
            sum_bin = model_configuration.BOND_BINS["not_set_or_missing"]
        elif proceeding_idx == 0:
            orig_bin = model_configuration.BOND_BINS["not_set_or_missing"]
            sum_bin = model_configuration.BOND_BINS["not_set_or_missing"]
        else:
            orig_bin = discretize_bond_amount(pdict.get("original_bond"))
            sum_bin = discretize_bond_amount(pdict.get("sum_of_bonds"))
        
        orig_bond_feat = bond_one_hot(orig_bin, current_device)
        sum_bond_feat = bond_one_hot(sum_bin, current_device)
        
        prior_case_count = float(pdict.get("prior_case_count", 0))
        prior_case_feat = torch.tensor(
            [[prior_case_count]],
            dtype=torch.float32,
            device=current_device
        )
        
        prior_charges = pdict.get("all_prior_charges", {})
        prior_charge_emb = self.case_emb(prior_charges).to(current_device)
        
        # Prior labels
        if "true_prior_labels" in pdict:
            prev_labels = pdict["true_prior_labels"]
            prev_tensor = torch.tensor(
                prev_labels,
                dtype=torch.float32,
                device=current_device
            ).unsqueeze(0)
            prior_label_feat = self.prior_label_emb(prev_tensor)
        else:
            prior_label_feat = torch.zeros(
                1, self.prior_label_emb.out_features,
                device=current_device
            )
        
        proceeding_date = pdict.get("date")
        temporal_feat = encode_month_year_features(proceeding_date, current_device)
        
        feature_block = torch.cat(
            [court_type_feature, orig_bond_feat, sum_bond_feat, prior_case_feat],
            dim=1
        )
        
        # Concatenate ALL features WITHOUT fusing
        tensors_to_cat = [
            de.to(current_device),
            ce.to(current_device),
            te.to(current_device),
            me.to(current_device),
            prior_label_feat.to(current_device),
            prior_charge_emb.to(current_device),
            feature_block.to(current_device),
            temporal_feat.to(current_device)
        ]
        
        combined = torch.cat(tensors_to_cat, dim=1)
        return combined  # DON'T call self.fuse()
    
    def forward(self, seq: List[Dict], return_all_timesteps = True) -> torch.Tensor:
        """
        Process sequence of proceedings through transformer encoder
        
        Args:
            seq: List of proceeding dictionaries
            
        Returns:
            Output tensor from classifier or encoder
        """
        current_device = next(self.parameters()).device

        #seq 
        seq = seq[-self.max_len:]
        T = len(seq)

        if T == 0:
            if return_all_timesteps:
                return torch.zeros(1, 1, self.fuse[0].out_features, device=current_device)
            else:
                return torch.zeros(1, self.num_labels, device=current_device)

        hidden_features = self.fuse[0].in_features

        
        # Encode each proceeding
        reps = [
            self.encode_proceeding(p, i, hidden_features)
            for i, p in enumerate(seq)
        ]
        
        # Stack representations
        x = torch.stack(reps, dim=1)
        
        # Add positional encoding
        x = x + self.pos[:, :T, :]
        
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=current_device)

        # Pass through encoder
        out = self.encoder(x, is_causal=True, mask=causal_mask)
        
        # Apply classifier to last timestep if available
        if self.classifier:
            if return_all_timesteps:
                # Return logits for ALL timesteps: (1, T, num_labels)
                logits = self.classifier(out)  # (1, T, num_labels)
                return logits
            else:
                return self.classifier(out[:, -1, :])
        return out if return_all_timesteps else out[:, -1, :]