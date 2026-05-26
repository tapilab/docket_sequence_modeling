# transformer_helpers.py
"""
Helper functions for the transformer model including bond discretization
and temporal encoding functions.
"""

import torch
import pandas as pd
from typing import Optional, Union
from config import get_model_config


def encode_month_year_features(date_obj: Union[pd.Timestamp, str, None], 
                              device: torch.device) -> torch.Tensor:
    """
    Encode date into month-year one-hot features
    
    Args:
        date_obj: Date object (pandas Timestamp, string, or None)
        device: PyTorch device for tensor creation
        
    Returns:
        One-hot tensor of shape (1, temporal_dim)
    """
    model_configuration = get_model_config()
    
    if date_obj is None or pd.isna(date_obj):
        # Default to zeros for missing dates
        return torch.zeros(1, model_configuration.temporal_dim, device=device)
    
    try:
        # Convert to pandas Timestamp if needed
        if not isinstance(date_obj, pd.Timestamp):
            date_obj = pd.to_datetime(date_obj)
        
        # Extract month (0-11) and year index
        month_idx = date_obj.month - 1  # Convert from 1-12 to 0-11
        year_idx = date_obj.year - model_configuration.min_year
        
        # Clamp to valid range
        year_idx = max(0, min(model_configuration.num_years - 1, year_idx))
        
        # Calculate combined index
        combined_idx = year_idx * model_configuration.num_months + month_idx
        
        # Create one-hot vector for the combined month-year
        month_year_one_hot = torch.nn.functional.one_hot(
            torch.tensor(combined_idx, device=device),
            num_classes=model_configuration.temporal_dim
        ).float().unsqueeze(0)
        
        return month_year_one_hot
        
    except Exception as e:
        print(f"Warning: Error encoding date {date_obj}: {e}")
        # Fallback to zeros
        return torch.zeros(1, model_configuration.temporal_dim, device=device)


def discretize_bond_amount(bond_amount: Union[float, int, str, None]) -> int:
    """
    Discretize bond amount into categorical bins
    
    Args:
        bond_amount: Bond amount value
        
    Returns:
        Bin ID (integer from -1 to 7)
    """
    model_configuration = get_model_config()
    
    if bond_amount is None or pd.isna(bond_amount):
        return model_configuration.BOND_BINS['not_set_or_missing']
    
    try:
        amount = float(bond_amount)
    except (ValueError, TypeError):
        return model_configuration.BOND_BINS['not_set_or_missing']
    
    if amount == 0:
        return model_configuration.BOND_BINS['set_at_0']
    elif 0 < amount <= 1000:
        return model_configuration.BOND_BINS['set_0_to_1000']
    elif 1000 < amount <= 5000:
        return model_configuration.BOND_BINS['set_1000_to_5000']
    elif 5000 < amount <= 10000:
        return model_configuration.BOND_BINS['set_5000_to_10000']
    elif 10000 < amount <= 25000:
        return model_configuration.BOND_BINS['set_10000_to_25000']
    elif 25000 < amount <= 50000:
        return model_configuration.BOND_BINS['set_25000_to_50000']
    elif 50000 < amount <= 100000:
        return model_configuration.BOND_BINS['set_50000_to_100000']
    else:
        return model_configuration.BOND_BINS['set_over_100000']


def bond_bin_to_index(bin_id: int) -> int:
    """
    Convert bond bin ID (-1..7) into contiguous indices (0..8)
    
    Args:
        bin_id: Bond bin ID from discretize_bond_amount
        
    Returns:
        Contiguous index for one-hot encoding
    """
    return 0 if bin_id == -1 else bin_id + 1


def bond_one_hot(bin_id: int, device: torch.device) -> torch.Tensor:
    """
    Create one-hot vector for bond bin
    
    Args:
        bin_id: Bond bin ID from discretize_bond_amount
        device: PyTorch device for tensor creation
        
    Returns:
        One-hot tensor of shape (1, NUM_BOND_CLASSES)
    """
    model_configuration = get_model_config()
    idx = bond_bin_to_index(bin_id)
    
    return torch.nn.functional.one_hot(
        torch.tensor(idx, device=device),
        num_classes=model_configuration.NUM_BOND_CLASSES
    ).float().unsqueeze(0)