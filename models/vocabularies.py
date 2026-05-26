# vocabularies.py
"""
Vocabulary classes for charges and magistrate sections
"""

import torch
import pandas as pd
from typing import List, Dict, Optional, Union


class ChargeVocabulary:
    """Vocabulary for mapping charge IDs to continuous indices"""
    
    def __init__(self, charges_data: Union[pd.DataFrame, List[str]]):
        """
        Initialize charge vocabulary from either DataFrame or list of charge names
        
        Args:
            charges_data: Either a DataFrame with 'charge_id' and 'charge_name' columns,
                         or a list of charge names
        """
        if isinstance(charges_data, pd.DataFrame):
            # Original format with DataFrame
            names = charges_data['charge_name'].unique().tolist()
            self.name2idx = {n: i for i, n in enumerate(names)}
            self.idx2name = {i: n for i, n in enumerate(names)}
            # Map original IDs to our idx
            self.orig2idx = {
                int(r['charge_id']): self.name2idx[r['charge_name']]
                for _, r in charges_data.iterrows()
            }
        else:
            # New format with list of charge names
            names = list(set(charges_data))  # Ensure unique
            self.name2idx = {n: i for i, n in enumerate(names)}
            self.idx2name = {i: n for i, n in enumerate(names)}
            # For list format, use name as key directly
            self.orig2idx = {i: i for i in range(len(names))}
        
        self.size = len(names)
    
    def encode(self, case_charges: dict, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Encode case charges into a count vector using charge vocabulary mapping
        
        Args:
            case_charges: Dictionary of charge_id -> info (with 'count' field)
            device: PyTorch device for tensor creation
            
        Returns:
            Tensor of shape (1, vocab_size) with charge counts
        """
        if device is None:
            device = torch.device('cpu')
        
        vec = torch.zeros(self.size, device=device)
        
        for charge_key, info in case_charges.items():
            # Try to get index from original ID mapping
            if isinstance(charge_key, (int, str)) and str(charge_key).isdigit():
                idx = self.orig2idx.get(int(charge_key))
            else:
                # Try to get index from name mapping
                idx = self.name2idx.get(charge_key)
            
            if idx is not None:
                count = info.get('count', 1) if isinstance(info, dict) else 1
                vec[idx] = count
        
        return vec.unsqueeze(0)  # Add batch dimension


class MagistrateVocabulary:
    """Vocabulary for one-hot encoding magistrate sections"""
    
    def __init__(self, magistrate_sections: List[str]):
        """
        Initialize magistrate vocabulary with one-hot encoding mapping
        
        Args:
            magistrate_sections: List of magistrate section IDs
        """
        # Ensure all sections are strings and add unknown category
        ids = [str(i) for i in magistrate_sections] + ["unknown"]
        self.map = {mid: idx for idx, mid in enumerate(ids)}
        self.size = len(ids)
        self.idx2mag = {idx: mid for mid, idx in self.map.items()}
    
    def encode(self, mag_id: Union[str, int], device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Encode magistrate ID into one-hot vector with fallback to unknown
        
        Args:
            mag_id: Magistrate section ID
            device: PyTorch device for tensor creation
            
        Returns:
            Tensor of shape (1, vocab_size) as one-hot vector
        """
        if device is None:
            device = torch.device('cpu')
        
        mid = str(mag_id)
        idx = self.map.get(mid, self.map["unknown"])
        vec = torch.zeros(self.size, device=device)
        vec[idx] = 1
        
        return vec.unsqueeze(0)  # Add batch dimension
    
    def decode(self, idx: int) -> str:
        """
        Decode index back to magistrate ID
        
        Args:
            idx: Index in vocabulary
            
        Returns:
            Magistrate section ID string
        """
        return self.idx2mag.get(idx, "unknown")