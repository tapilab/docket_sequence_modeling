# transformer_creation.py
"""
Functions for creating transformer models with vocabularies and embeddings
"""

import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
from transformers import AutoConfig

from models.transformer_model import CaseTransformerModel
from models.embeddings import BertEmbeddingExtraction
from models.vocabularies import ChargeVocabulary, MagistrateVocabulary
from config import get_model_config
from utils.cache_directory_utils import CacheDirectoryManager


def analyze_date_range_in_data(data: Dict) -> Tuple[Optional[int], Optional[int]]:
    print("Analyzing date range in data...")
    date_stats = {}
    all_min, all_max = [], []

    for source in ('mag', 'cdc'):
        df = data.get(source)
        if df is None or df.empty or 'date' not in df.columns:
            continue

        parsed = pd.to_datetime(df['date'], errors='coerce').dropna()

        if parsed.empty:
            print(f"No valid dates found in {source} data.")
            continue

        min_date, max_date = parsed.min(), parsed.max()
        date_stats[source] = {'min_date': min_date, 'max_date': max_date, 'count': len(parsed)}
        print(f"{source.upper()} data - Date range: {min_date} to {max_date}, Count: {len(parsed)}")

        all_min.append(min_date)
        all_max.append(max_date)

    if all_min:
        min_year, max_year = min(all_min).year, max(all_max).year
        print(f"Overall date range in data: {min_year} - {max_year}")
        return min_year, max_year

    print("No valid dates found in the entire dataset.")
    return None, None


def create_vocabularies_from_data(data: Dict) -> Tuple[ChargeVocabulary, MagistrateVocabulary]:
    """
    Create charge and magistrate vocabularies from data
    
    Args:
        data: Dictionary containing dataframes with charge and magistrate information
        
    Returns:
        Tuple of (ChargeVocabulary, MagistrateVocabulary)
    """
    # Process charges
    charge_data = []
    charge_sources = []
    
    # First check if we have a proper charges DataFrame with charge_id and charge_name
    if 'charges' in data and not data['charges'].empty:
        df = data['charges']
        if 'charge_id' in df.columns and 'charge_name' in df.columns:
            # Use DataFrame format for ChargeVocabulary
            charge_vocab = ChargeVocabulary(df)
            print(f"Creating charge vocabulary from charges DataFrame with {charge_vocab.size} unique charges")
        else:
            # Fall back to extracting charge descriptions
            for col_name in ['charge_description', 'charge_name', 'charge']:
                if col_name in df.columns:
                    valid_charges = df[col_name].dropna().astype(str).unique().tolist()
                    charge_data.extend(valid_charges)
                    charge_sources.append(f"charges ({len(valid_charges)} unique)")
                    break
    
    # If we don't have a proper charges DataFrame, look in mag_charges and cdc_charges
    if 'charge_vocab' not in locals():
        for df_name in ['mag_charges', 'cdc_charges']:
            if df_name in data and not data[df_name].empty:
                df = data[df_name]
                
                # Check if it has the proper structure
                if 'charge_id' in df.columns and 'charge_name' in df.columns:
                    charge_vocab = ChargeVocabulary(df)
                    print(f"Creating charge vocabulary from {df_name} DataFrame with {charge_vocab.size} unique charges")
                    break
                
                # Otherwise extract charge descriptions
                for col_name in ['charge_description', 'charge_name', 'charge']:
                    if col_name in df.columns:
                        valid_charges = df[col_name].dropna().astype(str).unique().tolist()
                        charge_data.extend(valid_charges)
                        charge_sources.append(f"{df_name} ({len(valid_charges)} unique)")
                        break
    
    # If we still don't have a vocab, create from list
    if 'charge_vocab' not in locals():
        if not charge_data:
            print("No charge data found in the provided data.")
            charge_data = ["UNKNOWN_CHARGE"]
        
        print(f"Creating charge vocabulary from list with {len(set(charge_data))} unique charges")
        if charge_sources:
            print(f"  Sources: {', '.join(charge_sources)}")
        charge_vocab = ChargeVocabulary(charge_data)
    
    # Process magistrates
    mag_ids = []
    
    if 'mag' in data and not data['mag'].empty:
        df = data['mag']
        if 'magistrate_section' in df.columns:
            valid_mags = df['magistrate_section'].dropna().astype(str).unique().tolist()
            mag_ids.extend(valid_mags)
            print(f"Found {len(valid_mags)} unique magistrate sections in mag data.")
        else:
            print("Warning: 'magistrate_section' column not found in mag dataframe.")
    
    if not mag_ids:
        print("No magistrate section data found in the provided data.")
        mag_ids = ["unknown"]
    
    print(f"Creating magistrate vocabulary of size {len(set(mag_ids))}")
    mag_vocab = MagistrateVocabulary(mag_ids)
    
    return charge_vocab, mag_vocab


def bert_embedding_check(data: Dict, cache_dir: Path, config, device = None, cache_manager = None, parameters_dict: Optional[Dict] = None) -> bool:
    """
    Check and create BERT embeddings if needed
    
    Args:
        data: Dictionary containing dataframes with text data
        cache_dir: Path to cache directory
        config: Model configuration object
        
    Returns:
        True if embeddings are ready, False otherwise
    """
    print("Checking BERT embeddings cache...")
    
    # Ensure cache_dir is a Path object
    if not isinstance(cache_dir, Path):
        cache_dir = Path(cache_dir)
    
    embedding_file = cache_dir / "embeddings.h5"
    index_file = cache_dir / "proceeding_index.pkl"
    processed_index_file = cache_dir / "processed_index.json"
    
    if embedding_file.exists() and index_file.exists():
        print("BERT embeddings cache found.")
        return True
    
    print(f"  Extracting BERT embeddings to {cache_dir}")
    print("  This may take a while for large datasets...")
    
    try:
        #get the model name from your parameters or config
        model_name = None
        if parameters_dict is not None:
            model_name = parameters_dict.get("bert_model_to_use")
        if model_name is None and hasattr(config, "bert_model_to_use"):
            model_name = config.bert_model_to_use
        if model_name is None:
            model_name = "bert-base-uncased"  # default

        #get the batch size for BERT processing
        bert_batch_size = 32 # default
        if parameters_dict is not None:
            bert_batch_size = parameters_dict.get("bert_batch_size", bert_batch_size)
        elif hasattr(config, "bert_batch_size"):
            bert_batch_size = config.bert_batch_size
        
        
        #derive output dim from the HF config
        hf_configuration = AutoConfig.from_pretrained(model_name)
        bert_out_dim = int(getattr(hf_configuration, "hidden_size", 768))

        print(f"  Using embedding model: {model_name}")
        print(f"  Detected hidden_size: {bert_out_dim}")


        extractor = BertEmbeddingExtraction(
            model_name=model_name,
            out_dim=bert_out_dim,
            cache_dir=cache_dir,
            batch_size= bert_batch_size,
            cache_manager = cache_manager,
            device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        success = extractor.stream_processing_embeddings_to_file(data, force_recompute=False)
        
        if success:
            print("BERT embeddings extraction completed successfully.")
            return True
        else:
            print("BERT embeddings extraction failed.")
            return False
            
    except Exception as e:
        print(f"Error during BERT embeddings extraction: {e}")
        return False
        
    finally:
        if 'extractor' in locals():
            extractor.cleanup()
        torch.cuda.empty_cache()


def create_model_from_data(data: Dict, cache_dir: Union[str, Path, None] = None, device = None, cache_manager = None, parameters_dict: Optional[Dict] = None, bert_out_dim: Optional[int] = None) -> CaseTransformerModel:
    """
    Create model with cached BERT embeddings
    
    Args:
        data: Dictionary containing training data
        cache_dir: Path to cache directory for BERT embeddings
        
    Returns:
        Initialized CaseTransformerModel
    """
    model_configuration = get_model_config()
    
    if cache_dir is None:
        cache_dir = cache_manager.get_cache_dir("bert")
    else:
        cache_dir = Path(cache_dir)
    
    # Analyze and update date range
    min_year, max_year = analyze_date_range_in_data(data)
    if min_year and max_year:
        model_configuration.update_date_range(min_year, max_year)
        print(f"Date range in data: {min_year} to {max_year}")
        print(f"Using MIN_YEAR={model_configuration.min_year}, NUM_YEARS={model_configuration.num_years}")
        print(f"Updated temporal dimensions: {model_configuration.temporal_dim} features "
              f"({model_configuration.num_months} months × {model_configuration.num_years} years)")
    else:
        print(f"Using default date range settings: {model_configuration.min_year} to {model_configuration.max_year}")
        print(f"Temporal dimensions: {model_configuration.temporal_dim} features "
              f"({model_configuration.num_months} months × {model_configuration.num_years} years)")
    
    # Create vocabularies
    charge_vocab, mag_vocab = create_vocabularies_from_data(data)

    
    # Check/create BERT embeddings
    if device is None:
        embeddings_checked = bert_embedding_check(data, cache_dir, model_configuration, cache_manager, parameters_dict)
    else:
        embeddings_checked = bert_embedding_check(data, cache_dir, model_configuration, device, cache_manager, parameters_dict)
    if not embeddings_checked:
        raise RuntimeError("BERT embeddings could not be created or found. Aborting model creation.")
    
    # Create model with cached embeddings
    return CaseTransformerModel(
        charge_vocab=charge_vocab,
        mag_vocab=mag_vocab,
        bert_cache_dir=cache_dir,
        bert_out_dim=bert_out_dim,
        num_labels=model_configuration.num_labels,
        bond_dim=model_configuration.NUM_BOND_CLASSES,
        temporal_dim=model_configuration.temporal_dim,
        cache_manager=cache_manager
    )

    