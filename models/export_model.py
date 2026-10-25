"""
Complete script to export a trained transformer model and all dependencies
"""
import pickle
import torch
from pathlib import Path
from utils.cache_directory_utils import CacheDirectoryManager
from config import get_model_config

def export_model_complete(
    model,
    label_processor,
    model_name="final_model",
    cache_manager = None
):
    output_dir = cache_manager.get_cache_dir('exported_models')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exported_files = {}
    
    # 1. Save model state dict
    model_path = output_dir / f"{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    exported_files['model_weights'] = model_path
    print(f"Saved model weights: {model_path}")
    
    # 2. Save label processor
    label_processor_path = output_dir / 'label_processor.pkl'
    with open(label_processor_path, 'wb') as f:
        pickle.dump(label_processor, f)
    exported_files['label_processor'] = label_processor_path
    print(f"Saved label processor: {label_processor_path}")
    
    # 3. Save vocabularies
    charge_vocab_path = output_dir / 'charge_vocab.pkl'
    with open(charge_vocab_path, 'wb') as f:
        pickle.dump(model.charge_vocab, f)
    exported_files['charge_vocab'] = charge_vocab_path
    print(f"Saved charge vocabulary: {charge_vocab_path}")
    
    mag_vocab_path = output_dir / 'mag_vocab.pkl'
    with open(mag_vocab_path, 'wb') as f:
        pickle.dump(model.mag_vocab, f)
    exported_files['mag_vocab'] = mag_vocab_path
    print(f"Saved magistrate vocabulary: {mag_vocab_path}")
    
    # 4. Save model configuration parameters
    model_config = get_model_config()
    config_dict = {
        'num_labels': model.num_labels,
        'hidden_dim': model.hidden_dim,
        'max_len': model.max_len,
        'temporal_dim': model_config.temporal_dim,
        'bond_dim': model_config.NUM_BOND_CLASSES,
        'min_year': model_config.min_year,
        'max_year': model_config.max_year,
        'num_months': model_config.num_months,
        'label_names': model_config.LABEL_NAMES,
        'enable_interpretability': model.enable_interpretability if hasattr(model, 'enable_interpretability') else False
    }
    
    config_path = output_dir / 'model_config.pkl'
    with open(config_path, 'wb') as f:
        pickle.dump(config_dict, f)
    exported_files['model_config'] = config_path
    print(f"Saved model configuration: {config_path}")
    
    # 5. Create a manifest file with all paths
    manifest = {
        'model_weights': str(model_path),
        'label_processor': str(label_processor_path),
        'charge_vocab': str(charge_vocab_path),
        'mag_vocab': str(mag_vocab_path),
        'model_config': str(config_path),
        'bert_cache_dir': str(cache_manager.get_cache_dir('bert')),
        'export_timestamp': str(torch.cuda.Event(enable_timing=False).record() if torch.cuda.is_available() else 'cpu')
    }
    
    manifest_path = output_dir / f'{model_name}_manifest.pkl'
    with open(manifest_path, 'wb') as f:
        pickle.dump(manifest, f)
    exported_files['manifest'] = manifest_path
    print(f"Saved manifest: {manifest_path}")
    
    print("\n" + "="*60)
    print("MODEL EXPORT COMPLETE")
    print("="*60)
    print(f"Export directory: {output_dir}")
    print(f"\nTo load this model in another notebook, use:")
    print(f"  from load_model import load_model_complete")
    print(f"  model, label_processor = load_model_complete('{output_dir}')")
    print("="*60)
    
    return exported_files