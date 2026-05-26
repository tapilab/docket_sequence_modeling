# Python file containing the embedding classes for the transformer model

import torch
import torch.nn as nn
import numpy as np
import h5py
import pickle
import json
import gc
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from typing import Union
import os

from utils.cache_directory_utils import CacheDirectoryManager
from models.vocabularies import ChargeVocabulary, MagistrateVocabulary
from config import get_model_config

# SECTION Cached BERT embedding

class StreamingCachedBertEmbedding(nn.Module):
    """Streaming BERT embedding module with cached embeddings and index lookup"""
    
    def __init__(self, cache_dir: Union[str, Path, None] = None, out_dim: int = 768, cache_manager = None):
        super().__init__()
        if cache_dir is None:
            cache_dir = cache_manager.get_cache_dir("bert")
        
        self.cache_dir = Path(cache_dir)
        self.out_dim = out_dim
        self.projector = torch.nn.Identity()
        
        # Load index
        index_file = self.cache_dir / "proceeding_index.pkl"
        
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found in {self.cache_dir}")
        
        with open(index_file, 'rb') as f:
            self.index = pickle.load(f)
        
        
        # Store path to embedding file
        self.embedding_file = self.cache_dir / "embeddings.h5"

        #open it once
        self._h5 = None
        self._pid = None
        self._embedding_ds = None

    def _ensure_h5_open(self):
        pid = os.getpid()
        if self._h5 is None or self._pid != pid:
            if self._h5 is not None:
                self._h5.close()
            self._h5 = h5py.File(self.embedding_file, 'r')
            self._embedding_ds = self._h5['embeddings']
            self._pid = pid
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the open HDF5 file handle from the state
        state['_h5'] = None
        state['_embedding_ds'] = None
        state['_pid'] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
    
    def forward(self, proceeding_id: str) -> torch.Tensor:
        """Retrieve cached embedding for given proceeding ID"""
        # Check if there are parameters, otherwise default to CPU
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        
        if proceeding_id and (proceeding_id.startswith('initial_') or proceeding_id == 'INITIAL_STATE'):
            return torch.zeros(1, self.out_dim, device=device)
        
        embedding_idx = self.index.get(proceeding_id)
        if embedding_idx is None:
            return torch.zeros(1, self.out_dim, device=device)
        self._ensure_h5_open()
        embedding = self._embedding_ds[embedding_idx]
        return torch.as_tensor(embedding, device=device).unsqueeze(0)
    
# SECTION Embed defendant demographics

class DefendantRoleEmbedding(nn.Module):
    """Embedding for defendant demographics (race, gender, age)"""
    
    def __init__(self, num_races=6, num_genders=3, max_age=100,
                 embed_dim=8, out_dim=32):
        super().__init__()
        self.race_emb = nn.Embedding(num_races, embed_dim)
        self.gender_emb = nn.Embedding(num_genders, embed_dim)
        self.age_linear = nn.Linear(1, embed_dim)
        self.max_age = max_age
        #set the output dimension for the sake of interpretability
        self.out_dim = out_dim
        
        nn.init.xavier_uniform_(self.race_emb.weight)
        nn.init.xavier_uniform_(self.gender_emb.weight)
        nn.init.xavier_uniform_(self.age_linear.weight)
        nn.init.zeros_(self.age_linear.bias)
        
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * 3, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, race_id, gender_id, age):
        """Embed defendant demographics by fusing race, gender, and age embeddings"""
        current_device = next(self.parameters()).device
        
        # Ensure tensors
        r = self.race_emb(torch.tensor([race_id], dtype=torch.long, device=current_device))
        g = self.gender_emb(torch.tensor([gender_id], dtype=torch.long, device=current_device))
        age_value = age if age is not None else 25.0  # default to 25 if missing
        age_value = float(min(max(age_value, 0), self.max_age))  # clip to valid range
        a = self.age_linear(torch.tensor([[age_value]], dtype=torch.float, device=current_device))
        
        return self.fuse(torch.cat([r, g, a], dim=1))

# SECTION Embed case charge information

class CaseInfoEmbedding(nn.Module):
    """Embedding for case charge information"""
    
    def __init__(self, vocab: ChargeVocabulary, embedding_dim=192):
        super().__init__()
        self.vocab = vocab
        #set the output dimension for the sake of interpretability
        self.out_dim = embedding_dim
        
        self.projector = nn.Sequential(
            nn.LayerNorm(vocab.size),
            nn.Linear(vocab.size, embedding_dim),
            nn.ReLU()
        )
        
        nn.init.xavier_uniform_(self.projector[1].weight)
        nn.init.zeros_(self.projector[1].bias)
    
    def forward(self, case_charges: dict) -> torch.Tensor:
        """Project case charges into embedding space using vocabulary encoding"""
        current_device = next(self.parameters()).device
        counts = self.vocab.encode(case_charges, device=current_device)
        return self.projector(counts)
    

# SECTION Embed magistrate section information
class MagistrateInfoEmbedding(nn.Module):
    """Embedding for magistrate section information"""
    
    def __init__(self, vocab: MagistrateVocabulary, embedding_dim=32):
        super().__init__()
        self.vocab = vocab
        #set the output dimension for the sake of interpretability
        self.out_dim = embedding_dim
        self.projector = nn.Sequential(
            nn.LayerNorm(vocab.size),
            nn.Linear(vocab.size, embedding_dim),
            nn.ReLU()
        )
        nn.init.xavier_uniform_(self.projector[1].weight)
        nn.init.zeros_(self.projector[1].bias)
    
    def forward(self, mag_id) -> torch.Tensor:
        """Project magistrate ID into embedding space using one-hot encoding"""
        current_device = next(self.parameters()).device
        onehot = self.vocab.encode(mag_id, device=current_device)
        return self.projector(onehot)
    
# SECTION Class for extracting and caching BERT embeddings
class BertEmbeddingExtraction:
    """Class for extracting and caching BERT embeddings"""
    
    def __init__(self, model_name="bert-base-uncased", out_dim=768,
                 cache_dir: Union[str, Path, None] = None, batch_size=32, cache_manager = None,
                 device=None):
        self.model_name = model_name
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = cache_manager.get_cache_dir("bert")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name).to(self.device)
        self.bert.eval()
        
        # Add projection layer if needed (handle base vs large correctly)
        self.hidden_size = int(self.bert.config.hidden_size)  # 768 for base, 1024 for large, etc.

        if out_dim != self.hidden_size:
            self.projector = nn.Linear(self.hidden_size, out_dim).to(self.device)
            nn.init.xavier_uniform_(self.projector.weight)
            nn.init.zeros_(self.projector.bias)
        else:
            self.projector = None

    
    def extract_text_embedding(self, text: str) -> np.ndarray:
        """Extract BERT embedding for a single text"""
        if not text or text.strip() == "":
            return np.zeros(self.out_dim, dtype=np.float32)
        
        # Tokenize and get BERT output
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :]
            
            # Apply projection if needed
            if self.projector is not None:
                embedding = self.projector(embedding)
        
        return embedding.cpu().numpy().squeeze()
    
    def stream_processing_embeddings_to_file(self, data: dict, force_recompute: bool = False):
        """Stream process embeddings and save to file"""
        embedding_file = self.cache_dir / "embeddings.h5"
        index_file = self.cache_dir / "proceeding_index.pkl"
        processed_index_file = self.cache_dir / "processed_index.json"
        
        # Check if already processed
        if not force_recompute and embedding_file.exists() and index_file.exists():
            print(f"Embeddings already cached at {self.cache_dir}")
            return True
        
        # Collect all proceedings with text
        all_proceeding_info = []
        
        # Process mag data
        if 'mag' in data and not data['mag'].empty:
            for _, row in data['mag'].iterrows():
                text = row.get('text', '')
                if text and text.strip():
                    all_proceeding_info.append({
                        'proceeding_id': row.get('proceeding_id'),
                        'text': text
                    })
        
        # Process cdc data
        if 'cdc' in data and not data['cdc'].empty:
            for _, row in data['cdc'].iterrows():
                text = row.get('text', '')
                if text and text.strip():
                    all_proceeding_info.append({
                        'proceeding_id': row.get('proceeding_id'),
                        'text': text
                    })
        
        if not all_proceeding_info:
            print("No proceedings with text found")
            return False
        
        print(f"Processing {len(all_proceeding_info)} proceedings...")
        
        # Load existing processed index if it exists
        processed_ids = set()
        if processed_index_file.exists() and not force_recompute:
            with open(processed_index_file, 'r') as f:
                processed_ids = set(json.load(f))
        
        # Filter to only unprocessed proceedings
        new_proceedings = [p for p in all_proceeding_info if p['proceeding_id'] not in processed_ids]
        
        if new_proceedings:
            print(f"Processing {len(new_proceedings)} new proceedings...")
            
            # Create/append to HDF5 file
            mode = 'a' if embedding_file.exists() else 'w'
            with h5py.File(embedding_file, mode) as h5f:
                # Initialize or get existing dataset
                if 'embeddings' not in h5f:
                    max_shape = (None, self.out_dim)
                    embeddings_ds = h5f.create_dataset(
                        'embeddings',
                        shape=(0, self.out_dim),
                        maxshape=max_shape,
                        dtype='float32',
                        chunks=(1, self.out_dim)
                    )
                    start_idx = 0
                else:
                    embeddings_ds = h5f['embeddings']
                    start_idx = embeddings_ds.shape[0]
                
                # Process in batches
                new_index = {}
                for i in tqdm(range(0, len(new_proceedings), self.batch_size), desc="Extracting embeddings"):
                    batch = new_proceedings[i:i+self.batch_size]
                    batch_embeddings = []
                    
                    for proc_info in batch:
                        embedding = self.extract_text_embedding(proc_info['text'])
                        batch_embeddings.append(embedding)
                    
                    # Resize dataset and add new embeddings
                    batch_size = len(batch_embeddings)
                    embeddings_ds.resize((start_idx + batch_size, self.out_dim))
                    embeddings_ds[start_idx:start_idx+batch_size] = np.array(batch_embeddings)
                    
                    # Update index
                    for j, proc_info in enumerate(batch):
                        new_index[proc_info['proceeding_id']] = start_idx + j
                    
                    start_idx += batch_size
                    
                    # Periodic cleanup
                    if i % (self.batch_size * 10) == 0:
                        gc.collect()
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
            
            # Update proceeding index
            if index_file.exists():
                with open(index_file, 'rb') as f:
                    existing_index = pickle.load(f)
                existing_index.update(new_index)
            else:
                existing_index = new_index
            
            with open(index_file, 'wb') as f:
                pickle.dump(existing_index, f)
            
            # Update processed index
            processed_ids.update(new_index.keys())
            with open(processed_index_file, 'w') as f:
                json.dump(list(processed_ids), f)
            
            print(f"Total proceedings with text: {len(all_proceeding_info)}")
            print(f"Total embeddings stored: {len(existing_index)}")
        else:
            print("No new embeddings to process")
        
        return True
    
    def get_embedding(self, proceeding_id: str) -> torch.Tensor:
        """Retrieve cached embedding for a specific proceeding ID"""
        # Load index
        index_file = self.cache_dir / "proceeding_index.pkl"
        embedding_file = self.cache_dir / "embeddings.h5"
        
        if not index_file.exists():
            raise FileNotFoundError(f"No index file found at {index_file}")
        
        with open(index_file, 'rb') as f:
            index = pickle.load(f)
        
        # Get embedding index directly from proceeding_id
        if proceeding_id not in index:
            return torch.zeros(1, self.out_dim)
        
        embedding_idx = index[proceeding_id]
        
        # Read embedding from HDF5
        with h5py.File(embedding_file, 'r') as h5f:
            embedding = h5f['embeddings'][embedding_idx]
        
        return torch.tensor(embedding).unsqueeze(0)  # Add batch dimension
    
    def cleanup(self):
        """Clean up model resources and clear GPU memory"""
        del self.bert
        del self.tokenizer
        if hasattr(self, 'projector') and self.projector is not None:
            del self.projector
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()


# SECTION Extract and cache BERT embeddings using streaming processing
def extract_and_cache_bert_embeddings_streaming(data: dict, cache_dir: Union[str, Path, None] = None,
                                               model_name: str = "bert-base-uncased",
                                               out_dim: int = 768, batch_size: int = 32,
                                               device=None, force_recompute: bool = False,
                                               cache_manager = None):
    """Extract and cache BERT embeddings using streaming processing"""
    model_configuration = get_model_config()
    
    # Proper path handling
    if cache_dir is None:
        cache_dir = cache_manager.get_cache_dir("bert")
    
    
    
    # Create extractor
    extractor = BertEmbeddingExtraction(
        model_name=model_name,
        out_dim=out_dim,
        cache_dir=cache_dir,
        batch_size=batch_size,
        cache_manager = cache_manager,
        device=device
    )
    
    try:
        # Process proceedings
        success = extractor.stream_processing_embeddings_to_file(data, force_recompute=force_recompute)
        if success:
            return extractor.cache_dir
        else:
            return None
    finally:
        # Clean up resources
        extractor.cleanup()