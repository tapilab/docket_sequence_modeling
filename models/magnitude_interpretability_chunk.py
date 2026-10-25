# magnitude_study_worker.py  <- save this as a file in your project root
import torch
import pickle
import h5py
import types
import threading
from pathlib import Path
from models.transformer_model import CaseTransformerModel
from models.interpretability import CaseTransformerInterpretability
from utils.cache_directory_utils import CacheDirectoryManager
from config import get_model_config
import pandas as pd



def _run_magnitude_chunk(args):
    """Runs in a separate process - each gets its own GPU and model copy."""
    (chunk_seqs, chunk_labels, gpu_idx, fold_idx,
     base_cache, output_dir, chunk_idx,
     vocab_path, temporal_dim, bond_dim, age_bins, prior_case_bins) = args

    device     = torch.device(f'cuda:{gpu_idx}')
    base_cache = Path(base_cache)
    output_dir = Path(output_dir)
    cache_mgr  = CacheDirectoryManager(base_path=base_cache)
    model_cfg  = get_model_config()

    with open(base_cache / 'saved_models' / 'label_processor.pkl', 'rb') as f:
        label_processor = pickle.load(f)

    with open(vocab_path, 'rb') as f:
        charge_vocab, mag_vocab = pickle.load(f)
    print(f'  [GPU {gpu_idx}] Vocabs loaded')

    pt_path = base_cache / 'saved_models' / f'k_fold_fold_{fold_idx + 1}' / \
              f'final_model_mlb_fold_{fold_idx + 1}.pt'

    model = CaseTransformerModel(
        charge_vocab=charge_vocab,
        mag_vocab=mag_vocab,
        bert_cache_dir=cache_mgr.get_cache_dir('bert'),
        bert_out_dim=768,
        num_labels=model_cfg.num_labels,
        bond_dim=bond_dim,
        temporal_dim=temporal_dim,
        cache_manager=cache_mgr,
    )
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()
    model.to(device)
    print(f'  [GPU {gpu_idx}] Model loaded')

    def _ensure_h5_open(self):
        if not hasattr(self._local, 'h5') or self._local.h5 is None:
            self._local.h5           = h5py.File(self.embedding_file, 'r')
            self._local.embedding_ds = self._local.h5['embeddings']

    def _forward(self, proceeding_id):
        device_ = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        if proceeding_id and (proceeding_id.startswith('initial_') or proceeding_id == 'INITIAL_STATE'):
            return torch.zeros(1, self.out_dim, device=device_)
        idx = self.index.get(proceeding_id)
        if idx is None:
            return torch.zeros(1, self.out_dim, device=device_)
        self._ensure_h5_open()
        emb = self._local.embedding_ds[idx]
        return torch.as_tensor(emb, device=device_).unsqueeze(0)

    model.text_emb._local          = threading.local()
    model.text_emb._ensure_h5_open = types.MethodType(_ensure_h5_open, model.text_emb)
    model.text_emb.forward         = types.MethodType(_forward, model.text_emb)

    model_cfg.update_date_range(2008, 2025)

    interpreter = CaseTransformerInterpretability(
        model=model, label_processor=label_processor,
        verbose=False, device=device,
    )

    result = interpreter.run_counterfactual_magnitude_study(
        val_sequences=chunk_seqs,
        val_labels=chunk_labels,
        output_dir=output_dir / f'chunk_{chunk_idx}',
        max_samples=len(chunk_seqs),
        parallel=False,
        tqdm_position=chunk_idx,
        age_bins=age_bins,
        prior_case_bins=prior_case_bins,
    )
    for feat, df in result['raw'].items():
        if not df.empty:
            df.to_pickle(output_dir / f'chunk_{chunk_idx}_{feat}.pkl')

    return chunk_idx