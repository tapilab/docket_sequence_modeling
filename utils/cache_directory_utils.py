# Python file containing the utility functions associated with the cache directory
from pathlib import Path


# utils/cache_utils.py
"""
Cache directory utilities
"""
from pathlib import Path

class CacheDirectoryManager:
    def __init__(self, base_path = None):
        if base_path is None:
            self.base_path = Path('/data/krosen5/cache/transformers/Transformer')
        else:
            self.base_path = Path(base_path)
        
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def get_cache_dir(self, cache_type: str) -> Path:
        cache_dirs = {
            'bert': self.base_path / 'bert_streaming',
            'labels': self.base_path / 'label_cache',
            'models': self.base_path / 'saved_models',
            'ngrams': self.base_path / 'ngram_analysis',
            'regular_expression_reports': self.base_path / 'regex_reports',
            'clerk_analysis': self.base_path / 'clerk_analysis',
            'exports': self.base_path / 'exports',
            'datasets': self.base_path / 'datasets',
            'confident_misclassification': self.base_path / 'confident_misclassification',
            'ablation_study': self.base_path / 'ablation_study',
            'interpretability_results': self.base_path / 'interpretability_results',
            'exported_models': self.base_path / 'exported_models'
        }

        cache_dir = cache_dirs.get(cache_type, self.base_path / cache_type)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir