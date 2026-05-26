from pathLib
from datetime import datetime
from typing import Optional, Union

class FileNamingManager:
    def __init__(
        self,
        run_timestamp: Optional[str] = None,
        fold_idx: Optional[int] = None,
        ablation_name: Optional[str] = None,
        use_timestamp: bool = True
    ):
        self.run_timestamp = run_timestamp or (datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp else "")
        self.fold_idx = fold_idx
        self.ablation_name = ablation_name
        self.use_timestamp = use_timestamp
    def get_filename(
        self,
        base_name: str,
        extension: str = ".xslx",
        fold_idx: Optional[int] = None,
        ablation_name: Optional[str] = None,
        include_timestamp: Optional[bool] = None):

        use_fold = fold_idx if fold_idx is not None else self.fold_idx
        use_ablation = ablation_name if ablation_name is not None else self.ablation
        use_timestamp = include_timestamp if include_timestamp is not None else self.use_timestamp
        parts = [base_name]
        if use_fold is not None:
            parts.append(f"fold{use_fold + 1}")
        if use_ablation:
            sanitized_ablation = use_ablation.replace(" ", "_").replace("/", "_")
            parts.append(sanitized_ablation)
        if use_timestamp and self.run_timestamp:
            parts.append(self.run_timestamp)
        filename = "_".join(parts) + extension
        return filename
    
    def get_filepath(
        self,
        directory: Union[str, Path],
        base_name: str,
        extension: str = ".xlsx",
        fold_idx: Optional[int] = None,
        ablation_name: Optional[str] = None,
        include_timestamp: Optional[bool] = None
    ):
        directory_path = Path(directory)
        filename = self.get_filename(
            base_name,
            extension,
            fold_idx,
            ablation_name,
            include_timestamp
        )
        return directory_path / filename
    
    def create_child_manager(
        self,
        fold_idx: Optional[int] = None,
        ablation_name: Optional[str] = None):
        return FileNamingManager(
            run_timestamp=self.run_timestamp,
            fold_idx=fold_idx if fold_idx is not None else self.fold_idx,
            ablation_name=ablation_name if ablation_name is not None else self.ablation_name,
            use_timestamp=self.use_timestamp
        )

def get_fold_filename(
    base_name: str,
    fold_idx: int,
    extension: str = ".xlsx",
    timestamp: Optional[str] = None):
    manager = FileNamingManager(run_timestamp=timestamp, fold_idx=fold_idx)
    return manager.get_filename(base_name, extension)

def get_ablation_filename(
    base_name: str,
    ablation_name: str,
    extension: str = ".xlsx",
    timestamp: Optional[str] = None):
    manager = FileNamingManager(run_timestamp=timestamp, ablation_name=ablation_name)
    return manager.get_filename(base_name, extension)

def get_fold_ablation_filename(
    base_name: str,
    fold_idx: int,
    ablation_name: str,
    extension: str = ".xlsx",
    timestamp: Optional[str] = None):
    manager = FileNamingManager(run_timestamp=timestamp, fold_idx=fold_idx, ablation_name=ablation_name)
    return manager.get_filename(base_name, extension)