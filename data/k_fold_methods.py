import copy
import torch
import numpy as np
from sklearn.model_selection import KFold

from training.trainer import train_case_transformer_with_mlb
from training.evaluation import evaluate_with_teacher_forcing
from notebook_helpers import evaluate_case_level
from utils.cache_directory_utils import CacheDirectoryManager


class KFoldCVRunner:
    def __init__(
        self,
        data,
        model_creation_func,
        label_processor,
        device,
        n_splits=5,
        epochs=10,
        batch_size=16,
        cache_manager=None,
        shuffle=True,
        use_mixed_precision=False,
        use_learning_rate_scheduler=False,
        gradient_clip_norm=1.0,
        random_state=None,
        weighted_loss=False,    
        ):
        self.data = list(data)
        self.model_creation_func = model_creation_func
        self.label_processor = label_processor
        self.device = device
        self.n_splits = n_splits
        self.epochs = epochs
        self.batch_size = batch_size
        self.cache_manager = cache_manager
        self.shuffle = shuffle
        self.random_state = random_state
        self.fold_results = []
        self.aggregate_metrics = None
        self.best_fold_idx = None
        self.weighted_loss = weighted_loss
        self.use_mixed_precision = use_mixed_precision
        self.use_learning_rate_scheduler = use_learning_rate_scheduler
        self.gradient_clip_norm = gradient_clip_norm

    def _iter_folds(self):
        kf = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.data), start=1):
            train_fold = [self.data[i] for i in train_idx]
            val_fold = [self.data[i] for i in val_idx]
            #return a tuple of fold index, train fold, val fold for each fold
            yield fold_idx, train_fold, val_fold
    
    def _train_single_fold(self, fold_idx, train_fold, val_fold):
        print(f"Training fold {fold_idx}/{self.n_splits}...")
        print(f"  train: {len(train_fold)}  val: {len(val_fold)}")
        optional_tag = f"fold_{fold_idx}" if fold_idx is not None else None
        model = self.model_creation_func()
        model.to(self.device)

        directory = self.cache_manager.get_cache_dir('models')/f'k_fold_fold_{fold_idx}'
        #make it into a directory if it doesn't exist
        directory.mkdir(parents=True, exist_ok=True)

        trained_model, logger = train_case_transformer_with_mlb(
            model=model,
            sequences_and_labels=train_fold,
            label_processor=self.label_processor,
            val_data=val_fold,
            epochs=self.epochs,
            batch_size=self.batch_size,
            device=self.device,
            directory_for_output=directory,
            optional_tag=optional_tag,
            use_mixed_precision=self.use_mixed_precision,
            use_learning_rate_scheduler=self.use_learning_rate_scheduler,
            gradient_clip_norm=self.gradient_clip_norm,
            weighted_loss=self.weighted_loss,
            cache_manager = self.cache_manager
        )
        return trained_model, logger
    
    def _prepare_val_for_eval(self, val_fold):
        val_data_for_eval = []
        for seq, labels_seq_raw in val_fold:
            try:
                labels_seq_binary = self.label_processor.transform_sequence(labels_seq_raw)
                tensors_for_steps = [
                    torch.tensor(step_bin, dtype=torch.float32, device=self.device)
                    for step_bin in labels_seq_binary
                ]
                val_data_for_eval.append((seq, tensors_for_steps))
            except Exception as e:
                print(f"Warning: could not process val seq: {e}")
        return val_data_for_eval

    def _evaluate_single_fold(self, trained_model, val_data_for_eval, fold_idx):
        optional_tag = f"fold_{fold_idx + 1}"
        val_metrics, _ = evaluate_with_teacher_forcing(
            trained_model,
            val_data_for_eval,
            self.device,
            label_processor=self.label_processor,
            epoch=self.epochs,
            optional_tag=optional_tag,
            cache_manager = self.cache_manager
        )
        return val_metrics
    
    def _record_fold_result(
        self,
        fold_idx,
        train_fold,
        val_fold,
        val_data_for_eval,
        val_metrics,
        trained_model,
        logger
    ):
        fold_record = {
            "fold_idx": fold_idx,
            "train_data": train_fold,
            "val_data": val_fold,
            "val_data_for_eval": val_data_for_eval,
            "train_size": len(train_fold),
            "val_size": len(val_fold),
            "metrics": val_metrics,
            "model_state": copy.deepcopy(trained_model.state_dict()),
            "logger_history": copy.deepcopy(logger.history),
        }
        self.fold_results.append(fold_record)

        # quick print
        if val_metrics and "SUMMARY" in val_metrics:
            mi = val_metrics["SUMMARY"]["micro_avg"]
            ma = val_metrics["SUMMARY"]["macro_avg"]
            print(f"  micro F1: {mi['f1']:.4f}  macro F1: {ma['f1']:.4f}")

    def _finalize_after_all_folds(self):
        """after all folds, compute aggregates and pick best."""
        self.aggregate_metrics = self._compute_aggregate_metrics()
        self.best_fold_idx = self._get_best_fold_idx()
        print(f"\nBest fold: {self.best_fold_idx + 1}")

    def _compute_aggregate_metrics(self):
        metric_names = [
            "micro_f1",
            "micro_precision",
            "micro_recall",
            "macro_f1",
            "weighted_f1",
        ]
        aggregated = {}

        for metric_name in metric_names:
            values = []
            for fr in self.fold_results:
                m = fr["metrics"]
                if m and "SUMMARY" in m:
                    if metric_name.startswith("micro_"):
                        key = metric_name.replace("micro_", "")
                        values.append(m["SUMMARY"]["micro_avg"][key])
                    elif metric_name.startswith("macro_"):
                        key = metric_name.replace("macro_", "")
                        values.append(m["SUMMARY"]["macro_avg"][key])
                    elif metric_name.startswith("weighted_"):
                        key = metric_name.replace("weighted_", "")
                        values.append(m["SUMMARY"]["weighted_avg"][key])
            if values:
                values = np.array(values)
                aggregated[metric_name] = {
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                    "values": values.tolist(),
                }
        return aggregated

    def _get_best_fold_idx(self):
        best_idx = 0
        best_val = -1.0
        for i, fr in enumerate(self.fold_results):
            m = fr["metrics"]
            if m and "SUMMARY" in m:
                micro_f1 = m["SUMMARY"]["micro_avg"]["f1"]
                if micro_f1 > best_val:
                    best_val = micro_f1
                    best_idx = i
        return best_idx

    def load_best_model(self):
        idx = self.best_fold_idx
        best_fold = self.fold_results[idx]
        model = self.model_creation_func()
        model.load_state_dict(best_fold["model_state"])
        model.to(self.device)
        model.eval()
        return model
    
    def load_fold_model(self, fold_idx):
        """Load model for a specific fold (0-indexed)"""
        if fold_idx < 0 or fold_idx >= len(self.fold_results):
            raise ValueError(f"Invalid fold_idx {fold_idx}. Must be 0-{len(self.fold_results)-1}")
        
        fold = self.fold_results[fold_idx]
        model = self.model_creation_func()
        model.load_state_dict(fold["model_state"])
        model.to(self.device)
        model.eval()
        return model

    def get_best_fold_val_data(self):
        idx = self.best_fold_idx
        best_fold = self.fold_results[idx]
        return best_fold["val_data"], best_fold["val_data_for_eval"]
    
    # NEW: Get validation data for any specific fold
    def get_fold_val_data(self, fold_idx):
        """Get validation data for a specific fold (0-indexed)"""
        if fold_idx < 0 or fold_idx >= len(self.fold_results):
            raise ValueError(f"Invalid fold_idx {fold_idx}. Must be 0-{len(self.fold_results)-1}")
        
        fold = self.fold_results[fold_idx]
        return fold["val_data"], fold["val_data_for_eval"]
        
    def get_fold_train_data(self, fold_idx):
        """Get training data for a specific fold (0-indexed)"""
        if fold_idx < 0 or fold_idx >= len(self.fold_results):
            raise ValueError(f"Invalid fold_idx {fold_idx}. Must be 0-{len(self.fold_results)-1}")
        
        fold = self.fold_results[fold_idx]
        return fold["train_data"]

    def run(self):
        print(f"K-Fold Cross-Validation: {self.n_splits} folds, {len(self.data)} sequences")

        for fold_idx, train_fold, val_fold in self._iter_folds():
            trained_model, logger = self._train_single_fold(
                fold_idx,
                train_fold,
                val_fold
            )

            val_data_for_eval = self._prepare_val_for_eval(val_fold)
            val_metrics = self._evaluate_single_fold(
                trained_model,
                val_data_for_eval,
                fold_idx
            )

            self._record_fold_result(
                fold_idx,
                train_fold,
                val_fold,
                val_data_for_eval,
                val_metrics,
                trained_model,
                logger,
            )

            del trained_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._finalize_after_all_folds()
        return self

    def print_evaluation_summary(self):
        if not self.aggregate_metrics or not self.fold_results:
            print("No CV results available. Run `.run()` first.")
            return

        print("Cross-Validation Evaluation Results")

        agg = self.aggregate_metrics

        print("Aggregate Metrics Across All Folds:")
        if "micro_f1" in agg:
            print(f"Micro F1:        {agg['micro_f1']['mean']:.4f} ± {agg['micro_f1']['std']:.4f}")
        if "micro_precision" in agg:
            print(f"Micro Precision: {agg['micro_precision']['mean']:.4f} ± {agg['micro_precision']['std']:.4f}")
        if "micro_recall" in agg:
            print(f"Micro Recall:    {agg['micro_recall']['mean']:.4f} ± {agg['micro_recall']['std']:.4f}")
        if "macro_f1" in agg:
            print(f"Macro F1:        {agg['macro_f1']['mean']:.4f} ± {agg['macro_f1']['std']:.4f}")

        print(f"Best Fold ({self.best_fold_idx + 1}) Metrics:")
        best_fold_metrics = self.fold_results[self.best_fold_idx]["metrics"]
        if best_fold_metrics and "SUMMARY" in best_fold_metrics:
            best_summary = best_fold_metrics["SUMMARY"]
            print(f"Micro F1:        {best_summary['micro_avg']['f1']:.4f}")
            print(f"Micro Precision: {best_summary['micro_avg']['precision']:.4f}")
            print(f"Micro Recall:    {best_summary['micro_avg']['recall']:.4f}")
    
    def aggregate_case_level_results(
        self,
        model_creation_func,
        evaluate_case_level_func,
        label_processor,
        device):
        if not self.fold_results:
            print("No CV folds found. Run `.run()` first.")
            return

        print("Aggregating case-level results across all folds...")
        all_fold_case_results = []

        for fold_result in self.fold_results:
            fold_val_data = fold_result["val_data"]

            # recreate and load model for this fold
            fold_model = model_creation_func()
            fold_model.load_state_dict(fold_result["model_state"])
            fold_model.to(device)
            fold_model.eval()

            # evaluate case-level results for this fold
            fold_case_results = evaluate_case_level_func(
                fold_model,
                fold_val_data,
                label_processor
            )
            all_fold_case_results.append(fold_case_results)

            # cleanup
            del fold_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # gather F1s across all folds
        all_f1_scores = []
        for fold_case_result in all_fold_case_results:
            all_f1_scores.extend([case["f1"] for case in fold_case_result])

        if not all_f1_scores:
            print("No case-level F1 scores available.")
            return

        print(f"Aggregate Case-Level F1 Across All Folds:")
        print(f"Mean: {np.mean(all_f1_scores):.4f}")
        print(f"Std:  {np.std(all_f1_scores):.4f}")
        print(f"Min:  {np.min(all_f1_scores):.4f}")
        print(f"Max:  {np.max(all_f1_scores):.4f}")

        return {
            "mean": float(np.mean(all_f1_scores)),
            "std": float(np.std(all_f1_scores)),
            "min": float(np.min(all_f1_scores)),
            "max": float(np.max(all_f1_scores)),
            "all_f1_scores": all_f1_scores,
        }

    def aggregate_case_level_metrics(self):
        return self.aggregate_case_level_results(
            model_creation_func=self.model_creation_func,
            evaluate_case_level_func=evaluate_case_level,
            label_processor=self.label_processor,
            device=self.device
        )

    def get_fold_filename(self, base_name, fold_idx, extension=".xlsx"):
        return f"{base_name}_fold{fold_idx + 1}{extension}"
    
