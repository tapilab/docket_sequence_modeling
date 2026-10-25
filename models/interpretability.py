import warnings
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients, DeepLift, Saliency, InputXGradient
from config import get_model_config
from training.teacher_forcing import build_teacher_forced_sequences
import io
from models.interpretability_helpers import CounterfactualMagnitudeStudy, plot_counterfactual_results


warnings.filterwarnings("ignore")

def zscore_by_feature(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalize each feature (column) across samples (rows)."""
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    return (arr - mean) / (std + eps)

def zscore_by_sample(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalize each sample (row) across features (columns)."""
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)
    return (arr - mean) / (std + eps)

def zscore_both_ways(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Apply z-score normalization first by feature, then by sample."""
    # First normalize each feature across samples
    arr_norm = zscore_by_feature(arr, eps)
    # Then normalize each sample across features
    arr_norm = zscore_by_sample(arr_norm, eps)
    return arr_norm


def get_feature_dims(model) -> Dict[str, int]:
    """
    Return explicit pre-fusion dims for each block in the exact concat order
    used by model.encode_proceeding_unfused(). Expose these on your model once,
    e.g. model.feature_dims = {...}, to avoid guessing.
    """
    if hasattr(model, "feature_dims"):
        return model.feature_dims  # authoritative if present

    # Fallback: expect these attributes on the model. Define them in your model once.
    required = [
        "defendant_dim", "case_dim", "text_dim", "magistrate_dim",
        "prior_label_dim", "prior_charge_dim",
        "court_type_dim", "bond_dim", "bond_sum_dim", "prior_case_count_dim",
        "temporal_dim"
    ]
    missing = [r for r in required if not hasattr(model, r)]
    if missing:
        raise AttributeError(
            f"Model must define feature dims or .feature_dims. Missing: {missing}"
        )

    # The order of this dictionary is critical and must match the concatenation order
    return {
        "Defendant_Embedding": model.defendant_dim,
        "Case_Embedding": model.case_dim,
        "BERT_Text_Embedding": model.text_dim,
        "Magistrate_Embedding": model.magistrate_dim,
        "Prior_Label_Embedding": model.prior_label_dim,
        "Prior_Charge_Embedding": model.prior_charge_dim,
        "Court_Type": model.court_type_dim,
        "Original_Bond": model.bond_dim,
        "Sum_Of_Bonds": model.bond_sum_dim,
        "Prior_Case_Count": model.prior_case_count_dim,
        "Temporal_Features": model.temporal_dim,
    }

def feature_indices_from_dims(dims: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    idxs = {}
    cursor = 0
    for name, dim in dims.items():
        idxs[name] = (cursor, cursor + dim)
        cursor += dim
    return idxs


class CaptumWrapper(nn.Module):
    def __init__(self, original_model: nn.Module, seq_len: int, use_unfused: bool = False):
        super().__init__()
        self.model = original_model
        # Max sequence length for positional encoding
        self.max_len = seq_len
        self.use_unfused = use_unfused
    def forward(self, input_features: torch.Tensor, target_idx: Optional[int] = None) -> torch.Tensor:
        self.model.eval()
        current_device = input_features.device
        
        if self.use_unfused:
            # Input is unfused features (batch, seq_len, 1904)
            # First apply fusion to get (batch, seq_len, 768)
            batch_size, seq_len, feature_dim = input_features.shape
            
            # Reshape to (batch*seq_len, feature_dim) for fusion
            x_reshaped = input_features.reshape(-1, feature_dim)
            x_fused = self.model.fuse(x_reshaped)
            
            # Reshape back to (batch, seq_len, 768)
            x = x_fused.reshape(batch_size, seq_len, -1).clone()
        else:
            # Input is already fused
            x = input_features.clone()
        
        # NOW add positional encoding (both are 768 dim)
        x = x + self.model.pos[:, :x.size(1), :].to(current_device)
        
        # Use causal mask to match main model behavior
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=current_device)
        out = self.model.encoder(x, is_causal=True, mask=causal_mask)
        
        if target_idx is None:
            raise ValueError("target_idx must be provided")
                
        if target_idx < 0 or target_idx >= out.size(1):
            raise IndexError(f"target_idx {target_idx} out of range for sequence length {out.size(1)}")
            
        if self.model.classifier:
            return self.model.classifier(out[:, target_idx, :])
        return out[:, target_idx, :]
    
class CaseTransformerInterpretability:
    def __init__(self, model, label_processor, verbose: bool, device: torch.device):  # FIXED: Added type hint for verbose
        self.model = model
        self.device = device
        self.model.eval()
        self.model_config = get_model_config()
        self.label_processor = label_processor
        self.outcome_labels = self.label_processor.label_names
        self.verbose = verbose
        if torch.cuda.is_available():
            self.original_cudnn_state = torch.backends.cudnn.enabled
            torch.backends.cudnn.enabled = False  # Disable cuDNN for reproducibility
        
        self._attributions_cache = {}
        if self.verbose:
            print(f"CombinedCaptumWrapper initialized")
            print(f"   Device: {device}")
            print(f"   Model type: {type(model).__name__}")
            print(f"   Labels: {len(label_processor.label_names)}")
            print(f"   CuDNN disabled for gradient computation")
    
    def __del__(self):
        """Restore CuDNN state on cleanup."""
        if torch.cuda.is_available() and hasattr(self, 'original_cudnn_state'):
            torch.backends.cudnn.enabled = self.original_cudnn_state
    
    def _get_baseline_features_tensor(self, seq: List[Dict]) -> torch.Tensor:        
        start_idx = max(0, len(seq) - self.model.max_len)

        with torch.no_grad():
            reps = []
            for i, p in enumerate(seq[-self.model.max_len:]):
                rep = self.model.encode_proceeding_unfused(p, start_idx + i)
                reps.append(rep)
        
        if not reps:  # FIXED: Changed logic
            return torch.empty(0, device=self.device)
        
        input_tensor = torch.stack(reps, dim=1).to(self.device)
        input_tensor = input_tensor.clone().detach()
        input_tensor.requires_grad = True
        return input_tensor
    
    def _get_feature_component_indices(self) -> Dict[str, Tuple[int, int]]:
        hidden_dim_totals = self.model.fuse[0].in_features
        feature_order = [
            ('Defendant_Embedding', self.model.feature_dims['Defendant_Embedding']),  # de
            ('Case_Embedding', self.model.feature_dims['Case_Embedding']),           # ce
            ('BERT_Text_Embedding', self.model.feature_dims['BERT_Text_Embedding']),  # te
            ('Magistrate_Embedding', self.model.feature_dims['Magistrate_Embedding']),  # me
            ('Prior_Label_Embedding', self.model.feature_dims['Prior_Label_Embedding']),  # prior_label_emb
            ('Prior_Charge_Embedding', self.model.feature_dims['Prior_Charge_Embedding']),  # prior_charge_emb
            ('Court_Type', self.model.feature_dims['Court_Type']), # feature_block[0]
            ('Original_Bond', self.model.feature_dims['Original_Bond']),  # feature_block[1]
            ('Sum_Of_Bonds', self.model.feature_dims['Sum_Of_Bonds']),  # feature_block[2]
            ('Prior_Case_Count', self.model.feature_dims['Prior_Case_Count']),  # feature_block[3]
            ('Temporal_Features', self.model.feature_dims['Temporal_Features'])  # temporal_features
        ]
        indices = {}
        current_index = 0
        for name, dim in feature_order:
            end_idx = current_index + dim
            indices[name] = (current_index, end_idx)
            current_index = end_idx
        if current_index != hidden_dim_totals:
            warnings.warn(f"Feature dimension mismatch: expected {hidden_dim_totals}, got {current_index}")
        return indices

    def attribute_sequence_features(
        self,
        seq: List[Dict],
        target_class_idx: int,
        target_proceeding_idx: int,
        method: str = 'IntegratedGradients',
        n_steps: int = 50,
        return_component_attributions: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, np.ndarray]]]:  # FIXED: Return type
        seq_trimmed = seq[-self.model.max_len:]
        if not seq_trimmed:  # FIXED: Typo from not_seq_trimmed
            raise ValueError("The input sequence is empty after trimming to max_len.")
        
        if self.verbose:
            print(f"Attributing sequence of length {len(seq_trimmed)} for class index {target_class_idx} at proceeding index {target_proceeding_idx}")
        
        input_features = self._get_baseline_features_tensor(seq_trimmed)
        references_features = torch.zeros_like(input_features, device=self.device)
        model_wrapper = CaptumWrapper(self.model, seq_len=input_features.size(1), use_unfused=True).to(self.device)
        
        if method == 'IntegratedGradients':
            attributor = IntegratedGradients(model_wrapper)
            attributor_kwargs = {'baselines': references_features, 'n_steps': n_steps}
        elif method == 'DeepLift':
            attributor = DeepLift(model_wrapper)
            attributor_kwargs = {'baselines': references_features}
        elif method == 'InputXGradient':
            attributor = InputXGradient(model_wrapper)
            attributor_kwargs = {}
        elif method == 'Saliency':
            attributor = Saliency(model_wrapper)
            attributor_kwargs = {}
        else:
            raise ValueError(f"Unsupported attribution method: {method}")
        
        additional_args = None
        if target_proceeding_idx is not None:
            additional_args = (target_proceeding_idx,)
        
        if target_class_idx < 0 or target_class_idx >= len(self.outcome_labels):
            raise IndexError(f"target_class_idx {target_class_idx} is out of bounds for {len(self.outcome_labels)} labels.")
        
        try:
            # FIXED: Handle return_convergence_delta properly for different methods
            if method in ['IntegratedGradients', 'DeepLift']:
                attributions, delta = attributor.attribute(
                    inputs=input_features,
                    target=target_class_idx,
                    additional_forward_args=additional_args,
                    return_convergence_delta=True,
                    **attributor_kwargs
                )
                if self.verbose:
                    print(f"   Attribution shape: {attributions.shape}")
                    print(f"   Convergence delta: {delta.item()}")
            else:
                # InputXGradient and Saliency don't support return_convergence_delta
                attributions = attributor.attribute(
                    inputs=input_features,
                    target=target_class_idx,
                    additional_forward_args=additional_args,
                    **attributor_kwargs
                )
                delta = None
                if self.verbose:
                    print(f"   Attribution shape: {attributions.shape}")
            
            attributions_tensor = attributions.squeeze(0)
            
            if return_component_attributions:
                component_indices = self._get_feature_component_indices()
                component_attributions = {}
                attributions_np = attributions_tensor.detach().cpu().numpy()
                for comp_name, (start_idx, end_idx) in component_indices.items():
                    comp_attr = attributions_np[:, start_idx:end_idx]
                    component_attributions[comp_name] = np.sum(np.abs(comp_attr), axis=1)  # FIXED: nb.abs -> np.abs
                
                return attributions_tensor, component_attributions
            
            return attributions_tensor
            
        except Exception as e:
            if self.verbose:
                print(f"Attribution failed: {e}")
            raise e
    def visualize_proceeding_importance(
        self,
        attributions_tensor: torch.Tensor,
        target_class_idx: int,
        case_id: str = "Case Sequence",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ):
        """
        Visualize the total absolute attribution score for each proceeding step.
        
        Args:
            attributions_tensor: Output from attribute_sequence_features (seq_len, hidden_dim)
            target_class_idx: Index of the predicted label being explained
            case_id: Identifier for the plot title
            save_path: Optional path to save figure
            show: Whether to display the plot
        """
        if attributions_tensor.dim() != 2:
            print("Error: Attributions tensor must have shape (seq_len, hidden_dim).")
            return
        
        # Aggregate attribution magnitude across all features
        total_importance = attributions_tensor.abs().sum(dim=1).detach().cpu().numpy()
        
        seq_len = total_importance.shape[0]
        proceeding_indices = np.arange(1, seq_len + 1)
        target_label = self.outcome_labels[target_class_idx]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(proceeding_indices, total_importance, color='skyblue', alpha=0.7)
        
        # Highlight last proceeding (most recent)
        bars[-1].set_color('salmon')
        
        plt.title(
            f"Proceeding Importance for '{target_label}' Prediction\n(Case: {case_id})", 
            fontsize=14, 
            fontweight='bold'
        )
        plt.xlabel(f"Proceeding Step (Total Steps: {seq_len})", fontsize=12)
        plt.ylabel("Total Absolute Attribution Score", fontsize=12)
        plt.xticks(proceeding_indices)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            if self.verbose:
                print(f"Saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_component_importance(
        self,
        component_attributions: Dict[str, np.ndarray],
        target_class_idx: int,
        case_id: str = "Case Sequence",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        normalize_heatmap: bool = True
    ):
        """
        Visualize importance of different feature components over time.
        
        Args:
            component_attributions: Dictionary from attribute_sequence_features
            target_class_idx: Index of predicted label
            case_id: Identifier for plot title
            save_path: Optional path to save figure
            show: Whether to display plot
        """
        target_label = self.outcome_labels[target_class_idx]
        
        # Create stacked area plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top panel: Stacked area chart
        seq_len = len(next(iter(component_attributions.values())))
        proceeding_indices = np.arange(1, seq_len + 1)
        
        # Prepare data for stacking
        components_list = []
        labels_list = []
        
        for name, values in sorted(component_attributions.items()):
            components_list.append(values)
            labels_list.append(name.replace('_', ' '))
        
        ax1.stackplot(
            proceeding_indices, 
            *components_list, 
            labels=labels_list,
            alpha=0.7
        )
        
        ax1.set_title(
            f"Feature Component Importance Over Time\n'{target_label}' - {case_id}",
            fontsize=14,
            fontweight='bold'
        )
        ax1.set_xlabel("Proceeding Step", fontsize=12)
        ax1.set_ylabel("Cumulative Attribution Score", fontsize=12)
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        
        # Bottom panel: Heatmap
        components_array = np.array(components_list)
        if normalize_heatmap:
            components_array = zscore_by_sample(components_array)
            heatmap_title = "Component Importance Heatmap (Z-score Normalized)"
            cbar_label = "z-score Attribution score"
        else:
            heatmap_title = "Component Importance Heatmap"
            cbar_label = "Attribution Score"

        im = ax2.imshow(
            components_array, 
            aspect='auto', 
            cmap='YlOrRd',
            interpolation='nearest'
        )
        
        ax2.set_yticks(range(len(labels_list)))
        ax2.set_yticklabels(labels_list, fontsize=9)
        ax2.set_xticks(range(seq_len))
        ax2.set_xticklabels(proceeding_indices)
        ax2.set_xlabel("Proceeding Step", fontsize=12)
        ax2.set_title(heatmap_title, fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label(cbar_label, fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            if self.verbose:
                print(f"Saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_feature_summary(
        self,
        seq: List[Dict],
        target_class_idx: int,
        target_proceeding_idx: int
    ) -> pd.DataFrame:
        """
        Get a summary dataframe of feature importance.
        
        Args:
            seq: Proceeding sequence
            target_class_idx: Target label index
            target_proceeding_idx: Target proceeding index
            
        Returns:
            DataFrame with feature importance summary
        """
        # Get component attributions
        _, comp_attrs = self.attribute_sequence_features(
            seq,
            int(target_class_idx),
            target_proceeding_idx,
            return_component_attributions=True
        )
        
        # Create summary
        summary_data = []
        for comp_name, comp_values in comp_attrs.items():
            summary_data.append({
                'Feature_Component': comp_name,
                'Total_Attribution': np.sum(np.abs(comp_values)),
                'Mean_Attribution': np.mean(np.abs(comp_values)),
                'Max_Attribution': np.max(np.abs(comp_values)),
                'Std_Attribution': np.std(comp_values)
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Total_Attribution', ascending=False)
        df['Target_Label'] = self.outcome_labels[target_class_idx]
        
        return df
    
    def attributions_table(self, dataloader, max_samples, show_top_n_features, ig_n_steps, verbose):
        class_attributions = defaultdict(lambda: defaultdict(float))
        class_counts = defaultdict(int)
        component_names = list(self._get_feature_component_indices().keys())
        print(f"Computing attributions for up to {max_samples} samples...")
        samples_processed = 0

        for batch in tqdm(dataloader, desc="Attribution Batches"):
            if samples_processed >= max_samples:
                break
            seq, labels_seq = batch

            if isinstance(seq, list) and len(seq) > 0 and isinstance(seq[0], dict):
                pass
            elif isinstance(seq, list) and len(seq) == 1:
                seq = seq[0]
                labels_seq = labels_seq[0]
            else:
                if verbose:
                    print(f"Skipping batch with unsupported structure (size {len(seq)}). Requires batch_size=1 or iterating a list.")
                continue
            
            if isinstance(labels_seq[0], list):
                labels_seq_binary = self.label_processor.transform_sequence(labels_seq)
            else:
                labels_seq_binary = labels_seq
                
            labels_binaries_list = []
            for lbl in labels_seq_binary:
                if isinstance(lbl, torch.Tensor):
                    labels_binaries_list.append(lbl.cpu().numpy())
                else:
                    labels_binaries_list.append(np.array(lbl))

            teacher_seq, _ = build_teacher_forced_sequences(
                seq, 
                labels_binaries_list, 
                self.label_processor, 
                include_initial_state=True
            )

            target_proceeding_idx = len(teacher_seq) - 1
            if target_proceeding_idx < 0:
                continue

            true_labels_data = labels_binaries_list[target_proceeding_idx]
            if isinstance(true_labels_data, torch.Tensor):
                true_labels_binary = true_labels_data.cpu().numpy()
            else:
                true_labels_binary = np.array(true_labels_data)

            true_class_indices = np.where(true_labels_binary == 1)[0]
            if true_class_indices.size == 0:
                continue

            for target_class_idx in true_class_indices:
                if target_class_idx >= len(self.label_processor.label_names):
                    if verbose:
                        print(f"Skipping out-of-bounds class_idx {target_class_idx}")
                    continue

                class_name = self.label_processor.label_names[target_class_idx]
                try:
                    # Use self.attribute_sequence_features
                    _, comp_attrs = self.attribute_sequence_features(
                        seq=teacher_seq,
                        target_class_idx=int(target_class_idx),
                        target_proceeding_idx=target_proceeding_idx,
                        return_component_attributions=True,
                        method='IntegratedGradients',
                        n_steps=ig_n_steps
                    )

                    for comp_name in component_names:
                        if comp_name in comp_attrs:
                            total_abs_attr = np.sum(np.abs(comp_attrs[comp_name]))
                            class_attributions[class_name][comp_name] += total_abs_attr

                    class_counts[class_name] += 1

                except Exception as e:
                    if verbose:
                        print(f"Warning: Attribution failed for sample {samples_processed}, class '{class_name}': {e}")
                    continue

            samples_processed += 1

        if class_attributions:
            print(f"Attributions computed for {samples_processed} samples.")
        else:
            print("No attributions were computed.")
            return pd.DataFrame()

        # Create DataFrame with raw attributions
        df = pd.DataFrame(class_attributions).fillna(0)
        avg_df = df.copy()
        seen_classes = []
        
        # Average by count for each class
        for class_name in avg_df.columns:
            count = class_counts.get(class_name, 0)
            if count > 0:
                avg_df[class_name] = avg_df[class_name] / count
                seen_classes.append(class_name)
            else:
                avg_df[class_name] = 0

        if not seen_classes:
            print("No valid classes with attributions were found.")
            return pd.DataFrame()

        final_df = avg_df[seen_classes].copy()

        # View 1: Raw average attributions
        final_df['Mean_Raw_Attribution'] = final_df[seen_classes].mean(axis=1)
        

        # View 2: By sample z-score (normalize within each label)
        # Answers: For each label independently, which features are most important? 
        
        # Z-SCORE NORMALIZE EACH LABEL SEPARATELY
        for class_name in seen_classes:
            values = final_df[class_name].values
            mean = values.mean()
            std = values.std()
            if std > 0:
                final_df[f'{class_name}_zscore_within_label'] = (values - mean) / std
            else:
                final_df[f'{class_name}_zscore_within_label'] = 0.0
        
        # Add mean of z-scores across classes (now this is meaningful!)
        within_label_zscore_cols = [col for col in final_df.columns if col.endswith('_zscore_within_label')]
        if within_label_zscore_cols:
            final_df['Mean_ZScore_Within_Label'] = final_df[within_label_zscore_cols].mean(axis=1)
        
        # View 3: By feature z-score (normalize within each feature across labels)
        # Answers: For each feature, which labels rely on it unusually heavily?
        for feature_name in final_df.index:
            values = final_df.loc[feature_name, seen_classes].values
            mean = values.mean()
            std = values.std()
            if std > 0:
                for class_name in seen_classes:
                    final_df.loc[feature_name, f'{class_name}_zscore_across_labels'] = \
                        (final_df.loc[feature_name, class_name] - mean) / std
            else:
                for class_name in seen_classes:
                    final_df.loc[feature_name, f'{class_name}_zscore_across_labels'] = 0.0
        across_label_zscore_cols = [col for col in final_df.columns if col.endswith('_zscore_across_labels')]
        if across_label_zscore_cols:
            final_df['Mean_ZScore_Across_Labels'] = final_df[across_label_zscore_cols].mean(axis=1)

        # output all
        final_df = final_df.sort_values('Mean_Raw_Attribution', ascending=False)
        return final_df

    def plot_feature_importance(self, attributions_df, view_type, save_path=None, title=None):
        if view_type == 'raw':
            cols = [col for col in attributions_df.columns
                    if not col.endswith(('_zscore_within_label', '_zscore_across_labels'))
                    and col not in ['Mean_Raw_Attribution', 'Mean_ZScore_Within_Label', 'Mean_ZScore_Across_Labels']]
            label_names = cols
            cmap = 'YlOrRd'
            cbar_label = 'Raw Attribution Score'
            center = None
            #annotation format
            annot_fmt = '.1f'
        elif view_type == 'within_label':
            cols = [col for col in attributions_df.columns if col.endswith('_zscore_within_label')]
            label_names = [col.replace('_zscore_within_label', '') for col in cols]
            cmap = 'RdBu_r'  # Diverging colormap
            cbar_label = 'Z-Score (within label)'
            center = 0  # Center at 0 for z-scores
            annot_fmt = '.2f'
        elif view_type == 'across_labels':
            cols = [col for col in attributions_df.columns if col.endswith('_zscore_across_labels')]
            label_names = [col.replace('_zscore_across_labels', '') for col in cols]
            cmap = 'RdBu_r'  # Diverging colormap
            cbar_label = 'Z-Score (across labels)'
            center = 0  # Center at 0 for z-scores
            annot_fmt = '.2f'
        else:
            raise ValueError(f"Unsupported view_type: {view_type}. Choose from 'raw', 'within_label', 'across_labels'.")
        
        if not cols:
            print(f"No columns found for view_type '{view_type}'. Please run attributions_table first.")
            return
        
        data_matrix = attributions_df[cols].T.values
        feature_names = attributions_df.index.tolist()

        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(12, len(feature_names) * 0.8), 
                                        max(8, len(label_names) * 0.4)))
        
        # Create heatmap with diverging colormap
        sns.heatmap(data_matrix, 
                    xticklabels=feature_names,
                    yticklabels=label_names,
                    cmap=cmap,
                    center=center,
                    annot=True,  # Show values
                    fmt=annot_fmt,
                    cbar_kws={'label': cbar_label},
                    ax=ax)
        
        ax.set_xlabel('Feature Component', fontsize=12, fontweight='bold')
        ax.set_ylabel('Label', fontsize=12, fontweight='bold')

        if title is None:
            if view_type == 'raw':
                title = 'Raw Attribution Magntiude\nAbsolute importance of each feature for each label'
            elif view_type == 'within_label':
                title = 'Feature Rankings Within Each Label\nRelative importance of features for each specific label'
            elif view_type == 'across_labels':
                title = 'Feature Specialization Across Labels\nWhich labels rely unusually heavily on each feature'


        ax.set_title(title,
                     fontsize=14, fontweight='bold', pad=20)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            buf = io.BytesIO()
            plt.savefig(buf, bbox_inches='tight', format='png')
            buf.seek(0)
            with open(save_path, 'wb') as f:
                f.write(buf.read())
            buf.close()
            if self.verbose:
                print(f"Saved {view_type} heatmap to {save_path}")
        
        plt.show()
        plt.close()

    def run_counterfactual_magnitude_study(self, val_sequences: list, val_labels: list, output_dir: Path, 
    max_samples: int = 200, top_n_labels: int = 8, show_plots: bool = True, parallel: bool = True, tqdm_position: int = 0, age_bins: list = None, prior_case_bins: list = None) -> Dict:
        output_dir = Path(output_dir)
        print(f"\n{'='*70}")
        print("COUNTERFACTUAL MAGNITUDE STUDY")
        print(f"  Samples : {min(max_samples, len(val_sequences))}")
        print(f"  Features: race, gender, age, original_bond, prior_case_count")
        print(f"{'='*70}\n")


        study = CounterfactualMagnitudeStudy(
            self.model,
            self.outcome_labels,
            self.device,
            self.label_processor,
            sequences=val_sequences,
            age_bins=age_bins,
            prior_case_bins=prior_case_bins,
        )
        
        aggregated, raw = study.run(
                sequences=val_sequences,
                label_sequences=val_labels,
                max_samples=max_samples,
                output_dir=output_dir,
                tqdm_position=tqdm_position,
            )

        plot_counterfactual_results(
                aggregated,
                raw=raw,
                output_dir=output_dir,
                show=show_plots,
                top_n_labels=top_n_labels,
        )

        return {"aggregated": aggregated, "raw": raw, "study": study}