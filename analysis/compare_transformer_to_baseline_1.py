import numpy as np
import random
import torch

#import completion labels
from config import COMPLETION_LABELS
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, multilabel_confusion_matrix

def compare_transformer_vs_baseline1_on_completion_labels(
    transformer_predictions,
    baseline1_predictions,
    val_data,
    label_processor):

    true_labels, transformer_preds = transformer_predictions
    baseline1_preds = baseline1_predictions

    if isinstance(true_labels, torch.Tensor):
        true_arr = true_labels.cpu().numpy()
    else:
        true_arr = np.array(true_labels)
        
    if isinstance(transformer_preds, torch.Tensor):
        trans_arr = transformer_preds.cpu().numpy()
    else:
        trans_arr = np.array(transformer_preds)
    
    if isinstance(baseline1_preds, torch.Tensor):
        baseline_arr = baseline1_preds.cpu().numpy()
    else:
        baseline_arr = np.array(baseline1_preds)
    
    print(f"✅ Using existing predictions:")
    print(f"   Transformer: {len(trans_arr):,} steps")
    print(f"   Ground truth: {len(true_arr):,} steps")
    print(f"   Baseline 1: {len(baseline_arr):,} cases")

    print("\nIdentifying final steps...")
    
    final_step_indices = []
    current_step = 0
    
    for case_idx, (seq, labels_raw) in enumerate(val_data):
        if isinstance(labels_raw[0], list):
            labels_binary = label_processor.transform_sequence(labels_raw)
        else:
            labels_binary = labels_raw
        
        num_steps = len(labels_binary)
        final_step_idx = current_step + num_steps - 1
        final_step_indices.append(final_step_idx)
        current_step += num_steps
    
    final_step_indices = np.array(final_step_indices)
    
    print(f"   Total cases: {len(final_step_indices):,}")
    print(f"   Average steps per case: {len(trans_arr)/len(final_step_indices):.1f}")

    trans_final = trans_arr[final_step_indices]
    true_final = true_arr[final_step_indices]
    baseline_final = baseline_arr  # Already one per case
    
    # Sanity check
    if len(trans_final) != len(baseline_final):
        raise ValueError(f"Mismatch: {len(trans_final)} transformer final steps vs {len(baseline_final)} baseline predictions")
    
    #we need to filter to only completion labels
    full_label_names = label_processor.label_names
    label_to_idx = {label: idx for idx, label in enumerate(full_label_names)}
    completion_labels = [label for label in COMPLETION_LABELS if label in label_to_idx]
    completion_label_indices = [label_to_idx[label] for label in completion_labels]

    if not completion_label_indices:
        raise ValueError("No completion labels found in label_processor.label_names")
    
    print(f"\n✅ Filtering to {len(completion_label_indices)} completion labels:")
    for idx in completion_label_indices:
        print(f"   - {label_processor.label_names[idx]}")

    trans_final_completion = trans_final[:, completion_label_indices]
    true_final_completion = true_final[:, completion_label_indices]
    baseline_final_completion = baseline_final[:, completion_label_indices]

    true_k_completion = true_final_completion.sum(axis=1)
    has_completion = true_k_completion > 0
    
    num_completion = has_completion.sum()
    
    print(f"\nFiltering to cases with completion labels:")
    print(f"   Total cases: {len(true_final_completion):,}")
    print(f"   Cases with completion labels: {num_completion:,} ({num_completion/len(true_final_completion)*100:.1f}%)")
    
    if num_completion == 0:
        print("\n⚠️  No cases with completion labels found!")
        return None
    
    # Filter to cases with completion
    trans_comp = trans_final_completion[has_completion]
    true_comp = true_final_completion[has_completion]
    baseline_comp = baseline_final_completion[has_completion]

    mcm_trans = multilabel_confusion_matrix(true_comp, trans_comp)
    mcm_base = multilabel_confusion_matrix(true_comp, baseline_comp)
    
    print(f"\n📊 Data shape: {true_comp.shape[0]} cases × {true_comp.shape[1]} completion labels")
    

    # Calculate for all of them together
    print("\nCalculating overall metrics for completion labels...")
    trans_micro_precision, trans_micro_recall, trans_micro_f1, trans_micro_support = precision_recall_fscore_support(
        true_comp, trans_comp, average='micro', zero_division=0
    )
    baseline_micro_precision, baseline_micro_recall, baseline_micro_f1, baseline_micro_support = precision_recall_fscore_support(
        true_comp, baseline_comp, average='micro', zero_division=0
    )
    trans_macro_precision, trans_macro_recall, trans_macro_f1, trans_macro_support = precision_recall_fscore_support(
        true_comp, trans_comp, average='macro', zero_division=0
    )
    baseline_macro_precision, baseline_macro_recall, baseline_macro_f1, baseline_macro_support = precision_recall_fscore_support(
        true_comp, baseline_comp, average='macro', zero_division=0
    )
    trans_weighted_precision, trans_weighted_recall, trans_weighted_f1, trans_weighted_support = precision_recall_fscore_support(
        true_comp, trans_comp, average='weighted', zero_division=0
    )
    baseline_weighted_precision, baseline_weighted_recall, baseline_weighted_f1, baseline_weighted_support = precision_recall_fscore_support(
        true_comp, baseline_comp, average='weighted', zero_division=0
    )

    trans_accuracy = accuracy_score(true_comp, trans_comp)
    baseline_accuracy = accuracy_score(true_comp, baseline_comp)

    trans_per_label_precision, trans_per_label_recall, trans_per_label_f1, trans_per_label_support = precision_recall_fscore_support(
        true_comp,
        trans_comp,
        average=None,  # Per-label
        zero_division=0
    )
    baseline_per_label_precision, baseline_per_label_recall, baseline_per_label_f1, baseline_per_label_support = precision_recall_fscore_support(
        true_comp,
        baseline_comp,
        average=None,  # Per-label 
        zero_division=0
    )

    per_label_metrics = {}
    for i, label_idx in enumerate(completion_label_indices):
        label_name = label_processor.label_names[label_idx]
        
        #ravel them out to get tp, fp, fn, tn
        #we need to do it for mcm_trans and mcm_base
        #first mcm_trans
        tn_trans, fp_trans, fn_trans, tp_trans = mcm_trans[i].ravel()
        #then mcm_base
        tn_base, fp_base, fn_base, tp_base = mcm_base[i].ravel()
        

        per_label_metrics[label_name] = {
            'transformer': {
                'f1': float(trans_per_label_f1[i]),
                'precision': float(trans_per_label_precision[i]),
                'recall': float(trans_per_label_recall[i]),
                'support': int(trans_per_label_support[i]),
                'tp': int(tp_trans), 'fp': int(fp_trans), 'fn': int(fn_trans),
                'tn': int(tn_trans)
            },  
            'baseline_1': {
                'f1': float(baseline_per_label_f1[i]),
                'precision': float(baseline_per_label_precision[i]),
                'recall': float(baseline_per_label_recall[i]),
                'support': int(baseline_per_label_support[i]),
                'tp': int(tp_base), 'fp': int(fp_base), 'fn': int(fn_base),
                'tn': int(tn_base)
            },
            'improvement': {
                'f1_diff': float(trans_per_label_f1[i] - baseline_per_label_f1[i]),
                'precision_diff': float(trans_per_label_precision[i] - baseline_per_label_precision[i]),
                'recall_diff': float(trans_per_label_recall[i] - baseline_per_label_recall[i]),
                'support_diff': int(trans_per_label_support[i] - baseline_per_label_support[i])
            }
        }
    results = {
        'comparison_type': 'final_step_completion_labels',
        'baseline': 'baseline1',
        'num_completion_labels': len(completion_label_indices),
        'completion_labels': completion_labels,
        'total_cases': int(len(true_final_completion)),
        'cases_with_completion': int(num_completion),
        'micro_avg': {
            'transformer': {
                'f1': float(trans_micro_f1),
                'precision': float(trans_micro_precision),
                'recall': float(trans_micro_recall)
            },
            'baseline1': {
                'f1': float(baseline_micro_f1),
                'precision': float(baseline_micro_precision),
                'recall': float(baseline_micro_recall)
            },
            'improvement': {
                'f1': float(trans_micro_f1 - baseline_micro_f1),
                'f1_percent': (trans_micro_f1 / baseline_micro_f1 - 1) * 100 if baseline_micro_f1 > 0 else None,
                'precision': float(trans_micro_precision - baseline_micro_precision),
                'recall': float(trans_micro_recall - baseline_micro_recall)
            }
        },
        'macro_avg': {
            'transformer': {
                'f1': float(trans_macro_f1),
                'precision': float(trans_macro_precision),
                'recall': float(trans_macro_recall)
            },
            'baseline1': {
                'f1': float(baseline_macro_f1),
                'precision': float(baseline_macro_precision),
                'recall': float(baseline_macro_recall)
            },
            'improvement': {
                'f1': float(trans_macro_f1 - baseline_macro_f1),
                'f1_percent': (trans_macro_f1 / baseline_macro_f1 - 1) * 100 if baseline_macro_f1 > 0 else None,
                'precision': float(trans_macro_precision - baseline_macro_precision),
                'recall': float(trans_macro_recall - baseline_macro_recall)
            }
        },
        'weighted_avg': {
            'transformer': {
                'f1': float(trans_weighted_f1),
                'precision': float(trans_weighted_precision),
                'recall': float(trans_weighted_recall)
            },
            'baseline1': {
                'f1': float(baseline_weighted_f1),
                'precision': float(baseline_weighted_precision),
                'recall': float(baseline_weighted_recall)
            },
            'improvement': {
                'f1': float(trans_weighted_f1 - baseline_weighted_f1),
                'f1_percent': (trans_weighted_f1 / baseline_weighted_f1 - 1) * 100 if baseline_weighted_f1 > 0 else None,
                'precision': float(trans_weighted_precision - baseline_weighted_precision),
                'recall': float(trans_weighted_recall - baseline_weighted_recall)
            }
        },
        'accuracy': {
            'transformer': float(trans_accuracy),
            'baseline1': float(baseline_accuracy),
        },
        'per_label': per_label_metrics
    }
    return results
    

def convert_baseline1_comparison_to_table_generator_format(baseline1_comparison_results):
    res = baseline1_comparison_results
    
    transformer_step_metrics = {
        'SUMMARY': {
            'micro_avg': {
                'f1': res['micro_avg']['transformer']['f1'],
                'precision': res['micro_avg']['transformer']['precision'],
                'recall': res['micro_avg']['transformer']['recall']
            },
            'macro_avg': {
                'f1': res['macro_avg']['transformer']['f1'],
                'precision': res['macro_avg']['transformer']['precision'],
                'recall': res['macro_avg']['transformer']['recall']
            },
            'weighted_avg': {
                'f1': res['weighted_avg']['transformer']['f1'],
                'precision': res['weighted_avg']['transformer']['precision'],
                'recall': res['weighted_avg']['transformer']['recall']
            },
            'total_samples': res['cases_with_completion']
        },
        'INDIVIDUAL_LABELS': {}
    }

    for label, metrics in res['per_label'].items():
        #get false positive, false negative, true positive, true negative counts
        #insert here

        transformer_step_metrics['INDIVIDUAL_LABELS'][label] = {
            'f1': metrics['transformer']['f1'],
            'precision': metrics['transformer']['precision'],
            'recall': metrics['transformer']['recall'],
            'support': metrics['transformer']['support'],
            'tp': metrics['transformer']['tp'],
            'fp': metrics['transformer']['fp'],
            'fn': metrics['transformer']['fn'],
        }
    
    baseline1_step_metrics = {
        'SUMMARY': {
            'micro_avg': {
                'f1': res['micro_avg']['baseline1']['f1'],
                'precision': res['micro_avg']['baseline1']['precision'],
                'recall': res['micro_avg']['baseline1']['recall']
            },
            'macro_avg': {
                'f1': res['macro_avg']['baseline1']['f1'],
                'precision': res['macro_avg']['baseline1']['precision'],
                'recall': res['macro_avg']['baseline1']['recall']
            },
            'weighted_avg': {
                'f1': res['weighted_avg']['baseline1']['f1'],
                'precision': res['weighted_avg']['baseline1']['precision'],
                'recall': res['weighted_avg']['baseline1']['recall']
            },
            'total_samples': res['cases_with_completion']
        },
        'INDIVIDUAL_LABELS': {}
    }

    for label, metrics in res['per_label'].items():
        #get false positive, false negative, true positive, true negative counts
        #insert here

        baseline1_step_metrics['INDIVIDUAL_LABELS'][label] = {
            'f1': metrics['baseline_1']['f1'],
            'precision': metrics['baseline_1']['precision'],
            'recall': metrics['baseline_1']['recall'],
            'support': metrics['baseline_1']['support'],
            'tp': metrics['baseline_1']['tp'],
            'fp': metrics['baseline_1']['fp'],
            'fn': metrics['baseline_1']['fn'],
        }
    return transformer_step_metrics, baseline1_step_metrics
