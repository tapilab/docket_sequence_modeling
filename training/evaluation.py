import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix
from collections import defaultdict
from pathlib import Path
from utils.cache_directory_utils import CacheDirectoryManager
from training.teacher_forcing import build_teacher_forced_sequences
from training.top_k_helper import predict_with_top_k_mlb



def evaluate_with_teacher_forcing(model, data, device, label_processor, epoch: int = None, optional_tag: str = None, cache_manager = None):
    model.eval()
    device = device or next(model.parameters()).device

    all_true = []
    all_pred = []
    case_errors = []
    prediction_distribution = defaultdict(int)
    true_distribution = defaultdict(int)

    # Use label_processor.label_names instead of LABEL_NAMES
    L = len(label_processor.label_names)

    for case_idx, (seq, labels_seq) in enumerate(data):
        if len(seq) == 0:
            continue  # Skip empty sequences

        case_all_true = []
        case_all_pred = []

        case_id = f"case_{case_idx}"
        if seq:
            first_proc = seq[0]
            defendant_id = first_proc.get('defendant_id', 'unknown')
            docket_id = first_proc.get('docket_id', 'unknown')
            proc_type = first_proc.get('type', 'unknown')
            case_id = f"{proc_type}_{docket_id}_{defendant_id}"

        try:
            labels_binary = []
            for lbl in labels_seq:
                if isinstance(lbl, torch.Tensor):
                    labels_binary.append(lbl.cpu().numpy())
                else:
                    labels_binary.append(np.array(lbl))
            teacher_seq, _ = build_teacher_forced_sequences(
                seq,
                labels_binary,
                label_processor,
                include_initial_state=True
            )
            with torch.no_grad():
                logits = model(teacher_seq, return_all_timesteps=True)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (T, L)
            
            T = len(labels_binary)
            for timestep in range(T):
                true_vec = labels_binary[timestep]
                k = int(true_vec.sum())
                if k == 0:
                    continue  # Skip steps with no true labels
                prob_vec = probs[timestep]
                pred_vec = predict_with_top_k_mlb(prob_vec, k, label_processor)
                case_all_true.append(true_vec)
                case_all_pred.append(pred_vec)
                all_true.append(true_vec)
                all_pred.append(pred_vec)
                for j, is_true in enumerate(true_vec):
                    if is_true:
                        true_distribution[label_processor.label_names[j]] += 1
                for j, is_pred in enumerate(pred_vec):
                    if is_pred:
                        prediction_distribution[label_processor.label_names[j]] += 1
        except Exception as e:
            print(f"Error processing case {case_idx} ({case_id}): {e}")
            continue
        if case_all_true and case_all_pred:
            case_true = np.array(case_all_true, dtype=np.int8)
            case_pred = np.array(case_all_pred, dtype=np.int8)
            case_errors_binary = (case_true != case_pred).astype(np.int8)
            overall_case_errors = case_errors_binary.mean()
            case_errors.append({
                'case_id': case_id,
                'case_idx': case_idx,
                'overall_error_rate': overall_case_errors,
                'total_labels': case_errors_binary.size,
                'total_errors': case_errors_binary.sum()
            })

    # Convert to numpy arrays for sklearn
    if not all_true or not all_pred:
        print("Warning: No valid predictions generated during evaluation")
        return {
            'SUMMARY': {
                'micro_avg': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                'macro_avg': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                'weighted_avg': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                'total_samples': 0,
                'total_positive_labels': 0
            },
            'CASE_ERRORS_SORTED': case_errors
        }, (None, None)

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    # Calculate metrics
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # Per-label metrics for backward compatibility
    per_label_precision, per_label_recall, per_label_f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    individual_labels = {}
    for i, label_name in enumerate(label_processor.label_names):
        individual_labels[label_name] = {
            'precision': float(per_label_precision[i]),
            'recall': float(per_label_recall[i]),
            'f1': float(per_label_f1[i]),
            'support': int(support[i])
        }

    # Build metrics dictionary
    metrics = {
        'INDIVIDUAL_LABELS': individual_labels,
        'SUMMARY': {
            'micro_avg': {
                'precision': float(micro_precision),
                'recall': float(micro_recall),
                'f1': float(micro_f1)
            },
            'macro_avg': {
                'precision': float(macro_precision),
                'recall': float(macro_recall),
                'f1': float(macro_f1)
            },
            'weighted_avg': {
                'precision': float(weighted_precision),
                'recall': float(weighted_recall),
                'f1': float(weighted_f1)
            },
            'total_samples': int(y_true.shape[0]),
            'total_positive_labels': int(y_true.sum())
        }
    }
    metrics.update(individual_labels)  # backward-compat keys

    # Get per-label confusion matrices using sklearn.metrics.multilabel_confusion_matrix
    cm = multilabel_confusion_matrix(y_true, y_pred)
    
    # Extract True Positives (Correct predictions) for each label
    tp = cm[:, 1, 1].astype(np.int64)                   # (L,) - True Positives per label
    support_vec = y_true.sum(axis=0).astype(np.int64)   # (L,) - Total true instances per label

    # Calculate "Confused" counts:
    # For each true label that was missed, count which predicted labels it was confused with
    misses = ((y_true == 1) & (y_pred == 0)).astype(np.int8)  # (N, L) - missed predictions
    
    # counts[i, j] = number of samples where true label i was missed AND predicted label j occurred
    try:
        from scipy.sparse import csr_matrix
        counts = (csr_matrix(misses).T @ csr_matrix(y_pred)).toarray().astype(np.int64)  # (L, L)
    except Exception:
        # Fallback to regular numpy if scipy not available
        counts = (misses.T @ y_pred).astype(np.int64)  # (L, L)

    # Build the table using label_processor.label_names
    rows = []
    for i, label_name in enumerate(label_processor.label_names):
        if support_vec[i] == 0:
            continue  # Skip labels that never appear as true
            
        # "Correct" column = True Positives for this label
        correct_val = int(tp[i])
        correct_display = "" if (correct_val == 0) else str(correct_val)

        # "Confused" column = list of predicted labels when this true label was missed
        row_counts = counts[i]  # (L,) - confusion counts for label i
        nz_indices = np.flatnonzero(row_counts)  # Non-zero confusion counts
        
        # Format using label_processor.label_names
        confusion_parts = [f"{label_processor.label_names[j]}: {int(row_counts[j])}" for j in nz_indices]
        # Sort by count descending, then by label name
        confusion_parts.sort(key=lambda s: (-int(s.split(": ")[1]), s.split(": ")[0]))
        confused_str = ", ".join(confusion_parts)

        rows.append({
            "True": label_name, 
            "Correct": correct_display, 
            "Confused": confused_str
        })

    # Create DataFrame for the confusion table
    table_df = pd.DataFrame(rows, columns=["True", "Correct", "Confused"])

    # Save the table to Excel
    out_dir = Path(cache_manager.get_cache_dir('models'))
    out_dir.mkdir(parents=True, exist_ok=True)
    if epoch is None:
        if optional_tag:
            out_path = out_dir / f"true_correct_confused_{optional_tag}.xlsx"
        else:
            out_path = out_dir / "true_correct_confused.xlsx"
    else:
        if optional_tag:
            out_path = out_dir / f"true_correct_confused_{optional_tag}_ep{int(epoch):03d}.xlsx"
        else:
            out_path = out_dir / f"true_correct_confused_ep{int(epoch):03d}.xlsx"

    with pd.ExcelWriter(out_path) as writer:
        table_df.to_excel(writer, sheet_name="Confusion_Table", index=False)
    
    print("\n" + "="*80)
    print("CAUSAL MODEL - MICRO-AVERAGED METRICS:")
    print("="*80)
    print(f"   Precision: {micro_precision:.4f}")
    print(f"   Recall:    {micro_recall:.4f}")
    print(f"   F1-Score:  {micro_f1:.4f}")
    print(f"   Steps processed: {len(all_true):,}")
    print(f"   Cases processed: {len(case_errors)}")
    
    print("\n1. True Label Distribution:")
    print("-"*40)
    for label, count in sorted(true_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label:<50} {count:>6}")
    
    print("\n2. Predicted Label Distribution:")
    print("-"*40)
    for label, count in sorted(prediction_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label:<50} {count:>6}")

    print("\n3. Distribution Comparison:")
    print("-"*40)
    print(f"  {'Label':<50} {'True':>8} {'Pred':>8} {'Diff':>8}")
    print("-"*40)
    for label in label_processor.label_names:
        true_count = true_distribution.get(label, 0)
        pred_count = prediction_distribution.get(label, 0)
        diff = pred_count - true_count
        print(f"  {label:<50} {true_count:>8} {pred_count:>8} {diff:>+8}")

    if case_errors:
        print(f"  Average case error rate: {np.mean([c['overall_error_rate'] for c in case_errors]):.3f}")

    # Sort case errors by error rate (highest first)
    case_errors_sorted = sorted(case_errors, key=lambda x: x['overall_error_rate'], reverse=True)

    # Add the confusion table to the metrics for return
    metrics['TRUE_CORRECT_CONFUSED_TABLE'] = table_df
    metrics['CASE_ERRORS_SORTED'] = case_errors_sorted
    
    predictions = (y_true, y_pred)

    return metrics, predictions