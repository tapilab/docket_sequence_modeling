import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import torch
from sklearn.metrics import multilabel_confusion_matrix

class FrequencyBasedRandomClassifier:
    def __init__(self, label_processor, random_seed=42):
        self.label_processor = label_processor
        self.label_names = label_processor.label_names
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        self.label_probabilities = None
        self.total_proceedings = 0
    
    def fit(self, train_sequences):
            print(f"Learning label frequencies from {len(train_sequences)} training sequences...")
            
            # Count label occurrences
            label_counts = Counter()
            total_proceedings = 0
            
            for seq, labels_seq in tqdm(train_sequences, desc="Counting labels"):
                for step_labels in labels_seq:
                    total_proceedings += 1
                    
                    # Handle different label formats (raw strings or indices)
                    if isinstance(step_labels, (list, tuple)) and len(step_labels) > 0:
                        if isinstance(step_labels[0], str):
                            # Raw string labels
                            for label_name in step_labels:
                                if label_name in self.label_names:
                                    label_counts[label_name] += 1
                        else:
                            # Label indices
                            for label_idx in step_labels:
                                if label_idx < len(self.label_names):
                                    label_name = self.label_names[label_idx]
                                    label_counts[label_name] += 1
            
            self.total_proceedings = total_proceedings
            
            # Compute label probabilities
            self.label_probabilities = np.array([
                label_counts.get(self.label_names[i], 0) / total_proceedings 
                if total_proceedings > 0 else 0
                for i in range(len(self.label_names))
            ])
            
            # Print statistics
            print(f"Training Statistics (Total proceedings: {total_proceedings:,}):")
            
            print(f"\n   Label Frequency Distribution:")
            print(f"   {'Label':<50} {'Count':>10} {'Frequency':>12}")
            print("   " + "-" * 72)
            
            # Sort by count for display
            label_count_pairs = [(self.label_names[i], label_counts.get(self.label_names[i], 0)) 
                                for i in range(len(self.label_names))]
            label_count_pairs.sort(key=lambda x: x[1], reverse=True)
            
            for label, count in label_count_pairs[:20]:
                prob = count / total_proceedings if total_proceedings > 0 else 0
                print(f"   {label:<50} {count:>10,} {prob:>11.4%}")
            
            if len(label_count_pairs) > 20:
                print(f"   ... and {len(label_count_pairs) - 20} more labels")
    def predict_sequences(self, val_sequences):
        if self.label_probabilities is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = []
        
        # Pre-compute indices with non-zero probabilities
        nonzero_mask = self.label_probabilities > 0
        nonzero_indices = np.where(nonzero_mask)[0]
        nonzero_probs = self.label_probabilities[nonzero_mask]
        if nonzero_probs.sum() > 0:
            nonzero_probs = nonzero_probs / nonzero_probs.sum()
        
        for seq, labels_seq in tqdm(val_sequences, desc="Generating random predictions"):
            seq_predictions = []
            
            for true_labels in labels_seq:
                # Get actual k from ground truth
                if isinstance(true_labels, (list, tuple)):
                    k = len(true_labels)
                elif isinstance(true_labels, torch.Tensor):
                    k = int(true_labels.sum().item())
                elif isinstance(true_labels, np.ndarray):
                    k = int(true_labels.sum())
                else:
                    k = 0
                
                # Choose k labels based on frequencies
                if k > 0 and len(nonzero_indices) > 0:
                    k_actual = min(k, len(nonzero_indices))
                    predicted_indices = self.rng.choice(
                        nonzero_indices, 
                        size=k_actual, 
                        replace=False, 
                        p=nonzero_probs
                    )
                    predicted_labels = predicted_indices.tolist()
                else:
                    predicted_labels = []
                
                seq_predictions.append(predicted_labels)
            
            predictions.append(seq_predictions)
        
        return predictions
    
    def evaluate(self, val_sequences):
        print(f"\n📊 Evaluating random classifier on {len(val_sequences)} sequences...")
        
        # Get predictions
        predictions = self.predict_sequences(val_sequences)
        
        # Count total proceedings for pre-allocation
        total_proceedings = sum(len(labels_seq) for _, labels_seq in val_sequences)
        
        # Pre-allocate arrays
        num_labels = len(self.label_names)
        all_true = np.zeros((total_proceedings, num_labels), dtype=np.int8)
        all_preds = np.zeros((total_proceedings, num_labels), dtype=np.int8)
        
        # Convert predictions and ground truth to binary format
        idx = 0
        for (seq, labels_seq), pred_seq in zip(val_sequences, predictions):
            for true_labels, pred_labels in zip(labels_seq, pred_seq):
                # Handle different true label formats
                if isinstance(true_labels, (list, tuple)):
                    if len(true_labels) > 0 and isinstance(true_labels[0], str):
                        # String labels
                        for label_name in true_labels:
                            if label_name in self.label_names:
                                label_idx = self.label_names.index(label_name)
                                all_true[idx, label_idx] = 1
                    else:
                        # Label indices
                        for label_idx in true_labels:
                            if label_idx < num_labels:
                                all_true[idx, label_idx] = 1
                elif isinstance(true_labels, torch.Tensor):
                    all_true[idx] = true_labels.cpu().numpy().astype(np.int8)
                elif isinstance(true_labels, np.ndarray):
                    all_true[idx] = true_labels.astype(np.int8)
                
                # Set predicted labels
                for label_idx in pred_labels:
                    if label_idx < num_labels:
                        all_preds[idx, label_idx] = 1
                
                idx += 1
        
        # Calculate per-label metrics
        per_label_precision, per_label_recall, per_label_f1, support = precision_recall_fscore_support(
            all_true, all_preds, average=None, zero_division=0
        )
        
        # Calculate aggregate metrics
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            all_true, all_preds, average='micro', zero_division=0
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            all_true, all_preds, average='macro', zero_division=0
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            all_true, all_preds, average='weighted', zero_division=0
        )
        
        # Calculate confusion matrices for TP, FP, FN
        cm = multilabel_confusion_matrix(all_true, all_preds)
        tp = cm[:, 1, 1].astype(np.int64)
        fp = cm[:, 0, 1].astype(np.int64)
        fn = cm[:, 1, 0].astype(np.int64)
        
        # Build individual labels dictionary
        individual_labels = {}
        for i, label_name in enumerate(self.label_names):
            individual_labels[label_name] = {
                'precision': float(per_label_precision[i]),
                'recall': float(per_label_recall[i]),
                'f1': float(per_label_f1[i]),
                'support': int(support[i]),
                'tp': int(tp[i]),
                'fp': int(fp[i]),
                'fn': int(fn[i])
            }
        
        # Build results dictionary matching project format
        results = {
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
                'total_samples': int(total_proceedings),
                'total_positive_labels': int(all_true.sum())
            }
        }
        
        # Print results
        print(f"\n🎯 RANDOM BASELINE RESULTS:")
        print(f"   Total samples: {total_proceedings:,}")
        print(f"   Total positive labels: {int(all_true.sum()):,}")
        print(f"\n✅ Micro-averaged Precision: {micro_precision:.4f}")
        print(f"✅ Micro-averaged Recall:    {micro_recall:.4f}")
        print(f"✅ Micro-averaged F1-Score:  {micro_f1:.4f}")
        print(f"\n📊 Other averages:")
        print(f"   Macro F1:    {macro_f1:.4f}")
        print(f"   Weighted F1: {weighted_f1:.4f}")
        
        return results


def run_random_baseline_comparison(train_data, val_data, label_processor, random_seed=42):
    print(f"\n{'='*80}")
    print("BASELINE MODEL 2: FREQUENCY-BASED RANDOM CLASSIFIER")
    print("="*80)
    
    try:
        print(f"🎲 Simple random baseline: Global frequencies + Actual k from ground truth")
        print(f"   Training sequences: {len(train_data)}")
        print(f"   Validation sequences: {len(val_data)}")
        print(f"   Total labels: {len(label_processor.label_names)}")
        
        # Initialize and fit
        classifier = FrequencyBasedRandomClassifier(label_processor, random_seed=random_seed)
        classifier.fit(train_data)
        
        # Evaluate
        results = classifier.evaluate(val_data)
        
        print(f"\n✅ Random baseline comparison completed")
        return classifier, results
        
    except Exception as e:
        print(f"\n⚠️ Random baseline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def convert_random_baseline_results_to_table_format(results):
    return results