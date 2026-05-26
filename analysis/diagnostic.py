# diagnostic.py
# Comprehensive diagnostic analysis for training and evaluation

import torch
import numpy as np
from collections import Counter, defaultdict

from config import get_model_config
from training.teacher_forcing import build_teacher_forced_sequences
from training.top_k_helper import predict_with_top_k_mlb

def comprehensive_training_diagnostic(validation_samples, label_processor, trained_model=None, device=None):
    """Find exactly what's wrong with the training/evaluation"""
    model_configuration = get_model_config()
    label_names = label_processor.label_names if label_processor is not None else model_configuration.LABEL_NAMES
    
    print("🔍 COMPREHENSIVE TRAINING DIAGNOSTIC")
    print("=" * 80)
    
    # 1. Check evaluation skipping pattern
    print("\n1. EVALUATION SKIPPING ANALYSIS")
    print("-" * 40)
    
    total_steps = 0
    skipped_steps = 0
    k_distribution = Counter()
    step_details = []
    
    for seq_idx, (seq, labels_seq) in enumerate(validation_samples):
        for i in range(-1, len(seq) - 1):
            total_steps += 1
            if i == -1:
                target_idx = 0
            else:
                target_idx = i + 1
            true_vec = labels_seq[target_idx]
            k = int(true_vec.sum())
            k_distribution[k] += 1
            
            step_info = {
                'seq_idx': seq_idx,
                'step_idx': i,
                'k': k,
                'true_labels': [label_names[j] for j, v in enumerate(true_vec) if v == 1],
                'skipped': k == 0,
                'is_initial': i == -1
            }
            step_details.append(step_info)
            
            if k == 0:
                skipped_steps += 1
    
    print(f"Total prediction steps: {total_steps:,}")
    print(f"Steps with k=0 (skipped): {skipped_steps:,} ({skipped_steps/total_steps*100:.1f}%)")
    print(f"Steps actually evaluated: {total_steps - skipped_steps:,} ({(total_steps-skipped_steps)/total_steps*100:.1f}%)")
    
    initial_steps = sum(1 for s in step_details if s['is_initial'])
    print(f"Initial State predictions: {initial_steps:,}")
    
    print(f"\nLabel count distribution (k values):")
    for k in sorted(k_distribution.keys()):
        count = k_distribution[k]
        pct = count/total_steps*100
        status = "SKIPPED" if k == 0 else "EVALUATED"
        print(f"  k={k}: {count:,} steps ({pct:.1f}%) - {status}")
    
    # 2. Sample what's being skipped vs evaluated
    print(f"\n2. SAMPLE OF SKIPPED vs EVALUATED STEPS")
    print("-" * 40)
    
    skipped_samples = [s for s in step_details if s['skipped']][:5]
    evaluated_samples = [s for s in step_details if not s['skipped']][:5]
    
    print(f"SKIPPED STEPS (k=0) - Examples:")
    for i, step in enumerate(skipped_samples, 1):
        print(f"  {i}. Seq {step['seq_idx']}, Step {step['step_idx']}: k={step['k']}, Labels: {step['true_labels']}")
    
    print(f"\nEVALUATED STEPS (k>0) - Examples:")
    for i, step in enumerate(evaluated_samples, 1):
        print(f"  {i}. Seq {step['seq_idx']}, Step {step['step_idx']}: k={step['k']}, Labels: {step['true_labels']}")
    
    # 3. Check label distribution in training data
    print(f"\n3. TRAINING DATA LABEL DISTRIBUTION")
    print("-" * 40)
    
    all_training_labels = []
    training_k_distribution = Counter()
    
    for seq, labels_seq in validation_samples:
        for step_labels in labels_seq:
            k = int(step_labels.sum()) if hasattr(step_labels, 'sum') else 0
            training_k_distribution[k] += 1
            
            for j, val in enumerate(step_labels):
                if val == 1 and j < len(label_names):
                    all_training_labels.append(label_names[j])
    
    training_label_counts = Counter(all_training_labels)
    
    print(f"Training data label distribution:")
    print(f"  Total label instances: {len(all_training_labels):,}")
    print(f"  Unique label types: {len(training_label_counts)}")
    print(f"  Most common labels:")
    
    for i, (label, count) in enumerate(training_label_counts.most_common(10), 1):
        pct = count/len(all_training_labels)*100 if all_training_labels else 0
        print(f"    {i:2d}. {label}: {count:,} ({pct:.1f}%)")
    
    # 4. Model prediction analysis (if model provided)
    if trained_model is not None and device is not None:
        print(f"\n4. MODEL PREDICTION BEHAVIOR ANALYSIS")
        print("-" * 40)
        
        trained_model.eval()

        # Analyze predictions on first few sequences
        for seq_idx, (seq, labels_seq) in enumerate(validation_samples[:3]):
            print(f"\nSequence {seq_idx + 1} prediction analysis:")

            labels_binary = []
            for lbl in labels_seq:
                if isinstance(lbl, torch.Tensor):
                    labels_binary.append(lbl.cpu().numpy())
                else:
                    labels_binary.append(np.array(lbl))
            
            teacher_seq, _ = build_teacher_forced_sequences(seq, labels_binary, label_processor, include_initial_state=True)

            with torch.no_grad():
                logits = trained_model(teacher_seq, return_all_timesteps=True)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
            for i in range(min(3, len(labels_binary))):
                target_idx = i
                if i == 0:
                    step_desc = "Initial --> 0"
                else:
                    step_desc = f"Step {i-1} --> {i}"
                true_vec = labels_binary[target_idx]
                k = int(true_vec.sum())
                prob_vec = probs[target_idx]
                if prob_vec.ndim > 1:
                    #flatten if needed
                    prob_vec = prob_vec.flatten()
                
                print(f"  {step_desc} (k={k}):")
                print(f"  True labels: {[label_processor.label_names[j] for j in range(len(true_vec)) if true_vec[j] == 1]}")

                # Show top predictions
                predicted_vec = predict_with_top_k_mlb(prob_vec, k, label_processor)
                predicted_labels = [label_processor.label_names[j] for j in range(len(predicted_vec)) if predicted_vec[j] == 1]
                print(f"  Predicted labels: {predicted_labels}")

                # Check correctness
                match = int((true_vec == predicted_vec).sum())
                total = len(true_vec)
                print(f"  Correct predictions: {match}/{total} labels matched")

                # Show top 5 probabilities
                top_indices = np.argsort(prob_vec)[-5:][::-1]
                print(f"  Top 5 predicted labels by probability:")
                for idx in top_indices:
                    lbl = label_processor.label_names[idx]
                    prob = prob_vec[idx]
                    indicator = "✅" if true_vec[idx] == 1 else "❌"
                    print(f"    {lbl}: {prob:.3f} {indicator}")

                if k == 0:
                    print(f"    ⚠️  This step would be SKIPPED in evaluation!")
    
    # 5. Critical issue identification and recommendations
    print(f"\n5. CRITICAL ISSUES & RECOMMENDATIONS")
    print("-" * 40)
    
    issues = []
    
    if skipped_steps / total_steps > 0.5:
        issues.append(f"🚨 SEVERE: {skipped_steps/total_steps*100:.1f}% of steps skipped in evaluation")
    
    if total_steps - skipped_steps < 50:
        issues.append(f"🚨 SEVERE: Only {total_steps - skipped_steps} steps actually evaluated")
    
    if len(training_label_counts) < 5:
        issues.append(f"🚨 SEVERE: Only {len(training_label_counts)} unique label types in training")
    
    if not issues:
        issues.append("✅ No obvious critical issues detected")
    
    for issue in issues:
        print(f"  {issue}")
    
    # Show the smoking gun
    print(f"\n🎯 THE SMOKING GUN:")
    if skipped_steps/total_steps > 0.5:
        print(f"   {skipped_steps/total_steps*100:.1f}% of evaluation steps are SKIPPED!")
        print(f"   Your 'bad' metrics are based on only {(total_steps-skipped_steps)/total_steps*100:.1f}% of the data!")
        print(f"   This explains the terrible performance metrics.")
        print(f"\n   🔧 FIX: Remove the 'if k == 0: continue' line from evaluation function")
    else:
        print(f"   Evaluation skipping is not the main issue ({skipped_steps/total_steps*100:.1f}% skipped)")
        print(f"   Problem is likely data quality or model architecture.")
    
    return {
        'total_steps': total_steps,
        'skipped_steps': skipped_steps,
        'skipped_percentage': skipped_steps/total_steps*100,
        'initial_steps': initial_steps,
        'k_distribution': k_distribution,
        'issues': issues
    }