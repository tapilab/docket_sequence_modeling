from collections import Counter
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from openpyxl.utils.dataframe import dataframe_to_rows
from operator import sub
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import types
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import contextlib
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import gc
import h5py  # For efficient storage of large arrays
import hashlib
import json
import matplotlib.pyplot as plt
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import openpyxl
import os
import pandas as pd
import pickle
import random
import re
import seaborn as sns
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import time
import shutil
import xlsxwriter
from scipy.sparse import csr_matrix
from captum.attr import IntegratedGradients, LayerConductance, LayerActivation
from captum.attr import visualization as viz
from config import get_model_config, COMPLETION_LABELS, EARLY_TERMINATION_LABELS, CASE_CONCLUSION_LABELS
from data.load_data import mount_drive, load_data
from data.process_data import filter_proceedings, remove_multi_defendant_cases, create_combined_dockets, filter_to_magistrate_started_cases, remove_extradition_cases
from data.sequence_builder import create_enhanced_training_sequences_cached
from regex_labels.label_generator_and_cache_manager import generate_and_cache_regex_labels
from models.transformer_creation import create_model_from_data
from training.trainer import train_case_transformer_with_mlb
from training.label_processor import EnhancedLabelProcessor
from training.evaluation import evaluate_with_teacher_forcing
from analysis.cooccurrence import export_cooccurrence_from_valid_completed_cases, create_cooccurrence_matrix_excel
from analysis.frequency_analysis import analyze_raw_label_frequencies_combined, analyze_raw_label_frequencies_singular
from analysis.diagnostic import comprehensive_training_diagnostic
from models.baseline_model import run_baseline
from utils.cache_directory_utils import CacheDirectoryManager
from utils.model_memory_utils import analyze_model_memory, debug_oom_sample
from models.embeddings import extract_and_cache_bert_embeddings_streaming
from analysis.byclerk import analyze_clerk_data
from models.interpretability import CaseTransformerInterpretability, CaptumWrapper
from models.most_confident_wrong_answer_analyzer import run_confident_misclassification_analysis
from models.ablation import run_ablation_study_function, compare_ablation_studies_dataframe
from copy import deepcopy
from training.top_k_helper import predict_with_top_k_mlb
from training.teacher_forcing import build_teacher_forced_sequences

# Filter to only use completed cases while doing experiments
def filter_to_completed_cases(court_data, valid_defendant_case_ids, cached_outcome_labels):
    completed_cases = []

    mag_lookup = {}
    if 'mag' in court_data and not court_data['mag'].empty:
        for mag_num, group in tqdm(court_data['mag'].groupby('mag_num'), desc="Indexing MAG"): 
            mag_lookup[mag_num] = group.to_dict('records')

    cdc_lookup = {}
    if 'cdc' in court_data and not court_data['cdc'].empty:
        for case_id, group in tqdm(court_data['cdc'].groupby('case_id'), desc="Indexing CDC"): 
            cdc_lookup[case_id] = group.to_dict('records')
       
    for mag_num, case_id, defendant_id in tqdm(valid_defendant_case_ids, desc="Filtering to completed cases"):
        #get all proceeding IDs for this case
        mag_procs = []
        cdc_procs = []
        if pd.notna(mag_num):
            mag_procs = mag_lookup.get(mag_num, [])
        
        if pd.notna(case_id):
            cdc_procs = cdc_lookup.get(case_id, [])
        
        all_procs = mag_procs + cdc_procs

        if not all_procs:
            continue

        all_procs.sort(key= lambda p: pd.to_datetime(p.get('date', pd.NaT), errors='coerce'))

        labeled_procs = []
        for proc in all_procs:
            proc_id = proc.get('proceeding_id')
            if proc_id in cached_outcome_labels:
                labels = cached_outcome_labels.get(proc_id, [])
                #skip if no labels or only continuance
                if not labels or labels == ['CONTINUANCE_OCCURRED']:
                    continue
                labeled_procs.append(
                    {
                        'proceeding': proc,
                        'labels': labels
                    }
                )
        if not labeled_procs:
            continue
        is_completed = False
        # Check 1: Final labeled proceeding has early termination label
        final_labels = set(labeled_procs[-1]['labels'])
        if final_labels & EARLY_TERMINATION_LABELS:
            is_completed = True
        # Check 2: Any proceeding has case conclusion label
        if not is_completed:
            for labeled_proc in labeled_procs:
                if set(labeled_proc['labels']) & CASE_CONCLUSION_LABELS: 
                    is_completed = True
                    break
        if is_completed:
            completed_cases.append((mag_num, case_id, defendant_id))
    return completed_cases

def filter_proceedings_to_end_of_case(sequences, label_sequences_raw):
    TOP_PRIORITY_LABELS = {
        'SENTENCE_THAT_INCLUDES_JAIL_TIME_WAS_IMPOSED',
        'SENTENCE_THAT_DOES_NOT_INCLUDE_JAIL_TIME_WAS_IMPOSED',
    }
    SECOND_PRIORITY_LABELS = {
        'TRIAL_VERDICT_REACHED'
    }

    filtered_sequences = []
    filtered_label_sequences = []
    removed_count = 0

    count_dicionary_of_removed = {
        'top_priority': 0,
        'second_priority': 0,
        'early_termination': 0,
        'no_truncation': 0
    }
    for seq, labels_seq in tqdm(zip(sequences, label_sequences_raw), desc="Filtering proceedings to end of case", total=len(sequences)):
        end_index = None
        truncation_reason  = None
        for i in range(len(labels_seq)):
            labels_set = set(labels_seq[i]) if isinstance(labels_seq[i], list) else {labels_seq[i]}
            if labels_set & TOP_PRIORITY_LABELS:
                end_index = i
                truncation_reason = 'top_priority'
                break  # Stop at FIRST top priority label
        if end_index is None:
            for i in range(len(labels_seq)):
                labels_set = set(labels_seq[i]) if isinstance(labels_seq[i], list) else {labels_seq[i]}
                if labels_set & SECOND_PRIORITY_LABELS:
                    end_index = i
                    truncation_reason = 'second_priority'
                    break  # Stop at FIRST second priority label
        if end_index is None:
            final_labels = set(labels_seq[-1]) if isinstance(labels_seq[-1], list) else {labels_seq[-1]}
            if final_labels & EARLY_TERMINATION_LABELS:
                end_index = len(labels_seq) - 1
                truncation_reason = 'early_termination'
        
        if end_index is not None:
            truncated_sequence = seq[:end_index + 1]
            truncated_labels_sequence = labels_seq[:end_index + 1]

            proceedings_removed = len(seq) - len(truncated_sequence)
            if proceedings_removed > 0:
                removed_count += proceedings_removed
            filtered_sequences.append(truncated_sequence)
            filtered_label_sequences.append(truncated_labels_sequence)
            if truncation_reason in count_dicionary_of_removed:
                count_dicionary_of_removed[truncation_reason] += proceedings_removed
        else:
            filtered_sequences.append(seq)
            filtered_label_sequences.append(labels_seq)
            count_dicionary_of_removed['no_truncation'] += 1
    print(f"Total proceedings removed: {removed_count}")
    print("Breakdown of removed proceedings by reason:")
    for reason, count in count_dicionary_of_removed.items():
        print(f"  {reason}: {count}")
    return filtered_sequences, filtered_label_sequences

## Converts label strings into binary vector representation using label-to-index mapping
def build_case_counts_df(court_data, cached_outcome_labels_v2):
    # 1) which proceedings have labels?
    labeled_pids = {pid for pid, labs in cached_outcome_labels_v2.items() if labs}

    # 2) light views
    mag = court_data['mag'][['mag_num','proceeding_id']].copy()
    cdc = court_data['cdc'][['case_id','proceeding_id']].copy()

    # 3) add is_labeled (vectorized)
    mag['is_labeled'] = mag['proceeding_id'].isin(labeled_pids).values
    cdc['is_labeled'] = cdc['proceeding_id'].isin(labeled_pids).values

    # 4) group once
    mag_counts = (mag.groupby('mag_num', dropna=True)
                    .agg(total_mag=('proceeding_id','size'),
                         labeled_mag=('is_labeled','sum'))
                    .reset_index())

    cdc_counts = (cdc.groupby('case_id', dropna=True)
                    .agg(total_cdc=('proceeding_id','size'),
                         labeled_cdc=('is_labeled','sum'))
                    .reset_index())

    # 5) start from combined_dockets distinct rows
    base = (court_data['combined_dockets']
              [['mag_num','case_id','defendant_id']]
              .drop_duplicates()
              .copy())

    # 6) merge counts; NaNs -> 0
    out = (base.merge(mag_counts, on='mag_num', how='left')
               .merge(cdc_counts, on='case_id',  how='left'))
    for col in ['total_mag','labeled_mag','total_cdc','labeled_cdc']:
        if col in out:
            out[col] = out[col].fillna(0).astype('int64')
        else:
            out[col] = 0

    # 7) totals + validity
    out['total_procs']   = out['total_mag']   + out['total_cdc']
    out['total_labeled'] = out['labeled_mag'] + out['labeled_cdc']
    out['valid'] = (out['total_procs'] >= 2) & (out['total_labeled'] >= 2)

    return out[['mag_num','case_id','defendant_id','total_procs','total_labeled','valid']]

# Remove proceedings that have zero labels from court_data.

def filter_zero_label_proceedings(court_data, cached_labels):
    # Find proceeding IDs with at least one label
    labeled_proceeding_ids = {
        proc_id for proc_id, labels in cached_labels.items()
        if labels and len(labels) > 0
    }
    
    print(f"Total proceedings in cache: {len(cached_labels)}")
    print(f"Proceedings with labels: {len(labeled_proceeding_ids)}")
    print(f"Proceedings with 0 labels: {len(cached_labels) - len(labeled_proceeding_ids)}")
    
    # Filter MAG proceedings
    original_mag_count = len(court_data['mag'])
    court_data['mag'] = court_data['mag'][
        court_data['mag']['proceeding_id'].isin(labeled_proceeding_ids)
    ].copy()
    removed_mag = original_mag_count - len(court_data['mag'])
    print(f"Removed {removed_mag} MAG proceedings with 0 labels")
    
    # Filter CDC proceedings
    original_cdc_count = len(court_data['cdc'])
    court_data['cdc'] = court_data['cdc'][
        court_data['cdc']['proceeding_id'].isin(labeled_proceeding_ids)
    ].copy()
    removed_cdc = original_cdc_count - len(court_data['cdc'])
    print(f"Removed {removed_cdc} CDC proceedings with 0 labels")
    
    # Also clean the cached_labels dictionary itself
    cached_labels_cleaned = {
        proc_id: labels for proc_id, labels in cached_labels.items()
        if labels and len(labels) > 0
    }
    
    return court_data, cached_labels_cleaned

def transform_labels_to_binary(labels_raw, label_processor):
    if isinstance(labels_raw, list) and len(labels_raw) > 0:
        if isinstance(labels_raw[0], list):
            # List of label sequences
            return [label_processor.transform_sequence(labels_seq) 
                    for labels_seq in labels_raw]
        else:
            # Single label sequence
            return label_processor.transform_sequence(labels_raw)
    return labels_raw


def extract_case_info_from_sequence(sequence, court_data=None):
    case_info = {}
    # Try to get info from sequence proceedings with docket IDs
    for proc in sequence:
        docket_id = proc.get('docket_id')
        proc_type = proc.get('type')
        
        
        if docket_id and court_data is not None:
            if proc_type == 'mag':
                mag_matches = court_data['mag_dockets'][
                    court_data['mag_dockets']['mag_num'] == docket_id
                ]
                
                if not mag_matches.empty:
                    case_info['mag_num'] = docket_id
                    case_info['case_id'] = mag_matches.iloc[0].get('case_id')
                    case_info['defendant_id'] = mag_matches.iloc[0].get('defendant_id')
                    break
            elif proc_type == 'cdc':
                cdc_matches = court_data['cdc_dockets'][
                    court_data['cdc_dockets']['case_id'] == docket_id
                ]
                if not cdc_matches.empty:
                    case_info['case_id'] = docket_id
                    case_info['mag_num'] = cdc_matches.iloc[0].get('mag_num')
                    case_info['defendant_id'] = cdc_matches.iloc[0].get('defendant_id')
                    break
    
    # Fallback: use docket_id as mag_num for mag proceedings
    if not case_info:
        first_proc = sequence[0]
        if first_proc.get('type') == 'mag' and first_proc.get('docket_id'):
            case_info = {
                'mag_num': first_proc.get('docket_id'),  # Use docket_id as mag_num
                'case_id': 'Unknown',
                'defendant_id': first_proc.get('defendant_id', 'Unknown')
            }
        elif first_proc.get('type') == 'cdc' and first_proc.get('docket_id'):
            case_info = {
                'mag_num': 'Unknown',
                'case_id': first_proc.get('docket_id'),  # Use docket_id as case_id
                'defendant_id': first_proc.get('defendant_id', 'Unknown')
            }
        else:
            case_info = {
                'mag_num': 'Unknown',
                'case_id': 'Unknown',
                'defendant_id': first_proc.get('defendant_id', 'Unknown')
            }
    
    return case_info


def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def collect_labels_from_binary(labels_binary, label_processor):
    """
    Collect all unique labels from binary label data.
    """
    import numpy as np
    import torch
    
    labels_set = set()
    # Handle numpy array
    if isinstance(labels_binary, np.ndarray):
        if labels_binary.ndim == 1:
            # Single step
            active_indices = np.where(np.abs(labels_binary - 1.0) < 0.5)[0]
            for label_idx in active_indices:
                if label_idx < len(label_processor.label_names):
                    labels_set.add(label_processor.label_names[label_idx])
        else:
            # Multiple steps
            for step_idx in range(labels_binary.shape[0]):
                step_labels = labels_binary[step_idx]
                active_indices = np.where(np.abs(step_labels - 1.0) < 0.5)[0]
                for label_idx in active_indices:
                    if label_idx < len(label_processor.label_names):
                        label_name = label_processor.label_names[label_idx]
                        labels_set.add(label_name)
    
    return labels_set

def convert_data_to_format(data, label_processor, format_type='tensor'):
    converted_data = []
    
    for seq, labels_raw in data:
        if format_type == 'tensor':
            labels_binary = label_processor.transform_sequence(labels_raw)
            labels_tensors = [torch.tensor(step, dtype=torch.float32) 
                            for step in labels_binary]
            converted_data.append((seq, labels_tensors))
        elif format_type == 'binary':
            labels_binary = label_processor.transform_sequence(labels_raw)
            converted_data.append((seq, labels_binary))
        else:  # 'raw'
            converted_data.append((seq, labels_raw))
    
    return converted_data

def filter_sequences_with_meaningful_labels(sequences, label_sequences_raw):
    training_data = list(zip(sequences, label_sequences_raw))
    
    print("Filtering sequences with meaningful labels...")
    filtered_training_data = []
    
    for seq, labels_seq in training_data:
        has_meaningful_labels = any(
            len(labels_seq[i+1]) > 0
            for i in range(len(labels_seq) - 1)
        )
        if len(seq) >= 2 and has_meaningful_labels:
            filtered_training_data.append((seq, labels_seq))
    
    print(f"Filtered from {len(training_data)} to {len(filtered_training_data)} sequences")
    return filtered_training_data


def split_train_validation(filtered_data, test_size=0.2, random_state=None):
    
    train_sequences_and_labels, val_sequences_and_labels = train_test_split(
        filtered_data, 
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"Training sequences: {len(train_sequences_and_labels)}")
    print(f"Validation sequences: {len(val_sequences_and_labels)}")
    
    return train_sequences_and_labels, val_sequences_and_labels

def k_fold_cross_validation_split(filtered_data, k=5):
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=k, shuffle=True)
    folds = []
    
    data_array = np.array(filtered_data)
    
    for train_index, val_index in kf.split(data_array):
        train_data = data_array[train_index].tolist()
        val_data = data_array[val_index].tolist()
        folds.append((train_data, val_data))
    
    return folds


def prepare_sequences_and_labels(data_with_labels):
    sequences_only = [seq for seq, _ in data_with_labels]
    labels_raw = [labels for _, labels in data_with_labels]
    return sequences_only, labels_raw



def generate_cooccurrence_matrix_excel(data, label_processor, output_folder, output_filename, for_what):
    cooccurences_output_file = output_folder / output_filename
    
    # Extract sequences and labels from data tuples
    sequences = [seq for seq, _ in data]
    labels_binary = [labels for _, labels in data]
    
    # Generate co-occurrence matrix Excel file
    create_cooccurrence_matrix_excel(
        sequences,
        labels_binary,
        label_processor.label_names,
        cooccurences_output_file
    )
    
    print(f"Co-occurrence matrix for {for_what} saved to {cooccurences_output_file}")

def perform_frequency_analysis(train_data, val_data):
    analyze_raw_label_frequencies_singular(
        train_data, 
        "TRAINING SET LABEL FREQUENCY ANALYSIS (Raw Labels)"
    )
    
    analyze_raw_label_frequencies_singular(
        val_data, 
        "VALIDATION SET LABEL FREQUENCY ANALYSIS (Raw Labels)"
    )
    
    analyze_raw_label_frequencies_combined(train_data, val_data)

def debug_model_sample(train_data, case_outcome_model):
    if train_data:
        sample_sequence, sample_labels = train_data[0]
        sample_proceeding = sample_sequence[0]
        
        print(f"Total model parameters: {sum(p.numel() for p in case_outcome_model.parameters()):,}\n")
        debug_oom_sample(sample_proceeding)

def train_model(case_outcome_model, train_data, label_processor, val_data, 
                computation_device, epochs=20, batch_size=4, 
                use_mixed_precision=False, use_learning_rate_scheduler=False, gradient_clip_norm=1.0,
                weighted_loss=False, fold_idx=None, cache_manager=None):
    optional_tag = f"fold_{fold_idx}" if fold_idx is not None else None
    trained_model, logger = train_case_transformer_with_mlb(
        case_outcome_model,
        train_data,
        label_processor,
        val_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        device=computation_device,
        directory_for_output=cache_manager.get_cache_dir('models'),
        optional_tag=optional_tag,
        use_mixed_precision=use_mixed_precision,
        use_learning_rate_scheduler=use_learning_rate_scheduler,
        gradient_clip_norm=gradient_clip_norm,
        weighted_loss=weighted_loss,
        cache_manager=cache_manager
    )
    
    return trained_model, logger

def prepare_validation_data_for_eval(val_data, label_processor):
    return convert_data_to_format(val_data, label_processor, format_type='tensor')
    
def evaluate_model(trained_model, val_data_for_eval, computation_device, label_processor, fold_idx=None, cache_manager=None):
    print("Running comprehensive evaluation with confusion table...")
    optional_tag = f"fold_{fold_idx}" if fold_idx is not None else None
    validation_metrics, predictions = evaluate_with_teacher_forcing(
        trained_model, 
        val_data_for_eval, 
        computation_device,
        label_processor,
        epoch=None,  # Final evaluation
        optional_tag=optional_tag,
        cache_manager=cache_manager
    )
    
    return validation_metrics, predictions
def print_micro_averaged_metrics(validation_metrics):
    print("\n" + "="*80)
    print("METRICS - MICRO-AVERAGED")
    print("="*80)
    
    if 'SUMMARY' in validation_metrics:
        summary = validation_metrics['SUMMARY']
        print(f"âœ… Micro-averaged Precision: {summary['micro_avg']['precision']:.4f}")
        print(f"âœ… Micro-averaged Recall:    {summary['micro_avg']['recall']:.4f}")
        print(f"âœ… Micro-averaged F1-Score:  {summary['micro_avg']['f1']:.4f}")
        print(f"   ðŸ“ Note: Weighted by number of true examples per label")
        
        print(f"\nðŸ“Š Other averages for comparison:")
        print(f"   Macro F1:    {summary['macro_avg']['f1']:.4f}")
        print(f"   Weighted F1: {summary['weighted_avg']['f1']:.4f}")
        print(f"   Total samples: {summary['total_samples']:,}")
        print(f"   Total positive labels: {summary['total_positive_labels']:,}")

def print_top_performing_labels(validation_metrics, top_n=10):
    if 'INDIVIDUAL_LABELS' in validation_metrics:
        individual = validation_metrics['INDIVIDUAL_LABELS']
        sorted_labels = sorted(individual.items(), key=lambda x: x[1]['f1'], reverse=True)
        
        print(f"\nðŸ† TOP {top_n} PERFORMING LABELS BY F1-SCORE:")
        print(f"{'Rank':<4} {'Label':<45} {'F1':<6} {'Precision':<9} {'Recall':<7} {'Support':<8}")
        print("-" * 85)
        
        for i, (label_name, metrics) in enumerate(sorted_labels[:top_n], 1):
            print(f"{i:<4} {label_name:<45} {metrics['f1']:<6.3f} "
                  f"{metrics['precision']:<9.3f} {metrics['recall']:<7.3f} {metrics['support']:<8}")


def predict_case_labels(trained_model, seq, labels_binary, label_processor):   
    pred_labels_set = set()

    teacher_seq, _ = build_teacher_forced_sequences(seq, labels_binary, label_processor, include_initial_state=True)

    with torch.no_grad():
        logits = trained_model(teacher_seq, return_all_timesteps=True)
        probs = torch.sigmoid(logits).squeeze(0).cpu()
    for i in range(len(labels_binary)):
        try:
            true_binary = labels_binary[i]
            k = int(true_binary.sum())
            
            if k == 0:
                continue
            
            prob_step = probs[i]
            pred_binary = predict_with_top_k_mlb(prob_step, k, label_processor)
            
            # Use shared helper instead of inline label collection
            step_pred_labels = collect_labels_from_binary(pred_binary, label_processor)
            pred_labels_set.update(step_pred_labels)
        except Exception:
            continue
    return pred_labels_set



def calculate_case_metrics(true_labels_set, pred_labels_set):
    correct_predictions = true_labels_set & pred_labels_set
    false_positives = pred_labels_set - true_labels_set
    false_negatives = true_labels_set - pred_labels_set
    
    total_true = len(true_labels_set)
    total_pred = len(pred_labels_set)
    total_correct = len(correct_predictions)
    total_fp = len(false_positives)
    total_fn = len(false_negatives)
    
    metrics = calculate_metrics(total_correct, total_fp, total_fn)
    
    return {
        'true_labels': sorted(list(true_labels_set)),
        'pred_labels': sorted(list(pred_labels_set)),
        'correct': sorted(list(correct_predictions)),
        'false_positives': sorted(list(false_positives)),
        'false_negatives': sorted(list(false_negatives)),
        'num_true': total_true,
        'num_pred': total_pred,
        'num_correct': total_correct,
        'num_fp': total_fp,
        'num_fn': total_fn,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'exact_match': (true_labels_set == pred_labels_set)
    }

def evaluate_case_level(trained_model, val_data, label_processor):
    print("\n" + "="*80)
    print("CASE-LEVEL EVALUATION")
    print("="*80)
    print("Comparing complete predicted label sets vs. complete true label sets per case")
    print("-"*80)
    
    case_results = []
    trained_model.eval()
    
    with torch.no_grad():
        for case_idx, (seq, labels_raw) in enumerate(val_data):
            labels_binary = label_processor.transform_sequence(labels_raw)
            true_labels_set = collect_labels_from_binary(labels_binary, label_processor)
            pred_labels_set = predict_case_labels(trained_model, seq, labels_binary, label_processor)            
            case_metrics = calculate_case_metrics(true_labels_set, pred_labels_set)
            case_metrics['case_idx'] = case_idx
            case_results.append(case_metrics)
    
    return case_results

def print_best_worst_cases(case_results, top_n=5):
    sorted_by_f1 = sorted(case_results, key=lambda x: x['f1'], reverse=True)
    
    print(f"\nðŸ† TOP {top_n} CASES BY F1-SCORE:")
    for i, case in enumerate(sorted_by_f1[:top_n], 1):
        print(f"   {i}. Case {case['case_idx']}: F1={case['f1']:.3f}, "
              f"P={case['precision']:.3f}, R={case['recall']:.3f}, "
              f"True={case['num_true']}, Pred={case['num_pred']}, Correct={case['num_correct']}")
    
    print(f"\nðŸ“‰ BOTTOM {top_n} CASES BY F1-SCORE:")
    for i, case in enumerate(sorted_by_f1[-top_n:], 1):
        print(f"   {i}. Case {case['case_idx']}: F1={case['f1']:.3f}, "
              f"P={case['precision']:.3f}, R={case['recall']:.3f}, "
              f"True={case['num_true']}, Pred={case['num_pred']}, Correct={case['num_correct']}")

def print_f1_distribution(case_results):
    f1_scores = [c['f1'] for c in case_results]
    
    print(f"\nðŸ“Š F1-SCORE DISTRIBUTION:")
    print(f"   Min:    {np.min(f1_scores):.3f}")
    print(f"   25th percentile: {np.percentile(f1_scores, 25):.3f}")
    print(f"   Median: {np.median(f1_scores):.3f}")
    print(f"   75th percentile: {np.percentile(f1_scores, 75):.3f}")
    print(f"   Max:    {np.max(f1_scores):.3f}")
def print_case_level_metrics(case_results):
    print("\n" + "="*80)
    print("AGGREGATE CASE-LEVEL METRICS")
    print("="*80)
    
    # Overall metrics
    avg_precision = np.mean([c['precision'] for c in case_results])
    avg_recall = np.mean([c['recall'] for c in case_results])
    avg_f1 = np.mean([c['f1'] for c in case_results])
    exact_match_rate = np.mean([c['exact_match'] for c in case_results])
    
    print(f"\nðŸŽ¯ CASE-LEVEL AVERAGES:")
    print(f"   Average Precision: {avg_precision:.4f}")
    print(f"   Average Recall:    {avg_recall:.4f}")
    print(f"   Average F1-Score:  {avg_f1:.4f}")
    print(f"   Exact Match Rate:  {exact_match_rate:.4f} ({exact_match_rate*100:.1f}%)")
    
    # Distribution analysis
    print(f"\nðŸ“Š CASE-LEVEL DISTRIBUTIONS:")
    print(f"   Avg true labels per case:     {np.mean([c['num_true'] for c in case_results]):.1f}")
    print(f"   Avg predicted labels per case: {np.mean([c['num_pred'] for c in case_results]):.1f}")
    print(f"   Avg correct per case:          {np.mean([c['num_correct'] for c in case_results]):.1f}")
    print(f"   Avg false positives per case:  {np.mean([c['num_fp'] for c in case_results]):.1f}")
    print(f"   Avg false negatives per case:  {np.mean([c['num_fn'] for c in case_results]):.1f}")
    
    # Perfect predictions
    perfect_cases = sum(1 for c in case_results if c['exact_match'])
    print(f"\nâœ… EXACT MATCHES:")
    print(f"   Cases with exact match: {perfect_cases}/{len(case_results)} ({perfect_cases/len(case_results)*100:.1f}%)")
    
    # Show best and worst cases
    print_best_worst_cases(case_results)
    print_f1_distribution(case_results)

def save_case_level_results(case_results, output_path):
    """Save case-level evaluation results to Excel"""
    from pathlib import Path
    import pandas as pd
    
    case_df = pd.DataFrame(case_results)
    
    # Check what columns actually exist
    available_cols = case_df.columns.tolist()
    
    with pd.ExcelWriter(output_path) as writer:
        # Summary statistics
        summary_dict = {
            'avg_precision': case_df['precision'].mean(),
            'avg_recall': case_df['recall'].mean(),
            'avg_f1': case_df['f1'].mean(),
            'exact_match_rate': case_df['exact_match'].mean() if 'exact_match' in available_cols else 0,
            'perfect_cases': case_df['exact_match'].sum() if 'exact_match' in available_cols else 0,
            'total_cases': len(case_results)
        }
        summary_df = pd.DataFrame([summary_dict])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Simplified metrics - only use columns that exist
        simple_cols = ['case_idx', 'num_true', 'num_pred', 
                       'num_correct', 'num_fp', 'num_fn', 
                       'precision', 'recall', 'f1']
        
        # Add optional columns if they exist
        if 'num_proceedings' in available_cols:
            simple_cols.insert(1, 'num_proceedings')
        if 'exact_match' in available_cols:
            simple_cols.append('exact_match')
        
        # Filter to only existing columns
        simple_cols = [col for col in simple_cols if col in available_cols]
        simple_df = case_df[simple_cols].copy()
        simple_df.to_excel(writer, sheet_name='Case_Metrics', index=False)
        
        # Full details
        case_df.to_excel(writer, sheet_name='Full_Details', index=False)
    
    print(f"ðŸ’¾ Case-level results saved to: {output_path}")


def calculate_label_metrics_from_cases(case_results, label_processor):
    print("\n" + "="*80)
    print("CASE-LEVEL EVALUATION BY LABEL")
    print("="*80)
    
    label_tp = defaultdict(int)
    label_fp = defaultdict(int)
    label_fn = defaultdict(int)
    label_support = defaultdict(int)
    
    for case in case_results:
        true_labels = set(case['true_labels'])
        pred_labels = set(case['pred_labels'])
        correct = true_labels & pred_labels
        false_pos = pred_labels - true_labels
        false_neg = true_labels - pred_labels
        
        for label in correct:
            label_tp[label] += 1
        for label in false_pos:
            label_fp[label] += 1
        for label in false_neg:
            label_fn[label] += 1
        for label in true_labels:
            label_support[label] += 1
    
    case_level_label_metrics = []
    all_labels = sorted(label_processor.label_names)
    
    for label in all_labels:
        tp = label_tp[label]
        fp = label_fp[label]
        fn = label_fn[label]
        support = label_support[label]
        
        # Use shared metric calculation
        metrics = calculate_metrics(tp, fp, fn)
        
        case_level_label_metrics.append({
            'Label': label,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1'],
            'Support': support,
            'TP': tp,
            'FP': fp,
            'FN': fn
        })
    
    label_df = pd.DataFrame(case_level_label_metrics)
    label_df = label_df.sort_values(by='F1_Score', ascending=False).reset_index(drop=True)
    
    print("ðŸ“Š Aggregate Case-Level Metrics Per Label:")
    print("-" * 80)
    print(label_df.to_string())
    
    return label_df

def save_label_metrics_to_excel(label_df, case_level_output_path):
    try:
        with pd.ExcelWriter(case_level_output_path, mode='a', engine='openpyxl', 
                           if_sheet_exists='replace') as writer:
            label_df.to_excel(writer, sheet_name='Case_Level_By_Label', index=False)
        print(f"\nðŸ’¾ Case-level by label metrics appended to: {case_level_output_path}")
    except FileNotFoundError:
        print(f"\nâš ï¸ Could not append to Excel. File not found: {case_level_output_path}")
        label_df.to_excel(case_level_output_path, sheet_name='Case_Level_By_Label', index=False)
        print(f"   Saved as a new file instead.")
    except Exception as e:
        print(f"\nâš ï¸ An error occurred while saving to Excel: {e}")
def run_diagnostic_analysis(val_data_for_eval, label_processor, trained_model, device):
    print(f"\n{'='*80}")
    print("DIAGNOSTIC ANALYSIS")
    print("="*80)
    
    diagnostic_results = comprehensive_training_diagnostic(
        val_data_for_eval,
        label_processor, 
        trained_model, 
        device
    )
    
    return diagnostic_results

def run_baseline_comparison(train_data, val_data, label_processor, epochs=20, batch_size=4):    
    print(f"\n{'='*80}")
    print("BASELINE MODEL COMPARISON")
    print("="*80)
    
    try:
        baseline_train = convert_data_to_format(train_data, label_processor, format_type='tensor')
        baseline_val = convert_data_to_format(val_data, label_processor, format_type='tensor')
        
        baseline_model, baseline_results = run_baseline(
            baseline_train, 
            baseline_val, 
            label_processor,
            epochs=epochs, 
            batch_size=batch_size
        )
        
        print("âœ… Baseline model comparison completed")
        return baseline_model, baseline_results, baseline_results.get('preds_binary')
        
    except Exception as e:
        print(f"âš ï¸ Baseline model failed: {e}")
        return None, None

def initialize_interpretability_analyzer(trained_model, label_processor, device):
    interpreter = CaseTransformerInterpretability(
        model=trained_model,
        label_processor=label_processor,
        verbose=True,
        device=device
    )
    
    print("âœ… Interpretability analyzer initialized")
    print(f"   Model: {type(trained_model).__name__}")
    print(f"   Device: {device}")
    print(f"   Number of labels: {len(label_processor.label_names)}")
    
    return interpreter
def select_samples_for_analysis(val_sequences, num_samples=3):
    sample_indices = random.sample(
        range(len(val_sequences)), 
        min(num_samples, len(val_sequences))
    )
    
    print(f"\nðŸ“Š Selected {len(sample_indices)} samples for interpretability analysis")
    return sample_indices

def get_model_prediction(trained_model, sequence, label_processor):
    with torch.no_grad():
        logits = trained_model(sequence)
        probs = torch.sigmoid(logits).squeeze(0).cpu()
        
        # If probs is 2D (timesteps x labels), take the last timestep
        if probs.dim() > 1:
            probs = probs[-1]
    
    top_pred_idx = torch.argmax(probs).item()
    top_pred_label = label_processor.label_names[top_pred_idx]
    top_pred_prob = probs[top_pred_idx].item()
    
    return top_pred_idx, top_pred_label, top_pred_prob, probs
    
def compute_attributions(interpreter, sequence, target_pred_idx, target_proceeding_idx):
    print(f"\nðŸ” Computing attributions for class index {target_pred_idx}...")
    
    try:
        attributions, component_attrs = interpreter.attribute_sequence_features(
            seq=sequence,
            target_class_idx=target_pred_idx,
            target_proceeding_idx=target_proceeding_idx,
            method="IntegratedGradients",
            n_steps=100,
            return_component_attributions=True
        )
        print("âœ… Attributions computed successfully")
        return attributions, component_attrs
        
    except Exception as e:
        print(f"âŒ Attribution failed: {e}")
        return None, None



def visualize_attributions(interpreter, attributions, component_attrs, target_pred_idx, 
                          case_info, sample_idx, interpretability_folder):
    case_id_str = f"Case {case_info.get('case_id', 'Unknown')}"
    
    # Visualize proceeding importance
    print(f"\nðŸ“Š Visualizing proceeding importance...")
    interpreter.visualize_proceeding_importance(
        attributions_tensor=attributions,
        target_class_idx=target_pred_idx,
        case_id=case_id_str,
        save_path=interpretability_folder / f"interpretability_proceeding_importance_sample_{sample_idx + 1}.png",
        show=True
    )
    
    # Visualize component importance over time
    print(f"ðŸ“ˆ Visualizing component importance over time...")
    interpreter.visualize_component_importance(
        component_attributions=component_attrs,
        target_class_idx=target_pred_idx,
        case_id=case_id_str,
        save_path=interpretability_folder / f"interpretability_component_importance_sample_{sample_idx + 1}.png",
        show=True
    )

def get_and_save_feature_summary(interpreter, sequence, target_pred_idx, 
                                target_proceeding_idx, sample_idx, interpretability_folder):
    print(f"\nðŸ“‹ Feature Importance Summary:")
    
    feature_summary_df = interpreter.get_feature_summary(
        seq=sequence,
        target_class_idx=target_pred_idx,
        target_proceeding_idx=target_proceeding_idx
    )
    
    print(feature_summary_df.to_string())
    
    # Save to Excel
    feature_summary_df.to_excel(
        interpretability_folder / f"feature_importance_summary_sample_{sample_idx + 1}.xlsx",
        index=False
    )
    print(f"ðŸ’¾ Feature importance summary saved to Excel")
    
    return feature_summary_df

def analyze_single_sample(interpreter, sequence, labels, labels_binary, case_info, 
                         sample_idx, trained_model, label_processor, interpretability_folder):
    print(f"\n{'='*70}")
    print(f"ANALYZING SAMPLE {sample_idx + 1}")
    print(f"{'='*70}")
    
    print(f"ðŸ“‹ Case Info: Mag#{case_info.get('mag_num', 'N/A')} | "
          f"CDC#{case_info.get('case_id', 'N/A')} | "
          f"Def#{case_info.get('defendant_id', 'N/A')}")
    print(f"ðŸ“ Sequence length: {len(sequence)} proceedings")
    
    # Analyze the final prediction
    target_proceeding_idx = len(sequence) - 2

    
    # Get true labels
    true_binary = labels_binary[target_proceeding_idx + 1]
    true_labels = label_processor.inverse_transform_step(true_binary)
    
    if len(true_labels) == 0:
        print(f"âš ï¸ No labels for target proceeding, skipping...")
        return None
    
    # Get model's prediction
    sequence_slice = sequence[:target_proceeding_idx + 1]
    top_pred_idx, top_pred_label, top_pred_prob, probs = get_model_prediction(
        trained_model = trained_model,
        sequence = sequence_slice,
        label_processor = label_processor
    )

    
    print(f" Target Proceeding: Step {target_proceeding_idx + 1}")
    print(f"   True labels: {list(true_labels)}")
    print(f"   Top prediction: {top_pred_label} (prob: {top_pred_prob:.3f})")
    
    # Compute attributions
    attributions, component_attrs = compute_attributions(
        interpreter, sequence, top_pred_idx, target_proceeding_idx
    )
    
    if attributions is None:
        return None
    
    # Create result dictionary
    result = {
        'case_info': case_info,
        'sequence_length': len(sequence),
        'target_proceeding_idx': target_proceeding_idx,
        'true_labels': true_labels,
        'predicted_label': top_pred_label,
        'prediction_prob': top_pred_prob,
        'attributions_ig': attributions,
        'component_attributions': component_attrs,
    }
    
    # Visualize
    visualize_attributions(interpreter, attributions, component_attrs, top_pred_idx,
                          case_info, sample_idx, interpretability_folder)
    
    # Get and save feature summary
    get_and_save_feature_summary(interpreter, sequence, top_pred_idx, 
                                 target_proceeding_idx, sample_idx, interpretability_folder)
    
    return result

def run_attribution_analysis(interpreter, val_sequences, val_labels, val_case_ids, 
                            trained_model, label_processor, interpretability_folder,
                            court_data, num_samples, original_sequences = None):
    interpretation_results = []
    
    for sample_idx in range(min(num_samples, len(val_sequences))):
        print(f"\n{'='*70}")
        print(f"ANALYZING SAMPLE {sample_idx + 1}/{min(num_samples, len(val_sequences))}")
        print(f"{'='*70}")
        
        sequence = val_sequences[sample_idx]
        labels = val_labels[sample_idx]
        if isinstance(sequence, tuple):
            sequence = sequence[0]   
        
        
        # FIX: Call transform_sequence directly
        labels_binary = label_processor.transform_sequence(labels)
        if original_sequences is not None:
            original_seq = original_sequences[sample_idx]
            if isinstance(original_seq, tuple):
                original_seq = original_seq[0]
        else:
            original_seq = sequence
        case_info = extract_case_info_from_sequence(original_seq, court_data)
        
        result = analyze_single_sample(
            interpreter, sequence, labels, labels_binary, case_info,
            sample_idx, trained_model, label_processor, interpretability_folder
        )
        
        if result is not None:
            interpretation_results.append(result)
    
    return interpretation_results


def aggregate_component_importance(interpretation_results):
    if not interpretation_results:
        return None
    
    print(f"\n{'='*70}")
    print("AGGREGATE INTERPRETABILITY ANALYSIS")
    print(f"{'='*70}")
    
    # Collect all component names
    all_component_names = set()
    for result in interpretation_results:
        all_component_names.update(result['component_attributions'].keys())
    
    # Aggregate importance
    component_importance_aggregate = {comp: [] for comp in all_component_names}
    
    for result in interpretation_results:
        for comp_name in all_component_names:
            if comp_name in result['component_attributions']:
                avg_importance = np.mean(np.abs(result['component_attributions'][comp_name]))
                component_importance_aggregate[comp_name].append(avg_importance)
    
    # Calculate statistics
    print("\nðŸ“Š Average Feature Component Importance Across Samples:")
    print("-" * 60)
    
    component_stats = []
    for comp_name, importances in component_importance_aggregate.items():
        if importances:
            mean_imp = np.mean(importances)
            std_imp = np.std(importances)
            component_stats.append({
                'Component': comp_name,
                'Mean_Importance': mean_imp,
                'Std_Importance': std_imp,
                'Samples': len(importances)
            })
    
    # Create DataFrame and sort
    component_stats_df = pd.DataFrame(component_stats)
    component_stats_df = component_stats_df.sort_values('Mean_Importance', ascending=False)
    
    print(component_stats_df.to_string())
    
    return component_stats_df
def save_and_visualize_aggregate_importance(component_stats_df, interpretability_folder):
    # Save to Excel
    component_stats_df.to_excel(
        interpretability_folder / "aggregate_feature_component_importance.xlsx",
        index=False
    )
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(
        component_stats_df['Component'], 
        component_stats_df['Mean_Importance'],
        yerr=component_stats_df['Std_Importance'],
        capsize=5,
        alpha=0.7,
        color='steelblue'
    )
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Feature Component')
    plt.ylabel('Average Attribution Score')
    plt.title('Average Feature Component Importance Across Analyzed Cases')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        interpretability_folder / "aggregate_feature_component_importance.png", 
        dpi=300, 
        bbox_inches='tight'
    )
    plt.show()
    
    print("\nâœ… Aggregate analysis complete!")

def find_sample_with_label(val_sequences, val_labels, label_processor, label_idx, max_sequences=20):
    import numpy as np
    import torch
    
    for seq_idx in range(min(max_sequences, len(val_sequences))):
        sequence = val_sequences[seq_idx]
        labels = val_labels[seq_idx]
        
        # Call transform_sequence DIRECTLY, not through transform_labels_to_binary
        labels_binary = label_processor.transform_sequence(labels)
        
        # labels_binary is shape (num_steps, num_labels)
        for proc_idx in range(len(labels_binary)):
            step_labels = labels_binary[proc_idx]  # Shape (num_labels,)
            
            # Convert to numpy array and ensure it's 1D
            if isinstance(step_labels, torch.Tensor):
                step_labels = step_labels.cpu().numpy()
            
            step_labels = np.asarray(step_labels).ravel()  # ravel() ensures 1D
            
            # Now safely check the value at label_idx
            if step_labels[label_idx] > 0.5:  # Use > 0.5 to handle both 0/1 and float values
                return seq_idx, proc_idx, sequence, labels_binary
    
    return None, None, None, None
    
def analyze_label_specific_features(interpreter, val_sequences, val_labels, 
                                    label_processor, labels_to_analyze):
    print(f"\n{'='*70}")
    print("LABEL-SPECIFIC FEATURE ANALYSIS")
    print(f"{'='*70}")
    
    available_labels = [label for label in labels_to_analyze 
                       if label in label_processor.label_names]
    print(f"\nðŸ·ï¸ Analyzing feature importance for labels: {available_labels}")
    
    label_feature_importance = {}
    
    for idx, label_name in enumerate(available_labels):  # Analyze first 2 to save time
        label_idx = label_processor.label_names.index(label_name)
        print(f"[{idx+1}/{len(available_labels)}] Analyzing label: {label_name} (index: {label_idx})")
        
        # Find a sample with this label
        seq_idx, proc_idx, sequence, labels_binary = find_sample_with_label(
            val_sequences, val_labels, label_processor, label_idx
        )
        
        if seq_idx is None:
            print(f"   âš ï¸ No samples found with label '{label_name}'")
            continue
        
        print(f"   Found in sequence {seq_idx}, proceeding {proc_idx}")

        teacher_seq, _ = build_teacher_forced_sequences(
             sequence, 
             labels_binary, 
             label_processor, 
             include_initial_state=True
        )
        
        # Compute attributions
        try:
            _, component_attrs = interpreter.attribute_sequence_features(
                seq=teacher_seq,
                target_class_idx=label_idx,
                target_proceeding_idx=proc_idx,
                method="IntegratedGradients",
                n_steps=30,
                return_component_attributions=True
            )
            
            # Store average importance
            label_feature_importance[label_name] = {
                comp: np.mean(np.abs(attrs)) 
                for comp, attrs in component_attrs.items()
            }
            
            print(f"   âœ… Attribution computed successfully")
            
        except Exception as e:
            print(f"   âš ï¸ Attribution failed: {e}")
            continue
    
    return label_feature_importance

def create_label_feature_comparison(label_feature_importance, interpretability_folder):
    if not label_feature_importance:
        return
    
    print(f"\nðŸ“Š Comparing Feature Importance Across Labels:")
    print("-" * 60)
    
    # Create comparison DataFrame
    comparison_data = []
    all_components = set()
    for label_features in label_feature_importance.values():
        all_components.update(label_features.keys())
    
    for comp in all_components:
        row = {'Component': comp}
        for label, features in label_feature_importance.items():
            row[label] = features.get(comp, 0)
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.set_index('Component')
    
    plot_df = comparison_df.T
    
    # 2. Now, standardize this new DataFrame by row (axis=1 means "per-row")
    # This standardizes each LABEL's row independently
    plot_df_std = plot_df.sub(plot_df.mean(axis=1), axis=0).div(plot_df.std(axis=1).replace(0, 1), axis=0)

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        plot_df_std,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Attribution Score (standardized by Label)'}
    )
    plt.title('Feature Component Importance by Label')
    plt.xlabel('Feature Component')
    plt.ylabel('Label')
    plt.tight_layout()
    plt.savefig(
        interpretability_folder / "interpretability_label_feature_heatmap.png", 
        dpi=300, 
        bbox_inches='tight'
    )
    plt.show()
    
    # Save comparison to Excel
    comparison_df.to_excel(
        interpretability_folder / "interpretability_label_feature_comparison.xlsx"
    )
    plot_df_std.to_excel(
        interpretability_folder / "interpretability_label_feature_comparison_standardized.xlsx"
    )
    
    print("\nâœ… Label-specific analysis complete!")

def run_interpretability_analysis(trained_model, label_processor, device, val_sequences,
                                  val_labels, val_case_ids, court_data, model_configuration,
                                  cache_manager, num_samples=3):
    # Prepare sequences with prior labels for interpretability
    print("Preparing sequences with prior labels for interpretability...")
    prepared_val_sequences = []
    for seq, labels_raw in zip(val_sequences, val_labels):
        prepared_seq = prepare_sequence_for_interpretability(seq, labels_raw, label_processor)
        prepared_val_sequences.append(prepared_seq)
    print(f"  Prepared {len(prepared_val_sequences)} sequences")

    # Initialize analyzer
    interpreter = initialize_interpretability_analyzer(trained_model, label_processor, device)
    
    # Get output folder
    interpretability_folder = cache_manager.get_cache_dir('interpretability_results')
    
    # Run attribution analysis
    interpretation_results = run_attribution_analysis(
        interpreter, prepared_val_sequences, val_labels, val_case_ids,
        trained_model, label_processor, interpretability_folder,
        court_data, num_samples, original_sequences=val_sequences
    )
    
    # Aggregate analysis
    if interpretation_results:
        component_stats_df = aggregate_component_importance(interpretation_results)
        save_and_visualize_aggregate_importance(component_stats_df, interpretability_folder)
    
    labels_to_analyze = model_configuration.LABEL_NAMES
    print(f"\n{'='*80}")
    print(f"ANALYZING ALL {len(labels_to_analyze)} LABELS (not just first 5)")
    print(f"{'='*80}")
    label_feature_importance = analyze_label_specific_features(
        interpreter, val_sequences, val_labels, label_processor, labels_to_analyze
    )
    
    create_label_feature_comparison(label_feature_importance, interpretability_folder)
    
    return {
        'interpreter': interpreter,
        'interpretation_results': interpretation_results,
        'label_feature_importance': label_feature_importance
    }

def run_confident_misclassification(trained_model, label_processor, computation_device,
                                   val_data, cache_manager, threshold=0.8, top_k=5, top_n=25, 
                                   max_sequences=100):
    print(f"\n{'='*80}")
    print("CONFIDENT MISCLASSIFICATION ANALYSIS!")
    print("="*80)
    
    results = run_confident_misclassification_analysis(
        model=trained_model,
        label_processor=label_processor,
        device=computation_device,
        dataloader=val_data,
        threshold=threshold,
        top_k=top_k,
        top_n_per_label=top_n,
        max_sequences=max_sequences,
        save_dir=cache_manager.get_cache_dir('confident_misclassification'),
        verbose=True,
    )
    
    print("âœ… Confident misclassification analysis complete")
    
    return results

def analyze_confident_misclass_for_label(
    confident_misclass_results,
    target_label: str = "charges_dropped",
    top_n: int = 5):
    print(f"OUTLIER ANALYSIS (per-label): '{target_label}'")

    mistakes_df = confident_misclass_results.get("mistakes_report", None)

    if mistakes_df is None or mistakes_df.empty:
        print("No mistakes data available for analysis.")
        return None

    # filter to THIS label only
    label_mistakes = mistakes_df[mistakes_df["wrong_label"] == target_label].copy()

    if label_mistakes.empty:
        print(f"No confident misclassifications found for label '{target_label}'.")
        return None

    conf_col = "wrong_label_confidence"

    print(f"Model confidently predicted '{target_label}' incorrectly: {len(label_mistakes)} times")
    print(f"Average confidence: {label_mistakes[conf_col].mean():.3f}")
    print(f"Std confidence: {label_mistakes[conf_col].std():.3f}")
    print(f"Range: {label_mistakes[conf_col].min():.3f} - {label_mistakes[conf_col].max():.3f}")

    all_missed_labels = []
    all_correctly_predicted_labels = []
    if 'true_labels' not in label_mistakes.columns or 'predicted_labels' not in label_mistakes.columns:
        print("Cannot provide context: 'true_labels' or 'predicted_labels' column missing.")
    else:
        for _, row in label_mistakes.iterrows():
            true_set = set(row["true_labels"])
            predicted_set = set(row["predicted_labels"])
            
            # Find what was on the answer key but *not* in the model's top-k predictions
            missed = true_set - predicted_set
            # Find what was on the answer key *and* in the model's top-k predictions
            correctly_predicted = true_set.intersection(predicted_set)
            
            all_missed_labels.extend(list(missed))
            all_correctly_predicted_labels.extend(list(correctly_predicted))
            
        missed_label_counts = Counter(all_missed_labels)
        correct_label_counts = Counter(all_correctly_predicted_labels)

        print(f"Context for the {len(label_mistakes)} '{target_label}' mistake(s):")

        if not missed_label_counts:
            print("No other labels were missed in these steps.")
        else:
            print(f"Labels the model MISSED in these steps (False Negatives):")
            for label, count in missed_label_counts.most_common(10):
                pct = (count / len(label_mistakes)) * 100
                print(f"    {label}: {count} times ({pct:.1f}%)")
        if not correct_label_counts:
            print("No other labels were correctly predicted in these steps.")
        else:
            print(f"Other labels the model CORRECTLY predicted in these steps (True Positives):")
            for label, count in correct_label_counts.most_common(10):
                pct = (count / len(label_mistakes)) * 100
                print(f"    {label}: {count} times ({pct:.1f}%)")
    

    # show top N highest-confidence mistakes
    label_mistakes = label_mistakes.sort_values(conf_col, ascending=False)
    top_mistakes = label_mistakes.head(top_n)

    print(f"\nTop {top_n} most confident '{target_label}' mistakes:")
    cols_to_show = [c for c in ["case_id", "proceeding_id", "step_idx", "true_labels", conf_col]
                    if c in top_mistakes.columns]
    print(top_mistakes[cols_to_show].to_string(index=False))

    # optional: group by case to see repeat offenders
    if "case_id" in label_mistakes.columns:
        cases_with_label_errors = (
            label_mistakes.groupby("case_id")
            .size()
            .sort_values(ascending=False)
        )
        if len(cases_with_label_errors) > 0:
            print(f"\nCases with most '{target_label}' mistakes:")
            for case_id, mistake_count in cases_with_label_errors.head(5).items():
                print(f"  Case {case_id}: {mistake_count} mistakes")

    # quick plots (purely for inspection)
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # histogram of confidences
    ax = axes[0]
    ax.hist(label_mistakes[conf_col], bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(label_mistakes[conf_col].mean(), color="red", linestyle="--",
               linewidth=2, label=f"Mean: {label_mistakes[conf_col].mean():.3f}")
    ax.set_xlabel("Confidence score")
    ax.set_ylabel("Count")
    ax.set_title(f"Confidence for wrong '{target_label}'")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # bar of missed true labels
    ax = axes[1]
    top_missed = missed_label_counts.most_common(10)
    if top_missed:
        labels_list, counts_list = zip(*top_missed)
        y_pos = np.arange(len(labels_list))
        ax.barh(y_pos, counts_list, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_list)
        ax.set_xlabel("Frequency")
        ax.set_title(f"Labels missed when '{target_label}' was wrongly predicted")
        ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()

    return label_mistakes

            

def count_processed_steps(sequence, labels_binary):
    processed_steps = 0
    
    for i in range(-1, len(sequence) - 1):
        if i == -1:
            target_idx = 0
        else:
            target_idx = i + 1
        
        true_binary = labels_binary[target_idx]
        k = int(true_binary.sum())
        if k > 0:
            processed_steps += 1
    
    return processed_steps

def predict_step(trained_model, sequence, true_binary, label_processor):    
    true_labels = collect_labels_from_binary(true_binary, label_processor)
    k = int(true_binary.sum())
    
    if k == 0:
        return None
    
    with torch.no_grad():
        logits = trained_model(sequence)
        probs = torch.sigmoid(logits).squeeze(0).cpu()
        
        # If probs is 2D (timesteps x labels), take the last timestep
        if probs.dim() > 1:
            probs = probs[-1]
    
    pred_binary = predict_with_top_k_mlb(probs, k, label_processor)
    
    pred_labels = collect_labels_from_binary(pred_binary, label_processor)
    
    return pred_labels, true_labels, k


def print_step_prediction_analysis(step_desc, true_labels, pred_labels):
    print(f"\n  {step_desc}:")
    print(f"    True labels:      {list(true_labels) if true_labels else 'None'}")
    print(f"    Predicted labels: {list(pred_labels) if pred_labels else 'None'}")
    
    # Calculate confusion
    true_set = set(true_labels) if true_labels else set()
    pred_set = set(pred_labels) if pred_labels else set()
    
    correct = true_set & pred_set
    missed = true_set - pred_set
    false_pos = pred_set - true_set
    
    print(f"    âœ… Correct:        {list(correct) if correct else 'None'}")
    print(f"    âŒ Missed:         {list(missed) if missed else 'None'}")
    print(f"    ðŸ”€ False Positive: {list(false_pos) if false_pos else 'None'}")
    
    # Calculate step error rate
    step_errors = len(missed) + len(false_pos)
    steps_total = len(true_set) + len(false_pos)
    step_error_rate = step_errors / steps_total if steps_total > 0 else 0
    print(f"    âš ï¸  Step Error Rate: {step_error_rate:.1%}")

def analyze_sample_predictions(trained_model, val_data, label_processor, 
                               validation_metrics, court_data):
    print(f"\n{'='*80}")
    print("SAMPLE PREDICTIONS")
    print("="*80)
    
    case_errors = validation_metrics.get('CASE_ERRORS_SORTED', [])
    error_cases_ordered = [case['case_idx'] for case in case_errors]
    
    for sample_index in error_cases_ordered:
        if sample_index >= len(val_data):
            continue
        
        sequence, labels_raw = val_data[sample_index]
        
        labels_binary = label_processor.transform_sequence(labels_raw)
        
        case_info = extract_case_info_from_sequence(sequence, court_data)
        
        error_info = ""
        if case_errors:
            matching_case = next((case for case in case_errors 
                                if case['case_idx'] == sample_index), None)
            if matching_case:
                error_rate = matching_case.get('overall_error_rate', 0.0)
                rank = next((i+1 for i, case in enumerate(case_errors) 
                           if case['case_idx'] == sample_index), 'N/A')
                error_info = f" | Error Rate: {error_rate:.2f} | Rank: {rank}"
        
        processed_steps = count_processed_steps(sequence, labels_binary)
        
        print(f"\nðŸ” SAMPLE {sample_index + 1} PREDICTION ANALYSIS")
        print(f"ðŸ“‹ Case Info: Mag#{case_info.get('mag_num', 'N/A')} | "
              f"CDC#{case_info.get('case_id', 'N/A')} | "
              f"Def#{case_info.get('defendant_id', 'N/A')} {error_info}")
        print("-" * 60)
        print(f"Sequence length: {len(sequence)} proceedings ({processed_steps} processed steps)")
        
        teacher_seq, _ = build_teacher_forced_sequences(sequence, labels_binary, label_processor)
        with torch.no_grad():
            logits = trained_model(teacher_seq)
            all_probs = torch.sigmoid(logits).squeeze(0).cpu()
        
        for i in range(len(labels_binary)):
            if i == 0:
                step_desc = "Initial → Step 0"
            else:
                step_desc = f"Step {i - 1} → Step {i}"
            true_binary = labels_binary[i]
            k = int(true_binary.sum())
            if k == 0:
                continue
            probs = all_probs[i]
            pred_binary = predict_with_top_k_mlb(probs, k, label_processor)

            pred_labels = collect_labels_from_binary(pred_binary, label_processor)
            true_labels = collect_labels_from_binary(true_binary, label_processor)
            print_step_prediction_analysis(step_desc, true_labels, pred_labels) 

def create_standard_ablation_configurations():
    #start with an empty dict
    ablation_configs = {}
    
    #define the standard configurations
    base_config = {
        'proceeding_text': False,
        'previous_cases': False,
        'defendant_demographics': False,
        'charges': False,
        'magistrate': False,
        'bond': False,
        'temporal': False,
        'prior_labels': False
    }
    
    # No ablation
    ablation_configs['no_ablation'] = deepcopy(base_config)
    
    # Single feature ablations
    ablation_configs['ablate_proceeding_text'] = deepcopy(base_config)
    ablation_configs['ablate_proceeding_text']['proceeding_text'] = True
    
    # ablation_configs['ablate_previous_cases'] = deepcopy(base_config)
    # ablation_configs['ablate_previous_cases']['previous_cases'] = True
    
    ablation_configs['ablate_defendant_demographics'] = deepcopy(base_config)
    ablation_configs['ablate_defendant_demographics']['defendant_demographics'] = True

    ablation_configs['ablate_prior_labels'] = deepcopy(base_config)
    ablation_configs['ablate_prior_labels']['prior_labels'] = True

    ablation_configs['ablate_proceeding_text_and_prior_labels'] = deepcopy(base_config)
    ablation_configs['ablate_proceeding_text_and_prior_labels']['proceeding_text'] = True
    ablation_configs['ablate_proceeding_text_and_prior_labels']['prior_labels'] = True
    
    # ablation_configs['ablate_charges'] = deepcopy(base_config)
    # ablation_configs['ablate_charges']['charges'] = True
    
    # ablation_configs['ablate_magistrate'] = deepcopy(base_config)
    # ablation_configs['ablate_magistrate']['magistrate'] = True
    
    # ablation_configs['ablate_bond'] = deepcopy(base_config)
    # ablation_configs['ablate_bond']['bond'] = True
    
    # ablation_configs['ablate_temporal'] = deepcopy(base_config)
    # ablation_configs['ablate_temporal']['temporal'] = True
    
    #multi feature ablations
    # ablation_configs['ablate_text_and_demographics'] = deepcopy(base_config)  # Fixed typo
    # ablation_configs['ablate_text_and_demographics']['proceeding_text'] = True
    # ablation_configs['ablate_text_and_demographics']['defendant_demographics'] = True
    
    # ablation_configs['ablate_text_and_previous_cases'] = deepcopy(base_config)
    # ablation_configs['ablate_text_and_previous_cases']['proceeding_text'] = True
    # ablation_configs['ablate_text_and_previous_cases']['previous_cases'] = True
    
    # ablation_configs['ablate_all_but_text'] = deepcopy(base_config)
    # ablation_configs['ablate_all_but_text']['previous_cases'] = True
    # ablation_configs['ablate_all_but_text']['defendant_demographics'] = True
    # ablation_configs['ablate_all_but_text']['charges'] = True
    # ablation_configs['ablate_all_but_text']['magistrate'] = True
    # ablation_configs['ablate_all_but_text']['bond'] = True
    # ablation_configs['ablate_all_but_text']['temporal'] = True
    
    # ablation_configs['ablate_all_but_text_and_charges'] = deepcopy(base_config)
    # ablation_configs['ablate_all_but_text_and_charges']['previous_cases'] = True
    # ablation_configs['ablate_all_but_text_and_charges']['defendant_demographics'] = True
    # ablation_configs['ablate_all_but_text_and_charges']['magistrate'] = True
    # ablation_configs['ablate_all_but_text_and_charges']['bond'] = True
    # ablation_configs['ablate_all_but_text_and_charges']['temporal'] = True
    
    # ablation_configs['ablate_case_context'] = deepcopy(base_config)
    # ablation_configs['ablate_case_context']['previous_cases'] = True
    # ablation_configs['ablate_case_context']['magistrate'] = True
    # ablation_configs['ablate_case_context']['bond'] = True
    
    # ablation_configs['ablate_all_structured'] = deepcopy(base_config)
    # ablation_configs['ablate_all_structured']['defendant_demographics'] = True
    # ablation_configs['ablate_all_structured']['charges'] = True
    # ablation_configs['ablate_all_structured']['magistrate'] = True
    # ablation_configs['ablate_all_structured']['bond'] = True
    # ablation_configs['ablate_all_structured']['temporal'] = True
    
    # ablation_configs['ablate_all_contextual'] = deepcopy(base_config)
    # ablation_configs['ablate_all_contextual']['proceeding_text'] = True
    # ablation_configs['ablate_all_contextual']['previous_cases'] = True
    # ablation_configs['ablate_all_contextual']['temporal'] = True
    
    return ablation_configs


def run_ablation_study(
    base_model,
    train_data,
    val_data,
    label_processor,
    device,
    ablation_configs,
    output_dir,
    epochs=5,
    batch_size=32,
    use_mixed_precision=False,
    use_learning_rate_scheduler=False,
    gradient_clip_norm=1.0,
    weighted_loss=False,
    fold_idx=None,
    cache_manager=None):
    return run_ablation_study_function(
        base_model=base_model,
        train_data=train_data,
        val_data=val_data,
        label_processor=label_processor,
        device=device,
        ablation_configs=ablation_configs,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        use_mixed_precision = use_mixed_precision,
        use_learning_rate_scheduler = use_learning_rate_scheduler,
        gradient_clip_norm=gradient_clip_norm,
        weighted_loss=weighted_loss,
        fold_idx=fold_idx,
        cache_manager=cache_manager
    )


def compare_ablation_table(results):
    return compare_ablation_studies_dataframe(results)

class _SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def prepare_sequence_with_binary_labels(seq, labels_binary, num_labels):
    prepared_seq = []
    for i, proc in enumerate(seq):
        proc_copy = dict(proc)
        if i == 0:
            proc_copy['true_prior_labels'] = [0] * num_labels
        else:
            prev_labels = labels_binary[i - 1]
            if hasattr(prev_labels, 'tolist'):
                proc_copy['true_prior_labels'] = prev_labels.tolist()
            else:
                proc_copy['true_prior_labels'] = list(prev_labels)
        prepared_seq.append(proc_copy)
    return prepared_seq

def compute_global_attributions_table(interpreter,
                                      val_sequences_and_labels,
                                      max_samples,
                                      show_top_n_features,
                                      ig_n_steps,
                                      optional_tag=None,
                                      interpretability_folder=None,
                                      verbose=True):
    # Prepare sequences with prior labels for interpretability
    num_labels = len(interpreter.label_processor.label_names)
    prepared_data = []
    for seq, labels_binary in val_sequences_and_labels:
        if seq and 'true_prior_labels' not in seq[0]:
            prepared_seq = prepare_sequence_with_binary_labels(seq, labels_binary, num_labels)
            prepared_data.append((prepared_seq, labels_binary))
        else:
            prepared_data.append((seq, labels_binary))
    
    loader = DataLoader(
        _SeqDataset(prepared_data),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: b[0]
    )
    full_attributions_df = interpreter.attributions_table(
        dataloader=loader,
        max_samples=max_samples,
        show_top_n_features=show_top_n_features,
        ig_n_steps=ig_n_steps,
        verbose=verbose
    )

    seen_classes = [col for col in full_attributions_df.columns
                    if not col.endswith(('_zscore_within_label', '_zscore_across_labels'))
                    and col not in ['Mean_Raw_Attribution', 'Mean_ZScore_Within_Label', 'Mean_ZScore_Across_Labels']]

    # Table 1: Raw attributions
    print()
    print("Table 1: Raw Attribution Magnitudes")
    raw_attributions_df = full_attributions_df[seen_classes + ['Mean_Raw_Attribution']].copy()
    raw_attributions_df = raw_attributions_df.sort_values(
        by='Mean_Raw_Attribution', ascending=False
    ).head(show_top_n_features)
    print(raw_attributions_df.to_string())
    
    save_file_raw = 'attribution_heatmap_raw.png' if optional_tag is None else f'attribution_heatmap_raw_{optional_tag}.png'
    save_path_raw = interpretability_folder / save_file_raw if interpretability_folder else save_file_raw

    interpreter.plot_feature_importance(
        attributions_df=raw_attributions_df,
        view_type = 'raw',
        save_path=save_path_raw,
        title = 'Raw Attribution Magnitudes By Label'
    )

    print("\nINSIGHTS: Raw Attribution Magnitudes\n")
    top_3_overall_for_raw = raw_attributions_df.nlargest(3, 'Mean_Raw_Attribution')
    print("Top 3 Features Overall (Raw Attributions):")
    for idx, (feature, row) in enumerate(top_3_overall_for_raw.iterrows(), start=1):
        print(f"  {idx}. {feature} : {row['Mean_Raw_Attribution']:.4f} (Average Across Labels)")
    print("\nPer Label Leaders (Raw Attributions):")
    for label_col in seen_classes:
        top_feature = raw_attributions_df[label_col].idxmax()
        top_value = raw_attributions_df.loc[top_feature, label_col]
        print(f"  Label '{label_col}': Top Feature: {top_feature} with Raw Attribution: {top_value:.4f}")

    # Table 2: Within Label Z-Scores
    print("\nWithin-Label Feature Rankings")
    within_label_cols = [col for col in full_attributions_df.columns if col.endswith('_zscore_within_label')]
    within_label_table = full_attributions_df[within_label_cols + ['Mean_ZScore_Within_Label']].copy()
    within_label_table = within_label_table.sort_values(
        by='Mean_ZScore_Within_Label', ascending=False
    ).head(show_top_n_features)
    print(within_label_table.to_string())

    save_file_within = 'attribution_heatmap_within_label.png' if optional_tag is None else f'attribution_heatmap_within_label_{optional_tag}.png'
    save_path_within = interpretability_folder / save_file_within if interpretability_folder else save_file_within

    interpreter.plot_feature_importance(
        attributions_df=within_label_table,
        view_type = 'within_label',
        save_path=save_path_within,
        title = 'Feature Rankings Within Each Label (Z-Scores)'
    )

    print("\nINSIGHTS: Within-Label Feature Rankings\n")
    print("Positive Z-Score would indicate above-average importance for that label")
    print("Negative Z-Score would indicate below-average importance for that label\n")

    for label_col in within_label_cols:
        label_name = label_col.replace('_zscore_within_label', '')
        label_data = within_label_table[label_col].dropna()
        if len(label_data) > 0:
            top_3_features = label_data.nlargest(3)
            print(f"  Label '{label_name}':")
            print("    Top 3 Features (relative to other features for this label):")
            for idx, (feature, score) in enumerate(top_3_features.items(), start=1):
                print(f"      {idx}. {feature} : Z-Score: {score:.4f}")

    # Table 3: Across Label Z-Scores
    print("\nAcross-Label Feature Specialization")
    across_label_cols = [col for col in full_attributions_df.columns if col.endswith('_zscore_across_labels')]
    across_label_table = full_attributions_df[across_label_cols + ['Mean_ZScore_Across_Labels']].copy()
    across_label_table = across_label_table.sort_values(
        by='Mean_ZScore_Across_Labels', ascending=False
    ).head(show_top_n_features)
    print(across_label_table.to_string())

    save_file_across = 'attribution_heatmap_across_labels.png' if optional_tag is None else f'attribution_heatmap_across_labels_{optional_tag}.png'
    save_path_across = interpretability_folder / save_file_across if interpretability_folder else save_file_across

    interpreter.plot_feature_importance(
        attributions_df=across_label_table,
        view_type = 'across_labels',
        save_path=save_path_across,
        title = 'Feature Specialization Across Labels (Z-Scores)'
    )
    print("\nINSIGHTS: Across-Label Feature Specialization\n")
    print("A positive Z-score indicates the label relies more heavily\non this feature than the average reliance across all labels.")
    print("A negative Z-score indicates the label relies less\non this feature than the average reliance across all labels.\n")

    for feature in across_label_table.index:
        feature_row = across_label_table.loc[feature, across_label_cols]
        max_label_col = feature_row.idxmax()
        max_label = max_label_col.replace('_zscore_across_labels', '')
        max_zscore = feature_row[max_label_col]

        min_label_col = feature_row.idxmin()
        min_label = min_label_col.replace('_zscore_across_labels', '')
        min_zscore = feature_row[min_label_col]

        print(f"  Feature '{feature}':")
        print(f"    Most Specialized For Label '{max_label}' with Z-Score: {max_zscore:.4f}")
        print(f"    Least Specialized For Label '{min_label}' with Z-Score: {min_zscore:.4f}")
    
    print("\n Most specialized features (high variance in across-label Z-Scores):")
    feature_variances = {}
    for feature in across_label_table.index:
        feature_row = across_label_table.loc[feature, across_label_cols]
        feature_variances[feature] = feature_row.std()
    
    top_variance = sorted(feature_variances.items(), key=lambda x: x[1], reverse=True)[:5]
    for feature, var in top_variance:
        print(f"  Feature '{feature}': Std Dev of Across-Label Z-Scores: {var:.4f}")
    
    print("\n Least specialized features (low variance in across-label Z-Scores):")
    bottom_variance = sorted(feature_variances.items(), key=lambda x: x[1])[:5]
    for feature, var in bottom_variance:
        print(f"  Feature '{feature}': Std Dev of Across-Label Z-Scores: {var:.4f}")
    print("\nGlobal attribution analysis complete\n")

    return {
        'full': full_attributions_df,
        'raw': raw_attributions_df,
        'within_label': within_label_table,
        'across_label': across_label_table
    }

def prepare_sequence_for_interpretability(seq, labels_raw, labels_processor):
    labels_binary = labels_processor.transform_sequence(labels_raw)

    teacher_seq, targets = build_teacher_forced_sequences(
        seq=seq,
        labels_seq_binary=labels_binary,
        label_processor=labels_processor,
        include_initial_state=True
    )
    return teacher_seq, targets

def apply_skipped_duplicates_on_imported_sequences(sequences, label_sequences_raw, return_indices = False, verbose=True):
    if len(sequences) != len(label_sequences_raw):
        raise ValueError("Length of sequences and label_sequences_raw must be the same.")
    
    total_skipped = 0
    total_proceedings = 0
    sequences_rejected_no_labels = 0
    sequences_rejected_too_short = 0
    
    filtered_sequences = []
    filtered_label_sequences_raw = []
    if return_indices:
        kept_indices = []

    for seq_idx, (seq, labels_seq) in enumerate(zip(sequences, label_sequences_raw)):
        if not seq or not labels_seq:
            continue

        filtered_seq = []
        filtered_lab = []

        previous_labels_set = None

        for proceeding, step_labels in zip(seq, labels_seq):
            total_proceedings += 1
            current_labels_set = set(step_labels)

            if previous_labels_set is not None and current_labels_set == previous_labels_set and current_labels_set:
                total_skipped += 1
                continue

            filtered_seq.append(proceeding)
            filtered_lab.append(step_labels)
            previous_labels_set = current_labels_set
        
        if len(filtered_seq) >= 2 and len(filtered_lab) >= 2:
            has_meaningful_next_labels = any(
                len(filtered_lab[i+1]) > 0  # Check if list of labels is not empty
                for i in range(len(filtered_lab) - 1)
            )
            if has_meaningful_next_labels:
                filtered_sequences.append(filtered_seq)
                filtered_label_sequences_raw.append(filtered_lab)
                if return_indices:
                    kept_indices.append(seq_idx)
            else:
                sequences_rejected_no_labels += 1
        else:
            sequences_rejected_too_short += 1
    if verbose:
        skip_percent = (total_skipped / total_proceedings * 100) if total_proceedings > 0 else 0
        print("Applied skipped duplicates filter:")
        print(f"  Total proceedings processed: {total_proceedings}")
        print(f"  Total proceedings skipped as duplicates: {total_skipped}")
        print(f"  Percentage of proceedings skipped: {skip_percent:.2f}%")
        print(f"  Sequences before: {len(sequences):,}")
        print(f"  Sequences after: {len(filtered_sequences):,}")
        print(f"  Rejected (too short): {sequences_rejected_too_short:,}")
        print(f"  Rejected (no meaningful labels): {sequences_rejected_no_labels:,}")
    if return_indices:
        return filtered_sequences, filtered_label_sequences_raw, kept_indices
    else:
        return filtered_sequences, filtered_label_sequences_raw