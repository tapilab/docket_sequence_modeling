
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
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
import contextlib
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from transformers import AutoModel, AutoTokenizer
import gc
import h5py
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
import time
import shutil
import xlsxwriter
from scipy.sparse import csr_matrix
from captum.attr import IntegratedGradients, LayerConductance, LayerActivation
from captum.attr import visualization as viz
from notebook_helpers import extract_case_info_from_sequence
from config import get_model_config, COMPLETION_LABELS
from data.load_data import mount_drive, load_data
from data.process_data import filter_proceedings, remove_multi_defendant_cases, create_combined_dockets, filter_to_magistrate_started_cases, remove_extradition_cases
from data.sequence_builder import create_enhanced_training_sequences_cached, get_proceeding_sequence_for_defendant
from regex_labels.label_generator_and_cache_manager import generate_and_cache_regex_labels
from models.transformer_creation import create_model_from_data
from training.trainer import train_case_transformer_with_mlb
from training.label_processor import EnhancedLabelProcessor
from training.evaluation import evaluate_with_teacher_forcing
from analysis.cooccurrence import export_cooccurrence_from_valid_completed_cases, create_cooccurrence_matrix_excel
from analysis.frequency_analysis import analyze_raw_label_frequencies_combined, analyze_raw_label_frequencies_singular, print_label_distribution_by_court
from analysis.diagnostic import comprehensive_training_diagnostic
from models.baseline_model import run_baseline
from utils.cache_directory_utils import CacheDirectoryManager
from utils.model_memory_utils import analyze_model_memory, debug_oom_sample
from models.embeddings import extract_and_cache_bert_embeddings_streaming
from analysis.byclerk import analyze_clerk_data

def get_case_lengths(cases):
    sequences_length = [len(seq) for seq, _ in cases]
    return sequences_length

def sample_from_the_longest_cases(cases, number_of_longest_cases, sample_size):
    sequences_length = get_case_lengths(cases)
    sorted_indices = np.argsort(sequences_length)[::-1]
    number_of_longest_cases = min(number_of_longest_cases, len(cases))
    longest_indices = sorted_indices[:number_of_longest_cases]
    longest_cases = [cases[i] for i in longest_indices]
    longest_case_lengths = [sequences_length[i] for i in longest_indices]

    sample_size = min(sample_size, len(longest_cases))
    sampled_indices = random.sample(range(len(longest_cases)), sample_size)
    sampled_cases = [longest_cases[i] for i in sampled_indices]
    sampled_case_lengths = [longest_case_lengths[i] for i in sampled_indices]
    return sampled_cases, sampled_case_lengths, sampled_indices

def export_long_cases_to_xlsx(sampled_cases, output_filepath, court_data=None):
    output_data = []
    for seq_idx, (sequence, label_sequence) in enumerate(tqdm(sampled_cases, desc="Processing specified cases")):
        if not sequence:
            continue
            
        case_info = extract_case_info_from_sequence(sequence, court_data=court_data)
        mag_num = case_info.get('mag_num', '')
        case_id = case_info.get('case_id', '')
        defendant_id = case_info.get('defendant_id', '')
        
        for proc_idx, (proc, labels) in enumerate(zip(sequence, label_sequence)):
            proc_id = proc.get('proceeding_id')
            
            # Get charge information
            charges_dict = proc.get('charges', {})
            charge_names = [charge_info.get('name', '') for charge_info in charges_dict.values()]
            charge_names_str = ' | '.join(charge_names) if charge_names else ''
            
            # Convert labels list to string
            if isinstance(labels, list):
                labels_str = '|'.join(labels) if labels else 'NO_LABELS'
                label_count = len(labels)
            else:
                labels_str = 'NO_LABELS'
                label_count = 0
            output_data.append({
                'sequence_index': seq_idx,
                'proceeding_index': proc_idx,
                'mag_num': mag_num if pd.notna(mag_num) else '',
                'case_id': case_id if pd.notna(case_id) else '',
                'defendant_id': defendant_id if pd.notna(defendant_id) else '',
                'proceeding_id': proc_id,
                'date': proc.get('date'),
                'date_month_year': proc.get('date_month_year', ''),
                'proceeding_type': proc.get('type'),
                'text': proc.get('text', '') if proc.get('text') else '',
                'docket_id': proc.get('docket_id', ''),
                'original_bond': proc.get('original_bond', ''),
                'sum_of_bonds': proc.get('sum_of_bonds', ''),
                'race_id': proc.get('race_id', ''),
                'gender_id': proc.get('gender_id', ''),
                'age': proc.get('age', ''),
                'magistrate_section': proc.get('magistrate_section', ''),
                'charges': charge_names_str,
                'num_charges': len(charges_dict),
                'labels': labels_str,
                'label_count': label_count,
                'sequence_length': len(sequence)
            })
    output_df = pd.DataFrame(output_data)
    output_df = output_df.sort_values(['defendant_id', 'date']).reset_index(drop=True)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_excel(output_filepath, index=False)

def sample_uniformly(cases, sample_size):
    number_of_cases = len(cases)
    if number_of_cases == 0:
        raise ValueError("No cases available for sampling.")
    
    if sample_size >= number_of_cases:
        sampled_indices = list(range(number_of_cases))
    elif sample_size <= 0:
        raise ValueError("Sample size must be a positive integer.")
    else:
        sampled_indices = random.sample(range(number_of_cases), sample_size)
    
    sampled_cases = [cases[i] for i in sampled_indices]
    sampled_case_lengths = [len(cases[i][0]) for i in sampled_indices]
    return sampled_cases, sampled_case_lengths, sampled_indices

def export_uniform_cases_to_xlsx(sampled_cases, output_filepath, court_data=None):
    output_data = []
    for seq_idx, (sequence, label_sequence) in enumerate(tqdm(sampled_cases, desc="Processing specified cases")):
        if not sequence:
            continue
            
        case_info = extract_case_info_from_sequence(sequence, court_data=court_data)
        mag_num = case_info.get('mag_num', '')
        case_id = case_info.get('case_id', '')
        defendant_id = case_info.get('defendant_id', '')
        
        for proc_idx, (proc, labels) in enumerate(zip(sequence, label_sequence)):
            proc_id = proc.get('proceeding_id')
            
            # Get charge information
            charges_dict = proc.get('charges', {})
            charge_names = [charge_info.get('name', '') for charge_info in charges_dict.values()]
            charge_names_str = ' | '.join(charge_names) if charge_names else ''
            
            # Convert labels list to string
            if isinstance(labels, list):
                labels_str = '|'.join(labels) if labels else 'NO_LABELS'
                label_count = len(labels)
            else:
                labels_str = 'NO_LABELS'
                label_count = 0
            output_data.append({
                'sequence_index': seq_idx,
                'proceeding_index': proc_idx,
                'mag_num': mag_num if pd.notna(mag_num) else '',
                'case_id': case_id if pd.notna(case_id) else '',
                'defendant_id': defendant_id if pd.notna(defendant_id) else '',
                'proceeding_id': proc_id,
                'date': proc.get('date'),
                'date_month_year': proc.get('date_month_year', ''),
                'proceeding_type': proc.get('type'),
                'text': proc.get('text', '')[:1000] if proc.get('text') else '',  # First 1000 chars
                'docket_id': proc.get('docket_id', ''),
                'original_bond': proc.get('original_bond', ''),
                'sum_of_bonds': proc.get('sum_of_bonds', ''),
                'race_id': proc.get('race_id', ''),
                'gender_id': proc.get('gender_id', ''),
                'age': proc.get('age', ''),
                'magistrate_section': proc.get('magistrate_section', ''),
                'charges': charge_names_str,
                'num_charges': len(charges_dict),
                'labels': labels_str,
                'label_count': label_count,
                'sequence_length': len(sequence)
            })
    output_df = pd.DataFrame(output_data)
    output_df = output_df.sort_values(['defendant_id', 'date']).reset_index(drop=True)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_excel(output_filepath, index=False)

