# %% [markdown]
# Parameters

# %%
import os
import json
from pathlib import Path

parameters_filename = 'balanced_config_9.json'
PROJECT_ROOT = Path("/home/krosen5/Court Case Sequence Transformer Model/court_case_transformer_project")
parameters_path = PROJECT_ROOT / "parameters_jsons" / parameters_filename
with open(parameters_path, "r", encoding="utf-8") as file:
    parameters_dict = json.load(file)

# %%
try:
    cuda_index = 0
    if cuda_index is None:
        raise NameError
except NameError:
    cuda_index = 0

# %% [markdown]
# # Imports

# %%
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
import time
import shutil
import xlsxwriter
from scipy.sparse import csr_matrix
from captum.attr import IntegratedGradients, LayerConductance, LayerActivation
from captum.attr import visualization as viz
import io
import pickle
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm import tqdm

# %%
# Set device globally
device = torch.device(f"cuda:{cuda_index}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    print(f"CUDA device #: {cuda_index}")
    print(f"GPU: {torch.cuda.get_device_name(cuda_index)}")
    print(f"CUDA Version: {torch.version.cuda}")

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)


# %%
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# %%
set_global_seed(parameters_dict['random_seed'])

# %%
from config import get_model_config, COMPLETION_LABELS
from data.load_data import mount_drive, load_data
from data.process_data import append_prefix_to_proceeding_ids, filter_proceedings, remove_multi_defendant_cases, create_combined_dockets, filter_to_magistrate_started_cases, remove_extradition_cases
from data.sequence_builder import create_enhanced_training_sequences_cached
from regex_labels.label_generator_and_cache_manager import generate_and_cache_regex_labels
from models.transformer_creation import create_model_from_data
from training.trainer import train_case_transformer_with_mlb
from training.label_processor import EnhancedLabelProcessor
from training.evaluation import evaluate_with_teacher_forcing
from analysis.cooccurrence import export_cooccurrence_from_valid_completed_cases, create_cooccurrence_matrix_excel
from analysis.frequency_analysis import print_label_distribution_by_court
from analysis.diagnostic import comprehensive_training_diagnostic
from models.baseline_model import run_baseline, convert_baseline_results_to_table_format
from utils.cache_directory_utils import CacheDirectoryManager
from utils.model_memory_utils import analyze_model_memory, debug_oom_sample
from models.embeddings import extract_and_cache_bert_embeddings_streaming
from analysis.byclerk import analyze_clerk_data
from models.interpretability import CaseTransformerInterpretability, CaptumWrapper
from models.most_confident_wrong_answer_analyzer import run_confident_misclassification_analysis
from data.k_fold_methods import KFoldCVRunner
from data.dataset_creation import sample_from_the_longest_cases, export_long_cases_to_xlsx, sample_uniformly, export_uniform_cases_to_xlsx
from analysis.tablegenerator import TableGenerator
from data.balanced_dataset_creation import create_balanced_dataset, export_balanced_cases_to_xlsx
from models.export_model import export_model_complete
from utils.case_identifier_utils import extract_case_identifier_from_sequences, save_case_identifiers, load_case_identifiers, compare_case_identifiers, load_and_compare_case_identifiers
from models.baseline_model_2 import run_random_baseline_comparison, convert_random_baseline_results_to_table_format
from analysis.compare_transformer_to_baseline_1 import compare_transformer_vs_baseline1_on_completion_labels, convert_baseline1_comparison_to_table_generator_format
# ═══════════════════════════════════════════════════════════════
# Embedding Clustering Analysis (PCA + KMeans on pre-classifier vectors)
# ═══════════════════════════════════════════════════════════════

from analysis.pca_vectors import (run_embedding_clusters_pipeline, run_embedding_clusters_kfolds_pipeline)
from label_tracking_methods import count_specific_labels_in_cache, count_labels_in_sequences, print_tracking_checkpoint, TRACKING_LABELS

from data.export_all_data import export_all_cases_to_xlsx

# %%
%load_ext autoreload
%autoreload 2
from notebook_helpers import (
    # Data filtering helpers
    filter_to_completed_cases,
    build_case_counts_df,
    filter_zero_label_proceedings,
    generate_cooccurrence_matrix_excel,
    filter_proceedings_to_end_of_case,
    apply_skipped_duplicates_on_imported_sequences,
    
    # Data transformation helpers
    transform_labels_to_binary,
    convert_data_to_format,
    
    # Sequence processing helpers
    filter_sequences_with_meaningful_labels,
    split_train_validation,
    prepare_sequences_and_labels,
    
    # Case info extraction
    extract_case_info_from_sequence,
    
    # Prediction helpers
    predict_case_labels,
    predict_step,
    
    # Metrics calculation helpers
    calculate_metrics,
    calculate_case_metrics,
    collect_labels_from_binary,
    count_processed_steps,
    
    # Analysis helpers
    perform_frequency_analysis,
    debug_model_sample,
    
    # Training helpers
    train_model,
    
    # Evaluation helpers
    prepare_validation_data_for_eval,
    evaluate_model,
    print_micro_averaged_metrics,
    print_top_performing_labels,
    
    # Case-level evaluation helpers
    evaluate_case_level,
    print_case_level_metrics,
    save_case_level_results,
    print_best_worst_cases,
    print_f1_distribution,
    calculate_label_metrics_from_cases,
    save_label_metrics_to_excel,
    
    # Advanced analysis helpers
    run_diagnostic_analysis,
    run_baseline_comparison,
    run_confident_misclassification,
    
    # Interpretability helpers
    initialize_interpretability_analyzer,
    select_samples_for_analysis,
    get_model_prediction,
    compute_attributions,
    visualize_attributions,
    get_and_save_feature_summary,
    analyze_single_sample,
    run_attribution_analysis,
    aggregate_component_importance,
    save_and_visualize_aggregate_importance,
    find_sample_with_label,
    analyze_label_specific_features,
    create_label_feature_comparison,
    run_interpretability_analysis,
    analyze_confident_misclass_for_label,
    compute_global_attributions_table,
    
    # Sample prediction helpers
    print_step_prediction_analysis,
    analyze_sample_predictions,

    # ablation study helpers
    create_standard_ablation_configurations,
    run_ablation_study,
    compare_ablation_table
)

# %%
folder = rf'/data/krosen5/cache/transformers/{parameters_dict["folder_name_directory"]}/'
if not os.path.exists(folder):
    os.makedirs(folder)
model_cache_manager = CacheDirectoryManager(base_path=folder)

# %%
if parameters_dict['sequence_data_mode'] == 'import' and parameters_dict.get('sequence_data_base_directory'):
    base_folder = rf'/data/krosen5/cache/transformers/{parameters_dict["sequence_data_base_directory"]}/'
    import_cache_manager = CacheDirectoryManager(base_path=base_folder)
elif parameters_dict['sequence_data_mode'] == 'import':
    import_cache_manager = model_cache_manager

# %%
if parameters_dict['regex_labels_mode'] == 'load_in' and parameters_dict.get('regex_labels_load_in_folder'):
    regex_folder = rf'/data/krosen5/cache/transformers/{parameters_dict["regex_labels_load_in_folder"]}/'
    #make it into a path
    regex_folder = Path(regex_folder)

# %%
table_generator = TableGenerator(
    output_dir=model_cache_manager.get_cache_dir('models')
)

# %%
from transformers import AutoModel, AutoTokenizer
bert_model_name = parameters_dict.get("bert_model_to_use", "bert-base-uncased")
bert_out_dim = AutoModel.from_pretrained(bert_model_name).config.hidden_size

# %% [markdown]
# # Main Method

# %% [markdown]
# ## 1. Mount and load data

# %%
drive_root_folder = mount_drive('tulane-courts-shared/v2')
court_data = load_data(drive_root_folder)

# %%
court_data = append_prefix_to_proceeding_ids(court_data)

# %%
model_configuration = get_model_config()

# %%
TRACKING_LABELS = model_configuration.LABEL_NAMES

# %%
def clean_the_dictionary(dictionary):
    elements_to_remove = model_configuration.EXCLUDED_LABELS
    cleaned_dict = {}
    
    for key, value in dictionary.items():
        if isinstance(value, list):
            # Remove unwanted elements while preserving order
            cleaned_list = [item for item in value if item not in elements_to_remove]
            # Only keep the key if the list is not empty
            if cleaned_list:
                cleaned_dict[key] = cleaned_list
        else:
            # Keep non-list values as they are
            cleaned_dict[key] = value
    
    return cleaned_dict

# %%
print("1. Processing regex labels BEFORE any data drops...")
labels_cache = model_cache_manager.get_cache_dir('labels')
if parameters_dict['regex_labels_mode'] == 'load_in' and parameters_dict.get('regex_labels_load_in_folder'):
    source_file = Path(rf'/data/krosen5/cache/transformers/{parameters_dict["regex_labels_load_in_folder"]}/') / 'labels_cache.pkl'
    destination_file = Path(labels_cache) / 'labels_cache.pkl'
    shutil.copy2(source_file, destination_file)

cached_outcome_labels = generate_and_cache_regex_labels(court_data, labels_cache, parameters_dict['regex_labels_mode'], cache_manager=model_cache_manager)

# %%
def get_active_label_counts(current_court_data, labels_dict):
    """Calculates label counts ONLY for proceedings that currently exist in court_data."""
    active_ids = set()
    if 'mag' in current_court_data and isinstance(current_court_data['mag'], pd.DataFrame) and not current_court_data['mag'].empty:
        active_ids.update(current_court_data['mag']['proceeding_id'].astype(str).tolist())
    if 'cdc' in current_court_data and isinstance(current_court_data['cdc'], pd.DataFrame) and not current_court_data['cdc'].empty:
        active_ids.update(current_court_data['cdc']['proceeding_id'].astype(str).tolist())
        
    c = {label: 0 for label in TRACKING_LABELS}
    for proc_id, labels in labels_dict.items():
        if str(proc_id) in active_ids:
            for label in labels:
                if label in c:
                    c[label] += 1
    return c

# %%
def print_tracking_checkpoint(current_counts, step_name, extra_info="", previous_counts=None):
    """Prints a styled tracking report comparing current counts to previous counts."""
    print(f"\n{'='*90}")
    print(f"📊 {step_name}")
    print(f"{'='*90}")
    if extra_info:
        print(f"{extra_info}\n")
    
    for label in TRACKING_LABELS:
        count = current_counts.get(label, 0)
        short_name = label.replace('_BY_JUDGE_OR_JURY', '').replace('SENTENCE_THAT_', '').replace('_WAS_IMPOSED', '')
        print(f"  - {short_name:<25}: {count:>8,}")
        
    if previous_counts:
        print(f"\nCHANGE FROM PREVIOUS CHECKPOINT:")
        for label in TRACKING_LABELS:
            prev = previous_counts.get(label, 0)
            curr = current_counts.get(label, 0)
            lost = prev - curr
            pct = (lost / prev * 100) if prev > 0 else 0
            short_name = label.replace('_BY_JUDGE_OR_JURY', '').replace('SENTENCE_THAT_', '').replace('_WAS_IMPOSED', '')
            print(f"  - {short_name:<25}: {prev:>8,} → {curr:>8,} (lost {lost:>6,}, {pct:>5.1f}%)")
    print(f"{'='*90}\n")

# %%
# === CHECKPOINT 1: Baseline ===
chk1_counts = get_active_label_counts(court_data, cached_outcome_labels)
print_tracking_checkpoint(chk1_counts, "CHECKPOINT 1: Raw Baseline (After Regex Generation)")

# %%
cached_outcome_labels_v2 = clean_the_dictionary(cached_outcome_labels)

# %%
# === CHECKPOINT 1: Baseline ===
chk2_counts = get_active_label_counts(court_data, cached_outcome_labels_v2)
print_tracking_checkpoint(chk2_counts, "CHECKPOINT 2: Raw Baseline with cleaned (After Regex Generation)")

# %%
import copy
court_data_before_filter = copy.deepcopy(court_data)

# %%
court_data = filter_proceedings(court_data)

# %%
court_data_after_filter = copy.deepcopy(court_data)

# %%
cdc_before_filter = copy.deepcopy(court_data_before_filter['cdc'])
mag_before_filter = copy.deepcopy(court_data_before_filter['mag'])

cdc_after_filter = copy.deepcopy(court_data_after_filter['cdc'])
mag_after_filter = copy.deepcopy(court_data_after_filter['mag'])

# %%
def attach_label_columns_df(df, labels_dict, tracking_labels):
    """
    Adds one column per label (0/1) to a DataFrame using proceeding_id.
    Returns a new DataFrame with the label columns attached.
    """

    if df is None or df.empty:
        return df

    df = df.copy()

    # map proceeding_id -> label list
    df["label_list"] = df["proceeding_id"].astype(str).map(
        lambda x: labels_dict.get(x, labels_dict.get(int(x), [])) if x.isdigit() else labels_dict.get(x, [])
    )

    # convert to set for faster membership checks
    df["label_list"] = df["label_list"].apply(lambda x: set(x) if isinstance(x, (list, set)) else set())

    # create indicator columns
    for label in tracking_labels:
        df[label] = df["label_list"].apply(lambda s: int(label in s))

    df.drop(columns=["label_list"], inplace=True)

    return df

# Attach label columns to both CDC and MAG DataFrames

cdc_before_filter_2 = attach_label_columns_df(cdc_before_filter, cached_outcome_labels_v2, TRACKING_LABELS)
mag_before_filter_2 = attach_label_columns_df(mag_before_filter, cached_outcome_labels_v2, TRACKING_LABELS)
cdc_after_filter_2 = attach_label_columns_df(cdc_after_filter, cached_outcome_labels_v2, TRACKING_LABELS)
mag_after_filter_2 = attach_label_columns_df(mag_after_filter, cached_outcome_labels_v2, TRACKING_LABELS)

# %%
def compare_filter_removals(df_before, df_after, tracking_labels, dataset_name="CDC"):
    """
    Compares before/after filter DataFrames to show what was removed.
    Shows total row counts and label-level breakdown of removed proceedings.
    """
    before_ids = set(df_before['proceeding_id'].astype(str))
    after_ids = set(df_after['proceeding_id'].astype(str))
    removed_ids = before_ids - after_ids
    
    removed_df = df_before[df_before['proceeding_id'].astype(str).isin(removed_ids)]
    
    print(f"\n{'='*70}")
    print(f"🔍 FILTER REMOVAL COMPARISON — {dataset_name}")
    print(f"{'='*70}")
    print(f"  Before filter : {len(df_before):>8,} proceedings")
    print(f"  After filter  : {len(df_after):>8,} proceedings")
    print(f"  Removed       : {len(removed_df):>8,} proceedings ({len(removed_df)/len(df_before)*100:.1f}%)")
    print(f"\n{'─'*70}")
    print(f"  {'Label':<35} {'Before':>8} {'Removed':>8} {'After':>8} {'% Lost':>8}")
    print(f"{'─'*70}")
    
    for label in tracking_labels:
        before_count = df_before[label].sum()
        removed_count = removed_df[label].sum()
        after_count = df_after[label].sum()
        pct_lost = (removed_count / before_count * 100) if before_count > 0 else 0
        short = label.replace('_BY_JUDGE_OR_JURY','').replace('SENTENCE_THAT_','').replace('_WAS_IMPOSED','')
        print(f"  {short:<35} {before_count:>8,} {removed_count:>8,} {after_count:>8,} {pct_lost:>7.1f}%")
    
    print(f"{'='*70}\n")
    return removed_df


# Run for both datasets
removed_cdc = compare_filter_removals(cdc_before_filter_2, cdc_after_filter_2, TRACKING_LABELS, dataset_name="CDC")
removed_mag = compare_filter_removals(mag_before_filter_2, mag_after_filter_2, TRACKING_LABELS, dataset_name="MAG")

# %%
#give me the rows of cdc removed where trial_verdict_reached is 1
removed_cdc_trial_verdict = removed_cdc[removed_cdc['TRIAL_VERDICT_REACHED'] == 1]
print(f"Number of proceedings removed from CDC with TRIAL_VERDICT_REACHED = 1: {len(removed_cdc_trial_verdict)}")

# %%
removed_cdc_trial_verdict_ids = removed_cdc_trial_verdict['case_id'].astype(str).unique().tolist()

# %%
# check how many of them are found in court_data_before_filter['cdc_dockets']
cdc_dockets = court_data_before_filter['cdc_dockets']
found_ids = cdc_dockets[cdc_dockets['case_id'].astype(str).isin(removed_cdc_trial_verdict_ids)]
print(f"Total removed case IDs with TRIAL_VERDICT_REACHED=1: {len(removed_cdc_trial_verdict_ids)}")
print(f"Found in cdc_dockets: {found_ids['case_id'].nunique()}")
print(f"Not found in cdc_dockets: {len(removed_cdc_trial_verdict_ids) - found_ids['case_id'].nunique()}")

# %%


# %%
cdc_dockets[cdc_dockets['defendant_id'].astype(str) == "74"]

# %%
defendant_ids = found_ids['defendant_id'].astype(str).unique().tolist()
print(f"Number of unique defendant IDs: {len(defendant_ids)}")
print(f"Defendant IDs: {defendant_ids}")


# %%
defendants_df = court_data_before_filter['defendants']
found_defendants = defendants_df[defendants_df['defendant_id'].astype(str).isin(defendant_ids)]
print(f"Total defendant IDs to look up: {len(defendant_ids)}")
print(f"Found in defendants table: {found_defendants['defendant_id'].nunique()}")
print(f"Not found: {len(defendant_ids) - found_defendants['defendant_id'].nunique()}")
found_defendants.to_csv('/home/krosen5/Court Case Sequence Transformer Model/court_case_transformer_project/removed_cdc_trial_verdict_defendants.csv', index=False)

# %%
removed_cdc_trial_verdict.to_csv('/home/krosen5/Court Case Sequence Transformer Model/court_case_transformer_project/removed_cdc_trial_verdict.csv', index=False)

# %%
chk3_counts = get_active_label_counts(court_data, cached_outcome_labels_v2)
print_tracking_checkpoint(chk3_counts, "CHECKPOINT 3: After filtering to proceedings with valid labels (After filter_proceedings)")

# %%
crawled_original_data_folder_path = Path("/data/krosen5/gdrive-all/tulane-courts-shared/crawled")
court_data, multi_defendant_cases  = remove_multi_defendant_cases(court_data, crawled_original_data_folder_path)

# %%
chk4_counts = get_active_label_counts(court_data, cached_outcome_labels_v2)
print_tracking_checkpoint(chk4_counts, "CHECKPOINT 4: After removing multi-defendant cases (After remove_multi_defendant_cases)")

# %%
print("   Creating combined dockets...")
court_data['combined_dockets'] = create_combined_dockets(court_data)

# %%
chk5_counts = get_active_label_counts(court_data, cached_outcome_labels_v2)
print_tracking_checkpoint(chk5_counts, "CHECKPOINT 5: After creating combined dockets (After create_combined_dockets)")

# %%
print("Removing cases that start in CDC and have no MAG proceedings...")
court_data = filter_to_magistrate_started_cases(court_data)

# %%
chk6_counts = get_active_label_counts(court_data, cached_outcome_labels_v2)
print_tracking_checkpoint(chk6_counts, "CHECKPOINT 6: After filtering to cases that start in magistrate court (After filter_to_magistrate_started_cases)")

# %%
print("Removing cases that involve some sort of extradition")
court_data = remove_extradition_cases(court_data)

# %%
chk7_counts = get_active_label_counts(court_data, cached_outcome_labels_v2)
print_tracking_checkpoint(chk7_counts, "CHECKPOINT 7: After removing extradition cases (After remove_extradition_cases)")

# %%
print("2b. Filtering out proceedings with zero labels...")
court_data, cached_outcome_labels_v2 = filter_zero_label_proceedings(
    court_data, 
    cached_outcome_labels_v2
)
print(f"   Final labels count after filtering: {len(cached_outcome_labels_v2)}")

# %%
chk8_counts = get_active_label_counts(court_data, cached_outcome_labels_v2)
print_tracking_checkpoint(chk8_counts, "CHECKPOINT 8: After filtering out proceedings with zero labels (After filter_zero_label_proceedings)")

# %%
print("3. Setting up model...")
computation_device = torch.device(f"cuda:{cuda_index}" if torch.cuda.is_available() else "cpu")
print(f"   Using device: {computation_device}") 
case_outcome_model = create_model_from_data(court_data, model_cache_manager.get_cache_dir('bert'), device = computation_device, cache_manager=model_cache_manager, parameters_dict=parameters_dict, bert_out_dim=bert_out_dim)
analyze_model_memory(case_outcome_model)

# %%
chk9_counts = get_active_label_counts(court_data, cached_outcome_labels_v2)
print_tracking_checkpoint(chk9_counts, "CHECKPOINT 9: After model creation (After create_model_from_data)")

# %%
print_label_distribution_by_court(
    court_data,
    cached_outcome_labels_v2,
    model_configuration.LABEL_NAMES,
    mag_key="mag",
    cdc_key="cdc",
    proceeding_col="proceeding_id")

# %% [markdown]
# ## 2. Instantiate model

# %%
computation_device = torch.device(f"cuda:{cuda_index}" if torch.cuda.is_available() else "cpu")
bert_cache_directory = model_cache_manager.get_cache_dir('bert')
proceeding_index_file = bert_cache_directory / "proceeding_index.pkl"
# only build the index (and embeddings.h5) if it doesn't exist
print(f"Checking for cached BERT embeddings in {bert_cache_directory}")
if not proceeding_index_file.exists():
    model_name = parameters_dict['bert_model_to_use'] if 'bert_model_to_use' in parameters_dict else "bert-base-uncased"
    extract_and_cache_bert_embeddings_streaming(
        court_data,
        cache_dir=bert_cache_directory,
        model_name=parameters_dict['bert_model_to_use'] if 'bert_model_to_use' in parameters_dict else "bert-base-uncased",
        out_dim=bert_out_dim,
        batch_size=parameters_dict['bert_batch_size'] if 'bert_batch_size' in parameters_dict else 32,
        device=computation_device,
        force_recompute=False,
        cache_manager=model_cache_manager
    )

# %%
print("Creating model from data")
case_outcome_model = create_model_from_data(court_data, bert_cache_directory, device=computation_device, cache_manager=model_cache_manager, parameters_dict=parameters_dict, bert_out_dim=bert_out_dim)

# %%
chk10_counts = get_active_label_counts(court_data, cached_outcome_labels_v2)
print_tracking_checkpoint(chk10_counts, "CHECKPOINT 10: After creating model from data (After create_model_from_data)")

# %% [markdown]
# ## 3. Build list of (mag_num, case_id, defendant_id) tuples

# %%
print("Building list of defendant cases for training")
combined_dockets_df = court_data['combined_dockets']
defendant_case_identifiers = []
for _, row in combined_dockets_df.iterrows():
    case_identifier = (row.get('mag_num'), row.get('case_id'), row.get('defendant_id'))
    defendant_case_identifiers.append(case_identifier)
valid_defendant_case_ids = [
    case_id for case_id in defendant_case_identifiers 
    if pd.notna(case_id[2]) and (pd.notna(case_id[0]) or pd.notna(case_id[1]))
]

# %%
chk11_counts = get_active_label_counts(court_data, cached_outcome_labels_v2)
print_tracking_checkpoint(chk11_counts, "CHECKPOINT 11: After building list of defendant cases for training")

# %% [markdown]
# ## 4. Sample a subset and create sequences with prior-case features
# 

# %%
print("Filtering to completed cases only for training")
completed_defendant_cases = filter_to_completed_cases(
    court_data, 
    valid_defendant_case_ids, 
    cached_outcome_labels_v2
)
print(f"Found {len(completed_defendant_cases)} completed cases for training")

# %%
chk12_counts = get_active_label_counts(court_data, cached_outcome_labels_v2)
print_tracking_checkpoint(chk12_counts, "CHECKPOINT 12: After filtering to completed cases only for training")

# %%
# Pre-filter valid cases first
print("Pre-filtering valid cases...")
_counts = build_case_counts_df(court_data, cached_outcome_labels_v2)

_df_cases = pd.DataFrame(completed_defendant_cases,
                         columns=['mag_num','case_id','defendant_id'])

_tagged = (_df_cases.merge(_counts, on=['mag_num','case_id','defendant_id'], how='left')
                    .assign(valid=lambda d: d['valid'].fillna(False)))

valid_completed_defendant_cases = [tuple(x) for x in
    _tagged.loc[_tagged['valid'], ['mag_num','case_id','defendant_id']].to_numpy()]

# %%
#give me the valid completed defendant cases with case id of "556090"
specific_cases = [case for case in valid_completed_defendant_cases if case[1] == "556090"]
print(f"Specific cases with case_id '556090': {specific_cases}")

# %%
df_cases = pd.DataFrame(valid_completed_defendant_cases, columns=['mag_num', 'case_id', 'defendant_id']).copy()

df_cases['case_id_str'] = df_cases['case_id'].astype(str).where(df_cases['case_id'].notna(), None)
df_cases['defendant_id_str'] = df_cases['defendant_id'].astype(str).where(df_cases['defendant_id'].notna(), None)
df_cases['has_mag'] = df_cases['mag_num'].notna()

problem_pairs = (
    df_cases.groupby(['case_id_str', 'defendant_id_str'])
    .agg(
        n_rows=('case_id_str', 'size'),
        has_linked_row=('has_mag', 'any'),
        has_unlinked_row=('has_mag', lambda s: (~s).any()),
        mag_nums=('mag_num', lambda x: sorted(set(str(v) for v in x if pd.notna(v))))
    )
    .reset_index()
)

problem_pairs = problem_pairs[
    problem_pairs['has_linked_row'] & problem_pairs['has_unlinked_row']
].copy()

print(problem_pairs.to_string(index=False))
print(f"\nTotal pairs with both linked and unlinked versions: {len(problem_pairs)}")

# %%
chk13_counts = get_active_label_counts(court_data, cached_outcome_labels_v2)
print_tracking_checkpoint(chk13_counts, "CHECKPOINT 13: After pre-filtering valid cases for training")

# %%
if not parameters_dict['new_dataset_creation'] and parameters_dict['sampling_strategy'] is None:
    print(f"Sampling {min(parameters_dict['number_of_cases_in_sample'], len(valid_completed_defendant_cases))} out of {len(valid_completed_defendant_cases)} valid cases")
    set_global_seed(parameters_dict['random_seed'])  # Set global random state
    training_case_sample = random.sample(valid_completed_defendant_cases, min(parameters_dict['number_of_cases_in_sample'], len(valid_completed_defendant_cases)))
elif parameters_dict['sequence_data_mode'] == 'import':
    training_case_sample = None
else: 
    training_case_sample = valid_completed_defendant_cases

# %%
models_directory_cooccurences_path = model_cache_manager.get_cache_dir('models')
models_directory_cooccurences_path.mkdir(parents=True, exist_ok=True)
cooccurrences_output_file = models_directory_cooccurences_path / "cooccurrence_matrix_valid_completed_cases.xlsx"

# %%
export_cooccurrence_from_valid_completed_cases(
    valid_completed_defendant_cases=valid_completed_defendant_cases,
    court_data=court_data,
    cached_outcome_labels_v2=cached_outcome_labels_v2,
    label_names=model_configuration.LABEL_NAMES,
    output_path=cooccurrences_output_file
)

# %%
clerk_cache_dir = model_cache_manager.get_cache_dir('clerk_analysis')
analyze_clerk_data(
    court_data,
    cached_outcome_labels_v2,
    valid_completed_defendant_cases, 
    output_dir = clerk_cache_dir
)

# %%
chk14_counts = get_active_label_counts(court_data, cached_outcome_labels_v2)
print_tracking_checkpoint(chk14_counts, "CHECKPOINT 14: After exporting cooccurrence matrix and analyzing clerk data")

# %%
if not parameters_dict['sequence_data_mode'] == 'import':
    print("Creating Enhanced Label Processor with predefined labels")
    # Initialize the label processor with your 22 predefined labels
    label_processor = EnhancedLabelProcessor(predefined_label_names=model_configuration.LABEL_NAMES)

    print("Creating training sequences with MultiLabelBinarizer support")
    # Use the new sequence creation function that returns raw string labels
    sequences, label_sequences_raw = create_enhanced_training_sequences_cached(
        training_case_sample, 
        court_data, 
        cached_outcome_labels_v2,
        skip_duplicate_labels=parameters_dict['skip_duplicate_labels']
    )

    print(f"Created {len(sequences)} sequences")
    print(f"Sample sequence length: {len(sequences[0]) if sequences else 0}")
    print(f"Sample labels: {label_sequences_raw[0][0] if label_sequences_raw else []}")

    # Fit the MultiLabelBinarizer on all your label data
    print("Fitting MultiLabelBinarizer on training data...")
    label_processor.fit(label_sequences_raw)

    # Verify the fitted labels match your expectations
    print(f"Label processor fitted with {len(label_processor.label_names)} labels:")
    print(f"Expected: {len(model_configuration.LABEL_NAMES)} labels")
    print(f"Match: {set(label_processor.label_names) == set(model_configuration.LABEL_NAMES)}")

# %%
# ============================================================
# DIAGNOSTIC 7: At STAGE 1 — unique proc_ids with PLED_GUILTY
# Run IMMEDIATELY after create_enhanced_training_sequences_cached
# ============================================================

target_labels = ['PLED_GUILTY_TO_CHARGES', 'TRIAL_VERDICT_REACHED',
                 'SENTENCE_THAT_INCLUDES_JAIL_TIME_WAS_IMPOSED',
                 'SENTENCE_THAT_DOES_NOT_INCLUDE_JAIL_TIME_WAS_IMPOSED']

# 1. Count unique proc_ids per label, and how many are duplicated
from collections import Counter, defaultdict

for target_label in target_labels:
    proc_counts = Counter()  # proc_id -> how many times it appears with this label
    
    for seq, label_seq in zip(sequences, label_sequences_raw):
        for proc, step_labels in zip(seq, label_seq):
            if target_label in step_labels:
                pid = proc.get('proceeding_id', '')
                proc_counts[pid] += 1
    
    unique_procs = len(proc_counts)
    total_step_count = sum(proc_counts.values())
    multi_counted = {pid: cnt for pid, cnt in proc_counts.items() if cnt > 1}
    extra_from_dupes = sum(c - 1 for c in proc_counts.values())
    
    chk14 = chk14_counts.get(target_label, 0)
    
    print(f"\n{target_label}:")
    print(f"  Checkpoint 14 (unique procs in ALL court_data): {chk14}")
    print(f"  Unique proc_ids in sequences with this label:   {unique_procs}")
    print(f"  Total step-level count:                         {total_step_count}")
    print(f"  Procs appearing >1 time:                        {len(multi_counted)}")
    print(f"  Extra counts from duplication:                   {extra_from_dupes}")
    print(f"  unique_procs + extra = step_count?               {unique_procs + extra_from_dupes} == {total_step_count}: {unique_procs + extra_from_dupes == total_step_count}")
    print(f"  unique_procs vs chk14:                           {unique_procs} vs {chk14}: {'⚠️ MORE THAN CHK14' if unique_procs > chk14 else '✓ SUBSET'}")

# %%
from collections import Counter
import pandas as pd

target_labels = [
    'PLED_GUILTY_TO_CHARGES',
    'TRIAL_VERDICT_REACHED',
    'SENTENCE_THAT_INCLUDES_JAIL_TIME_WAS_IMPOSED',
    'SENTENCE_THAT_DOES_NOT_INCLUDE_JAIL_TIME_WAS_IMPOSED'
]

def get_checkpoint_proc_ids_for_label(court_data, labels_dict, target_label):
    """
    Get the set of proceeding_ids currently active in court_data that have target_label
    according to cached labels.
    """
    active_ids = set()

    if 'mag' in court_data and isinstance(court_data['mag'], pd.DataFrame) and not court_data['mag'].empty:
        active_ids.update(court_data['mag']['proceeding_id'].astype(str).tolist())

    if 'cdc' in court_data and isinstance(court_data['cdc'], pd.DataFrame) and not court_data['cdc'].empty:
        active_ids.update(court_data['cdc']['proceeding_id'].astype(str).tolist())

    labeled_ids = set()
    for proc_id, labels in labels_dict.items():
        proc_id_str = str(proc_id)
        if proc_id_str in active_ids and target_label in labels:
            labeled_ids.add(proc_id_str)

    return labeled_ids


def get_sequence_proc_counts_for_label(sequences, label_sequences_raw, target_label):
    """
    Count how many times each proceeding_id appears in sequences with target_label.
    """
    proc_counts = Counter()

    for seq, label_seq in zip(sequences, label_sequences_raw):
        for proc, step_labels in zip(seq, label_seq):
            if target_label in step_labels:
                pid = str(proc.get('proceeding_id', ''))
                if pid:
                    proc_counts[pid] += 1

    return proc_counts


def build_proc_lookup_df(court_data):
    """
    Build one combined lookup table from mag + cdc for easier inspection.
    """
    frames = []

    if 'mag' in court_data and isinstance(court_data['mag'], pd.DataFrame) and not court_data['mag'].empty:
        mag_df = court_data['mag'].copy()
        mag_df['source_table'] = 'mag'
        frames.append(mag_df)

    if 'cdc' in court_data and isinstance(court_data['cdc'], pd.DataFrame) and not court_data['cdc'].empty:
        cdc_df = court_data['cdc'].copy()
        cdc_df['source_table'] = 'cdc'
        frames.append(cdc_df)

    if not frames:
        return pd.DataFrame()

    lookup_df = pd.concat(frames, ignore_index=True)
    lookup_df['proceeding_id'] = lookup_df['proceeding_id'].astype(str)
    return lookup_df


proc_lookup_df = build_proc_lookup_df(court_data)

all_duplicate_rows = []
all_missing_rows = []

for target_label in target_labels:
    checkpoint_ids = get_checkpoint_proc_ids_for_label(court_data, cached_outcome_labels_v2, target_label)
    seq_proc_counts = get_sequence_proc_counts_for_label(sequences, label_sequences_raw, target_label)
    sequence_ids = set(seq_proc_counts.keys())

    duplicate_ids = sorted([pid for pid, cnt in seq_proc_counts.items() if cnt > 1])
    missing_ids = sorted(checkpoint_ids - sequence_ids)

    print(f"\n{'='*100}")
    print(f"{target_label}")
    print(f"{'='*100}")
    print(f"Checkpoint unique ids: {len(checkpoint_ids)}")
    print(f"Sequence unique ids:   {len(sequence_ids)}")
    print(f"Duplicate ids:         {len(duplicate_ids)}")
    print(f"Missing ids:           {len(missing_ids)}")

    if duplicate_ids:
        dup_df = pd.DataFrame({
            'proceeding_id': duplicate_ids,
            'times_seen_in_sequences': [seq_proc_counts[pid] for pid in duplicate_ids],
            'label': target_label
        })

        if not proc_lookup_df.empty:
            dup_df = dup_df.merge(
                proc_lookup_df,
                on='proceeding_id',
                how='left'
            )

        print("\nTop duplicated proceeding_ids:")
        display_cols = [c for c in [
            'proceeding_id',
            'times_seen_in_sequences',
            'label',
            'source_table',
            'case_id',
            'mag_num',
            'defendant_id',
            'date',
            'text'
        ] if c in dup_df.columns]
        print(dup_df[display_cols].head(20).to_string(index=False))

        all_duplicate_rows.append(dup_df)

    if missing_ids:
        missing_df = pd.DataFrame({
            'proceeding_id': missing_ids,
            'label': target_label
        })

        if not proc_lookup_df.empty:
            missing_df = missing_df.merge(
                proc_lookup_df,
                on='proceeding_id',
                how='left'
            )

        print("\nTop missing proceeding_ids:")
        display_cols = [c for c in [
            'proceeding_id',
            'label',
            'source_table',
            'case_id',
            'mag_num',
            'defendant_id',
            'date',
            'text'
        ] if c in missing_df.columns]
        print(missing_df[display_cols].head(20).to_string(index=False))

        all_missing_rows.append(missing_df)

# Save full outputs
if all_duplicate_rows:
    from collections import Counter
import pandas as pd

target_labels = [
    'PLED_GUILTY_TO_CHARGES',
    'TRIAL_VERDICT_REACHED',
    'SENTENCE_THAT_INCLUDES_JAIL_TIME_WAS_IMPOSED',
    'SENTENCE_THAT_DOES_NOT_INCLUDE_JAIL_TIME_WAS_IMPOSED'
]

def get_checkpoint_proc_ids_for_label(court_data, labels_dict, target_label):
    """
    Get the set of proceeding_ids currently active in court_data that have target_label
    according to cached labels.
    """
    active_ids = set()

    if 'mag' in court_data and isinstance(court_data['mag'], pd.DataFrame) and not court_data['mag'].empty:
        active_ids.update(court_data['mag']['proceeding_id'].astype(str).tolist())

    if 'cdc' in court_data and isinstance(court_data['cdc'], pd.DataFrame) and not court_data['cdc'].empty:
        active_ids.update(court_data['cdc']['proceeding_id'].astype(str).tolist())

    labeled_ids = set()
    for proc_id, labels in labels_dict.items():
        proc_id_str = str(proc_id)
        if proc_id_str in active_ids and target_label in labels:
            labeled_ids.add(proc_id_str)

    return labeled_ids


def get_sequence_proc_counts_for_label(sequences, label_sequences_raw, target_label):
    """
    Count how many times each proceeding_id appears in sequences with target_label.
    """
    proc_counts = Counter()

    for seq, label_seq in zip(sequences, label_sequences_raw):
        for proc, step_labels in zip(seq, label_seq):
            if target_label in step_labels:
                pid = str(proc.get('proceeding_id', ''))
                if pid:
                    proc_counts[pid] += 1

    return proc_counts


def build_proc_lookup_df(court_data):
    """
    Build one combined lookup table from mag + cdc for easier inspection.
    """
    frames = []

    if 'mag' in court_data and isinstance(court_data['mag'], pd.DataFrame) and not court_data['mag'].empty:
        mag_df = court_data['mag'].copy()
        mag_df['source_table'] = 'mag'
        frames.append(mag_df)

    if 'cdc' in court_data and isinstance(court_data['cdc'], pd.DataFrame) and not court_data['cdc'].empty:
        cdc_df = court_data['cdc'].copy()
        cdc_df['source_table'] = 'cdc'
        frames.append(cdc_df)

    if not frames:
        return pd.DataFrame()

    lookup_df = pd.concat(frames, ignore_index=True)
    lookup_df['proceeding_id'] = lookup_df['proceeding_id'].astype(str)
    return lookup_df


proc_lookup_df = build_proc_lookup_df(court_data)

all_duplicate_rows = []
all_missing_rows = []

for target_label in target_labels:
    checkpoint_ids = get_checkpoint_proc_ids_for_label(court_data, cached_outcome_labels_v2, target_label)
    seq_proc_counts = get_sequence_proc_counts_for_label(sequences, label_sequences_raw, target_label)
    sequence_ids = set(seq_proc_counts.keys())

    duplicate_ids = sorted([pid for pid, cnt in seq_proc_counts.items() if cnt > 1])
    missing_ids = sorted(checkpoint_ids - sequence_ids)

    print(f"\n{'='*100}")
    print(f"{target_label}")
    print(f"{'='*100}")
    print(f"Checkpoint unique ids: {len(checkpoint_ids)}")
    print(f"Sequence unique ids:   {len(sequence_ids)}")
    print(f"Duplicate ids:         {len(duplicate_ids)}")
    print(f"Missing ids:           {len(missing_ids)}")

    if duplicate_ids:
        dup_df = pd.DataFrame({
            'proceeding_id': duplicate_ids,
            'times_seen_in_sequences': [seq_proc_counts[pid] for pid in duplicate_ids],
            'label': target_label
        })

        if not proc_lookup_df.empty:
            dup_df = dup_df.merge(
                proc_lookup_df,
                on='proceeding_id',
                how='left'
            )

        print("\nTop duplicated proceeding_ids:")
        display_cols = [c for c in [
            'proceeding_id',
            'times_seen_in_sequences',
            'label',
            'source_table',
            'case_id',
            'mag_num',
            'defendant_id',
            'date',
            'text'
        ] if c in dup_df.columns]
        print(dup_df[display_cols].head(20).to_string(index=False))

        all_duplicate_rows.append(dup_df)

    if missing_ids:
        missing_df = pd.DataFrame({
            'proceeding_id': missing_ids,
            'label': target_label
        })

        if not proc_lookup_df.empty:
            missing_df = missing_df.merge(
                proc_lookup_df,
                on='proceeding_id',
                how='left'
            )

        print("\nTop missing proceeding_ids:")
        display_cols = [c for c in [
            'proceeding_id',
            'label',
            'source_table',
            'case_id',
            'mag_num',
            'defendant_id',
            'date',
            'text'
        ] if c in missing_df.columns]
        print(missing_df[display_cols].head(20).to_string(index=False))

        all_missing_rows.append(missing_df)

# Save full outputs
if all_duplicate_rows:
    duplicate_output_df = pd.concat(all_duplicate_rows, ignore_index=True)
else:
    duplicate_output_df = pd.DataFrame()

if all_missing_rows:
    missing_output_df = pd.concat(all_missing_rows, ignore_index=True)
else:
    missing_output_df = pd.DataFrame()

duplicate_output_path = model_cache_manager.get_cache_dir('analysis') / "duplicate_proceeding_ids_by_label.xlsx"
missing_output_path = model_cache_manager.get_cache_dir('analysis') / "missing_proceeding_ids_by_label.xlsx"

duplicate_output_df.to_excel(duplicate_output_path, index=False)
missing_output_df.to_excel(missing_output_path, index=False)

print(f"\nSaved duplicates to: {duplicate_output_path}")
print(f"Saved missing ids to: {missing_output_path}") = pd.concat(all_duplicate_rows, ignore_index=True)
else:
    duplicate_output_df = pd.DataFrame()

if all_missing_rows:
    missing_output_df = pd.concat(all_missing_rows, ignore_index=True)
else:
    missing_output_df = pd.DataFrame()

# %%
#give me the sequences where the duplicate proceeding_ids appear (the whole sequence with the proc ids)
duplicate_proc_ids = set(duplicate_output_df['proceeding_id'].astype(str).tolist())
sequences_with_duplicates = []
for seq, label_seq in zip(sequences, label_sequences_raw):
    seq_proc_ids = {str(proc.get('proceeding_id', '')) for proc in seq}
    if seq_proc_ids.intersection(duplicate_proc_ids):
        sequences_with_duplicates.append((seq, label_seq))



# %%
#just give me the sequence proceeding_ids for the sequences with duplicates
for seq, label_seq in sequences_with_duplicates:
    seq_proc_ids = [str(proc.get('proceeding_id', '')) for proc in seq]
    print(seq_proc_ids)


# %%
!ls

# %%
# ============================================================
# DIAGNOSTIC 6: Count labels at each stage to show the overwrite
# Run this by temporarily inserting counters at each stage
# ============================================================

# --- RIGHT AFTER create_enhanced_training_sequences_cached ---
print(f"=== STAGE 1: After sequence creation ===")
print(f"  len(sequences) = {len(sequences)}")
stage1_count = Counter()
for seq_labels in label_sequences_raw:
    for step in seq_labels:
        stage1_count.update(step)
print(f"  PLED_GUILTY count: {stage1_count.get('PLED_GUILTY_TO_CHARGES', 0)}")

# Show how many sequences per case
from collections import Counter
# Extract docket_id from first proceeding of each sequence to group by case
case_to_seq_count = Counter()
for seq in sequences:
    first_proc = seq[0] if seq else {}
    docket_id = first_proc.get('docket_id', 'unknown')
    case_to_seq_count[docket_id] += 1

multi_seq_cases = {k: v for k, v in case_to_seq_count.items() if v > 1}
print(f"  Cases producing >1 sequence: {len(multi_seq_cases)}")
print(f"  Max sequences from one case: {max(case_to_seq_count.values())}")
seq_count_dist = Counter(case_to_seq_count.values())
print(f"  Distribution of sequences-per-case:")
for n_seqs, n_cases in sorted(seq_count_dist.items()):
    print(f"    {n_seqs} sequences: {n_cases} cases")

# %%
print("Count total label occurences across all sequences:")
labels_count = Counter()
for sequence in label_sequences_raw:
    for step_labels in sequence:
        labels_count.update(step_labels)

#print it
for label in model_configuration.LABEL_NAMES:
    count = labels_count.get(label, 0)
    print(f"- {label}: {count}")

# %%
print("Count how many sequences contain each label")
label_sequence_counts = Counter()
for sequence in label_sequences_raw:
    labels_in_seq = set(label for label_set in sequence for label in label_set)
    label_sequence_counts.update(labels_in_seq)

for label in model_configuration.LABEL_NAMES:
    count = label_sequence_counts.get(label, 0)
    print(f"- {label}: {count} sequences")

# %%
if not parameters_dict['sequence_data_mode'] == 'import':
    print("\n" + "="*80)
    print("TRUNCATING SEQUENCES AT CASE CONCLUSION")
    print("="*80)

    sequences, label_sequences_raw = filter_proceedings_to_end_of_case(
        sequences,
        label_sequences_raw
    )

    print(f"\nFinal sequences after truncation: {len(sequences)}")
    print("="*80)

# %%
print(f"\n=== STAGE 2: After filter_proceedings_to_end_of_case ===")
print(f"  len(sequences) = {len(sequences)}")
stage2_count = Counter()
for seq_labels in label_sequences_raw:
    for step in seq_labels:
        stage2_count.update(step)
print(f"  PLED_GUILTY count: {stage2_count.get('PLED_GUILTY_TO_CHARGES', 0)}")

# %%
print("Count total label occurences across all sequences after truncation of sequences at case conclusion:")

labels_count_after_truncation = Counter()
for sequence in label_sequences_raw:
    for step_labels in sequence:
        labels_count_after_truncation.update(step_labels)

for label in model_configuration.LABEL_NAMES:
    count = labels_count_after_truncation.get(label, 0)
    print(f"- {label}: {count}")

# %%
print("Count how many sequences contain each label after truncation of sequences at case conclusion")

labels_sequence_counts_after_truncation = Counter()
for sequence in label_sequences_raw:
    labels_in_seq = set(label for label_set in sequence for label in label_set)
    labels_sequence_counts_after_truncation.update(labels_in_seq)

for label in model_configuration.LABEL_NAMES:
    count = labels_sequence_counts_after_truncation.get(label, 0)
    print(f"- {label}: {count} sequences")

# %%
cases = list(zip(sequences, label_sequences_raw))
export_all_cases_to_xlsx(cases, model_cache_manager.get_cache_dir('datasets') / "all_cases_after_truncation.xlsx", court_data)

# %%
# Check what mag_dockets has for this specific case
print("=== mag_dockets for mag_num 600648 ===")
print(court_data['mag_dockets'][court_data['mag_dockets']['mag_num'] == 600648][['mag_num', 'case_id', 'defendant_id']])
print(court_data['mag_dockets'][court_data['mag_dockets']['mag_num'] == '600648'][['mag_num', 'case_id', 'defendant_id']])

# Check what cdc_dockets has for case_id 556090
print("\n=== cdc_dockets for case_id 556090 ===")
print(court_data['cdc_dockets'][court_data['cdc_dockets']['case_id'] == 556090][['case_id', 'defendant_id']])
print(court_data['cdc_dockets'][court_data['cdc_dockets']['case_id'] == '556090'][['case_id', 'defendant_id']])

# Check combined_dockets for this defendant
print("\n=== combined_dockets for defendant 798256 ===")
print(court_data['combined_dockets'][court_data['combined_dockets']['defendant_id'] == 798256][['mag_num', 'case_id', 'defendant_id']])

# Check dtypes
print("\n=== dtypes ===")
print("mag_dockets case_id:", court_data['mag_dockets']['case_id'].dtype)
print("cdc_dockets case_id:", court_data['cdc_dockets']['case_id'].dtype)
print("mag_dockets mag_num:", court_data['mag_dockets']['mag_num'].dtype)

# %%
data_mode = parameters_dict.get('sequence_data_mode', 'current')
if data_mode == 'current':
    print("Using current sequences and labels.")
elif data_mode == 'export':
    if parameters_dict['new_dataset_creation']:
        print("Applying new_dataset_creation sampling...")
        set_global_seed(parameters_dict['random_seed'])  # Set global random state
        cases = list(zip(sequences, label_sequences_raw))
        if parameters_dict['sampling_strategy'] == 'oversample_long_sequences':
            print("Sampling strategy: oversample_long_sequences")
            sampled_cases, sampled_lengths, sampled_indices = sample_from_the_longest_cases(
                cases,
                number_of_longest_cases=5000,
                sample_size=parameters_dict['number_of_cases_in_sample']        
                )
            export_long_cases_to_xlsx(
                sampled_cases, 
                output_filepath=model_cache_manager.get_cache_dir('datasets') / "longest_case_samples.xlsx",
                court_data=court_data
            )
        elif parameters_dict['sampling_strategy'] == 'uniform':
            print("Sampling strategy: uniform")
            sampled_cases, sampled_lengths, sampled_indices = sample_uniformly(
                cases,
                sample_size=parameters_dict['number_of_cases_in_sample']
            )
            export_uniform_cases_to_xlsx(
                sampled_cases, 
                output_filepath=model_cache_manager.get_cache_dir('datasets') / "uniform_case_samples.xlsx",
                court_data=court_data
            )
        elif parameters_dict['sampling_strategy'] == 'balanced': 
            print("Sampling strategy: balanced")
            sampled_cases, sampled_lengths, sampled_indices, low_freq_labels = create_balanced_dataset(
                cases,
                low_frequency_threshold=300,
                target_size=parameters_dict['number_of_cases_in_sample']
            )
            export_balanced_cases_to_xlsx(
                sampled_cases,
                sampled_indices,
                output_filepath=model_cache_manager.get_cache_dir('datasets') / "balanced_case_samples.xlsx",
                low_freq_labels=low_freq_labels,
                court_data=court_data
            )
        else:
            print("No sampling strategy applied, using all cases.")
            sequences, label_sequences_raw = zip(*cases)
        # Set sequences and label_sequences_raw to the sampled cases
        sequences, label_sequences_raw = zip(*sampled_cases)
        sequences = list(sequences)
        label_sequences_raw = list(label_sequences_raw)
        print(f"Sampled down to {len(sequences)} sequences")
    
    case_identifiers = extract_case_identifier_from_sequences(sequences, label_sequences_raw)
    identifiers_path = model_cache_manager.get_cache_dir('datasets') / 'case_identifiers'
    save_case_identifiers(case_identifiers, identifiers_path)
    
    export_filename = 'sequences_and_labels.pkl'
    export_path = model_cache_manager.get_cache_dir('datasets') / export_filename
    
    print(f"Exporting {len(sequences)} sequences to: {export_path}")
    with open(export_path, 'wb') as file:
        pickle.dump((sequences, label_sequences_raw), file)

        indices_path = model_cache_manager.get_cache_dir('datasets') / 'sampled_indices.pkl'
    with open(indices_path, 'wb') as file:
        pickle.dump(sampled_indices, file)

    print(f"Saved sampled indices to: {indices_path}")

    
    # Also save the label processor
    label_processor_path = model_cache_manager.get_cache_dir('models') / 'label_processor.pkl'
    with open(label_processor_path, 'wb') as file:
        pickle.dump(label_processor, file)
    
    print(f"Exported {len(sequences)} sequences")
    print(f"Sequences: {export_path}")
    print(f"Label processor: {label_processor_path}")

elif data_mode == 'import':
    import_filename = 'sequences_and_labels.pkl'
    import_path = import_cache_manager.get_cache_dir('datasets') / import_filename
    
    if not import_path.exists():
        raise FileNotFoundError(f"Import file not found: {import_path}")
    
    print(f"Importing sequences from: {import_path}")
    with open(import_path, 'rb') as file:
        sequences, label_sequences_raw = pickle.load(file)

    #sampled indices loading
    sampled_indices_path = import_cache_manager.get_cache_dir('datasets') / 'sampled_indices.pkl'
    if sampled_indices_path.exists():
        with open(sampled_indices_path, 'rb') as file:
            original_sampled_indices = pickle.load(file)
        print(f"Loaded original sampled indices: {len(original_sampled_indices)} indices")
    else:
        print("Warning: Original sampled_indices not found. Cannot map back to original cases.")
        original_sampled_indices = None
    
    case_identifiers_before_skipping = extract_case_identifier_from_sequences(sequences, label_sequences_raw)
    case_identifiers_before_path = model_cache_manager.get_cache_dir('datasets') / 'case_identifiers_before_skipping'
    save_case_identifiers(case_identifiers_before_skipping, case_identifiers_before_path)
    
    # Load the label processor
    label_processor_path = import_cache_manager.get_cache_dir('models') / 'label_processor.pkl'
    if label_processor_path.exists():
        print(f"Loading label processor from: {label_processor_path}")
        with open(label_processor_path, 'rb') as file:
            label_processor = pickle.load(file)
    else:
        raise FileNotFoundError(f"Label processor not found: {label_processor_path}. Cannot proceed without it.")
    
    #make a copy of label_processor in the model cache manager
    label_processor_copy_path = model_cache_manager.get_cache_dir('models') / 'label_processor.pkl'
    with open(label_processor_copy_path, 'wb') as file:
        pickle.dump(label_processor, file)
    print(f"Copied label processor to: {label_processor_copy_path}")
    

    
    if parameters_dict['skip_duplicate_labels_on_import']:
        #use apply_skipped_duplicates_on_imported_sequences
        sequences, label_sequences_raw, kept_relative_indices = apply_skipped_duplicates_on_imported_sequences(
            sequences,
            label_sequences_raw,
            return_indices = True
        )
        if original_sampled_indices is not None:
            print("Mapping back to original cases using sampled indices...")
            kept_original_indices = [original_sampled_indices[i] for i in kept_relative_indices]
            # Optionally, save the updated sampled indices
            original_indices_path = model_cache_manager.get_cache_dir('datasets') / 'kept_original_indices.pkl'
            with open(original_indices_path, 'wb') as file:
                pickle.dump(kept_original_indices, file)
            print(f"  Saved to: {original_indices_path}")
        else:
            kept_original_indices = kept_relative_indices
            #make a warning message
            print("Warning: original_sampled_indices is None, so kept_original_indices are relative to imported data.")
        
        case_identifiers_after_skipping = extract_case_identifier_from_sequences(sequences, label_sequences_raw)
        case_identifiers_after_path = model_cache_manager.get_cache_dir('datasets') / 'case_identifiers_after_skipping'
        save_case_identifiers(case_identifiers_after_skipping, case_identifiers_after_path)

        new_import_path = model_cache_manager.get_cache_dir('datasets') / import_filename
        #redump the sequences and labels after applying skipped duplicates
        with open(new_import_path, 'wb') as file:
            pickle.dump((sequences, label_sequences_raw), file)
        #depending on the sampling strategy, we will export it to excel  
        if parameters_dict['sampling_strategy'] == 'balanced':
            export_balanced_cases_to_xlsx(
                list(zip(sequences, label_sequences_raw)),
                kept_original_indices,
                output_filepath=model_cache_manager.get_cache_dir('datasets') / "balanced_case_samples_after_skipping.xlsx",
                court_data=court_data
            )
        elif parameters_dict['sampling_strategy'] == 'uniform':
            export_uniform_cases_to_xlsx(
                list(zip(sequences, label_sequences_raw)),
                output_filepath=model_cache_manager.get_cache_dir('datasets') / "uniform_case_samples_after_skipping.xlsx",
                court_data=court_data
            )
        elif parameters_dict['sampling_strategy'] == 'oversample_long_sequences':
            export_long_cases_to_xlsx(
                list(zip(sequences, label_sequences_raw)),
                output_filepath=model_cache_manager.get_cache_dir('datasets') / "longest_case_samples_after_skipping.xlsx",
                court_data=court_data
            )
        else:
            print("No sampling strategy applied, not exporting to excel after skipping duplicates.")
    if parameters_dict['skip_duplicate_labels_on_import']:
        compare_dictionary = compare_case_identifiers(
            case_identifiers_before_skipping,
            case_identifiers_after_skipping
        )
        #export comparison results
        comparison_path = model_cache_manager.get_cache_dir('datasets') / 'case_identifier_comparison.pkl'
        with open(comparison_path, 'wb') as file:
            pickle.dump(compare_dictionary, file)
        print(f"Saved case identifier comparison to: {comparison_path}")

    print(f"Imported {len(sequences)} sequences")
    print(f"  - Sequences: {import_path}")
    print(f"  - Label processor: {label_processor_path}")


# %%
print(f"\n=== STAGE 3: After sampling/import ===")
print(f"  len(sequences) = {len(sequences)}")
stage3_count = Counter()
for seq_labels in label_sequences_raw:
    for step in seq_labels:
        stage3_count.update(step)
print(f"  PLED_GUILTY count: {stage3_count.get('PLED_GUILTY_TO_CHARGES', 0)}")

# %%
print("Count total label occurences across all sequences after making the balanced dataframe")
labels_count_after_balanced_df = Counter()
for sequence in label_sequences_raw:
    for step_labels in sequence:
        labels_count_after_balanced_df.update(step_labels)

for label in model_configuration.LABEL_NAMES:
    count = labels_count_after_balanced_df.get(label, 0)
    print(f"- {label}: {count}")

# %%
print("Count how many sequences contain each label after making the balanced dataframe")
labels_sequence_counts_after_balanced_df = Counter()
for sequence in label_sequences_raw:
    labels_in_seq = set(label for label_set in sequence for label in label_set)
    labels_sequence_counts_after_balanced_df.update(labels_in_seq)

for label in model_configuration.LABEL_NAMES:
    count = labels_sequence_counts_after_balanced_df.get(label, 0)
    print(f"- {label}: {count} sequences")

# %%
#Checkpoint label comparison table
checkpoint_comparison = []
checkpoints = [
    ("Checkpoint 1: Raw Baseline (After Regex Generation)", chk1_counts),
    ("Checkpoint 2: Raw Baseline with cleaned (After Regex Generation)", chk2_counts),
    ("Checkpoint 3: After filtering to proceedings with valid labels (After filter_proceedings)", chk3_counts),
    ("Checkpoint 4: After removing multi-defendant cases (After remove_multi_defendant_cases)", chk4_counts),
    ("Checkpoint 5: After creating combined dockets (After create_combined_dockets)", chk5_counts),
    ("Checkpoint 6: After filtering to cases that start in magistrate court (After filter_to_magistrate_started_cases)", chk6_counts),
    ("Checkpoint 7: After removing extradition cases (After remove_extradition_cases)", chk7_counts),
    ("Checkpoint 8: After filtering out proceedings with zero labels (After filter_zero_label_proceedings)", chk8_counts),
    ("Checkpoint 9: After model creation (After create_model_from_data)", chk9_counts),
    ("Checkpoint 10: After creating model from data (After create_model_from_data)", chk10_counts),
    ("Checkpoint 11: After building list of defendant cases for training", chk11_counts),
    ("Checkpoint 12: After filtering to completed cases only for training", chk12_counts),
    ("Checkpoint 13: After pre-filtering valid cases for training", chk13_counts),
    ("Checkpoint 14: After exporting cooccurrence matrix and analyzing clerk data", chk14_counts)
]

for checkpoint_name, counts in checkpoints:
    checkpoint_comparison.append({
        "Checkpoint": checkpoint_name,
        **{label: counts.get(label, 0) for label in model_configuration.LABEL_NAMES}
    })

checkpoint_comparison_df = pd.DataFrame(checkpoint_comparison)

# %%
checkpoint_comparison_df.to_excel(model_cache_manager.get_cache_dir('analysis') / "checkpoint_label_comparison.xlsx", index=False)

# %%
#now do the othe rcheckpoints
other_checkpoints = [
    ("After truncating sequences at case conclusion", labels_count_after_truncation, labels_sequence_counts_after_truncation),
    ("After making balanced dataframe", labels_count_after_balanced_df, labels_sequence_counts_after_balanced_df)
]
other_comparison = []
for checkpoint_name, label_counts, sequence_counts in other_checkpoints:
    other_comparison.append({
        "Checkpoint": checkpoint_name,
        **{f"{label} Count": label_counts.get(label, 0) for label in model_configuration.LABEL_NAMES},
        **{f"{label} Sequence Count": sequence_counts.get(label, 0) for label in model_configuration.LABEL_NAMES}
    })
other_comparison_df = pd.DataFrame(other_comparison)
other_comparison_df.to_excel(model_cache_manager.get_cache_dir('analysis') / "other_checkpoint_comparison.xlsx", index=False)

# %%
# ============================================================
# DIAGNOSTIC 3: Where are the 6036 counts coming from with
# only 811 unique proceeding_ids?
# ============================================================

target_label = 'PLED_GUILTY_TO_CHARGES'

# 1. Verify: does sum of per-proc counts match labels_count?
seq_proc_counts_v2 = Counter()
for seq, label_seq in zip(sequences, label_sequences_raw):
    for proc, step_labels in zip(seq, label_seq):
        pid = str(proc.get('proceeding_id', ''))
        if target_label in step_labels:
            seq_proc_counts_v2[pid] += 1

print(f"sum(seq_proc_counts) = {sum(seq_proc_counts_v2.values())}")
print(f"labels_count[PLED_GUILTY] = {labels_count.get(target_label, 0)}")

# 2. Show the distribution of counts for those 13 multi-counted procs
multi = {pid: cnt for pid, cnt in seq_proc_counts_v2.items() if cnt > 1}
print(f"\nMulti-counted procs ({len(multi)}):")
for pid, cnt in sorted(multi.items(), key=lambda x: -x[1])[:20]:
    print(f"  {pid}: appears {cnt} times")

# 3. CHECK FOR REFERENCE BUG: are multiple sequence entries 
#    actually the same Python object?
same_object_count = 0
for i in range(len(sequences)):
    for j in range(i + 1, min(i + 50, len(sequences))):  # check nearby
        if sequences[i] is sequences[j]:  # 'is' checks same object, not equality
            same_object_count += 1
            if same_object_count <= 3:
                print(f"\n⚠️ sequences[{i}] IS sequences[{j}] (same Python object!)")
                print(f"   len={len(sequences[i])}, id={id(sequences[i])}")

if same_object_count > 0:
    print(f"\n🚨 FOUND {same_object_count} pairs sharing the same list object!")
else:
    print("\n✓ No shared list objects found in nearby sequences")

# 4. More thorough: group sequences by object id
from collections import Counter
object_ids = Counter(id(seq) for seq in sequences)
duplicated_objects = {oid: cnt for oid, cnt in object_ids.items() if cnt > 1}
print(f"\nUnique list objects in sequences: {len(object_ids)}")
print(f"List objects appearing >1 time: {len(duplicated_objects)}")
if duplicated_objects:
    total_extra = sum(c - 1 for c in duplicated_objects.values())
    print(f"Total extra references: {total_extra}")
    print(f"\nDistribution of reference counts:")
    ref_count_dist = Counter(duplicated_objects.values())
    for count, freq in sorted(ref_count_dist.items()):
        print(f"  Appears {count} times: {freq} list objects")


