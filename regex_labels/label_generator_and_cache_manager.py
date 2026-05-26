# Python File that has functions related to generating and caching the labels that were made from regular expression

import pickle
import re
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from datetime import datetime
import pandas as pd

import io
import pickle
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm import tqdm

from regex_labels.patterns_and_extractors import (
    extract_first_appearance_hearing, determine_bond_status, should_include_ror,
    extract_the_court_appointed_the_orleans_public_defender, extract_ankle_monitoring,
    extract_domestic_stay_away_order, determine_charge_status, 
    #extract_gwens_law_hearing,
    has_probable_cause, extract_defendant_in_custody_and_not_in_court,
    extract_the_state_was_not_present, extract_pre_trial_conference,
    determine_continuance, plea_status, verdict_status, sentence_jail_status
)
from config import get_model_config
from utils.cache_directory_utils import CacheDirectoryManager

# SECTION Generate True Labels
def true_labels_generator(proceeding):
  labels = []
  for method in (extract_first_appearance_hearing, determine_bond_status, should_include_ror, extract_the_court_appointed_the_orleans_public_defender, extract_ankle_monitoring, extract_domestic_stay_away_order, determine_charge_status, 
  #extract_gwens_law_hearing, 
  has_probable_cause, extract_defendant_in_custody_and_not_in_court, extract_the_state_was_not_present, extract_pre_trial_conference, determine_continuance, plea_status, verdict_status, sentence_jail_status):
    output = method(proceeding or "")
    if output is None:
      continue
    if isinstance(output, str):
      labels.append(output)
    else:
      labels.extend(output)
  return list(dict.fromkeys(labels))

# SECTION Optimized label generation using cache lookup with fallback
def true_labels_generator_optimization(proceeding_text, proceeding_id, labels_cache):
  model_configuration = get_model_config()
  if labels_cache is not None and proceeding_id is not None:
    labels = labels_cache.get(proceeding_id, [])
    return [lbl for lbl in labels if lbl not in model_configuration.EXCLUDED_LABELS]
  if proceeding_text is not None:
    return true_labels_generator(proceeding_text)
  return []

# SECTION Method to generate and cache regex labels
# def generate_and_cache_regex_labels(data, dir_for_cache, load_in_or_start_from_scratch, cache_manager = None):
#     if dir_for_cache is None:
#         dir_for_cache = cache_manager.get_cache_dir("labels")   
#     if not dir_for_cache.exists():
#         dir_for_cache.mkdir(parents=True, exist_ok=True)
#     labels_cache_file = dir_for_cache / "labels_cache.pkl"

#     proceeding_labels = {}
#     # Load existing cache
#     if load_in_or_start_from_scratch == "start_from_scratch":
#         proceeding_labels = {}
#     elif load_in_or_start_from_scratch == "load_in":
#       if labels_cache_file.exists():
#           with open(labels_cache_file, "rb") as f:
#               print(f"Loading labels cache from {labels_cache_file}")
#               proceeding_labels = pickle.load(f)

#     combined_data = pd.concat([data['mag'], data['cdc']], ignore_index=True)

#     current_ids = set(combined_data['proceeding_id'])
#     ids = combined_data['proceeding_id']
#     texts = combined_data['text'] if 'text' in combined_data.columns else pd.Series([''] * len(combined_data))


#     for proceeding_id, text in tqdm(zip(ids, texts), total=len(combined_data)):
#         if proceeding_id not in proceeding_labels:
#             proceeding_labels[proceeding_id] = true_labels_generator(text)

#     if len(proceeding_labels) > len(current_ids):
#       proceeding_labels = {k: v for k, v in proceeding_labels.items() if k in current_ids}

#     # Save updated cache
#     with open(labels_cache_file, "wb") as f:
#         pickle.dump(proceeding_labels, f)

#     return proceeding_labels

def _generate_labels(text):
    return true_labels_generator(text)

def generate_and_cache_regex_labels(data, dir_for_cache, load_in_or_start_from_scratch, cache_manager=None):
    if dir_for_cache is None:
        dir_for_cache = cache_manager.get_cache_dir("labels")
    if not dir_for_cache.exists():
        dir_for_cache.mkdir(parents=True, exist_ok=True)
    labels_cache_file = dir_for_cache / "labels_cache.pkl"

    proceeding_labels = {}
    if load_in_or_start_from_scratch == "start_from_scratch":
        proceeding_labels = {}
    elif load_in_or_start_from_scratch == "load_in":
        if labels_cache_file.exists():
            with open(labels_cache_file, "rb") as f:
                print(f"Loading labels cache from {labels_cache_file}")
                proceeding_labels = pickle.load(f)

    combined = pd.concat([data['mag'], data['cdc']], ignore_index=True)
    current_ids = set(combined['proceeding_id'])

    # Only send uncached rows to workers
    uncached_mask = ~combined['proceeding_id'].isin(proceeding_labels)
    uncached = combined[uncached_mask]

    if not uncached.empty:
        ids = list(uncached['proceeding_id'])
        texts = list(uncached['text'] if 'text' in uncached.columns else [''] * len(uncached))

        n_workers = multiprocessing.cpu_count()
        chunksize = max(1, len(ids) // n_workers)

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(tqdm(
                pool.map(_generate_labels, texts, chunksize=chunksize),
                total=len(ids)
            ))

        # Results come back in same order as input — zip is safe
        proceeding_labels.update(zip(ids, results))

    if len(proceeding_labels) > len(current_ids):
        proceeding_labels = {k: v for k, v in proceeding_labels.items() if k in current_ids}
    with open(labels_cache_file, "wb") as f:
        pickle.dump(proceeding_labels, f)
    return proceeding_labels