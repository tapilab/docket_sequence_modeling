from collections import Counter
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from collections import defaultdict

from data.sequence_builder import get_proceeding_sequence_for_defendant
from config import get_model_config

def analyze_label_frequencies(cached_labels, title="LABEL FREQUENCY ANALYSIS"):
    """Analyze and print the frequency of each label in the cached labels dictionary."""
    # Count all labels
    all_labels = []
    for proceeding_id, labels in tqdm(cached_labels.items(), desc="Counting labels"):
        all_labels.extend(labels)

    label_counts = Counter(all_labels)
    total_proceedings = len(cached_labels)

    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"Total proceedings: {total_proceedings:,}")
    print(f"Total label instances: {sum(label_counts.values()):,}")
    print("=" * 80)

    # Sort by count (descending)
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Label':<50} {'Count':<10} {'Percentage':<12}")
    print("-" * 72)

    for label, count in sorted_labels:
        percentage = (count / total_proceedings) * 100
        print(f"{label:<50} {count:<10,} {percentage:<12.2f}%")

def analyze_label_frequencies_by_cases(cases, cached_labels, data, case_sequences_dict, title="LABEL FREQUENCY ANALYSIS BY CASES"):
    """
    Analyze label frequencies as fraction of total cases (not proceedings).
    Shows what percentage of cases have at least one instance of each label.
    """
    model_configuration = get_model_config()
    
    # Track which cases have each label
    cases_with_label = {label: set() for label in model_configuration.LABEL_NAMES}
    
    # Process each case
    for mag_num, case_id, def_id in tqdm(cases, desc="Processing cases"):
        case_key = f"{mag_num}_{case_id}_{def_id}"
        
        # Get all labels for this case
        case_labels = set()
        for case in get_proceeding_sequence_for_defendant(def_id, mag_num, case_id, data):
            for proc in case['proceedings']:
                proc_id = proc.get('proceeding_id')
                if proc_id and proc_id in cached_labels:
                    case_labels.update(cached_labels[proc_id])
        
        # Add this case to the set for each label it contains
        for label in case_labels:
            if label in cases_with_label:
                cases_with_label[label].add(case_key)
    
    total_cases = len(cases)
    
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"Total cases analyzed: {total_cases:,}")
    print("=" * 80)
    
    # Sort by count (descending)
    label_case_counts = [(label, len(case_set)) for label, case_set in cases_with_label.items()]
    sorted_labels = sorted(label_case_counts, key=lambda x: x[1], reverse=True)
    
    print(f"{'Label':<50} {'Cases':<10} {'Percentage':<12}")
    print("-" * 72)
    
    for label, count in sorted_labels:
        percentage = (count / total_cases) * 100 if total_cases > 0 else 0
        print(f"{label:<50} {count:<10,} {percentage:<12.2f}%")

def analyze_sequence_labels(sequences, title="SEQUENCE LABEL FREQUENCY ANALYSIS"):
    """Analyze label frequencies from training sequences."""
    model_configuration = get_model_config()
    
    all_labels = []
    total_proceedings = 0

    for seq, labels_seq in tqdm(sequences, desc="Counting labels"):
        total_proceedings += len(labels_seq)
        for label_vec in labels_seq:
            for i, val in enumerate(label_vec):
                if val == 1 and i < len(model_configuration.LABEL_NAMES):
                    all_labels.append(model_configuration.LABEL_NAMES[i])

    label_counts = Counter(all_labels)

    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"Total proceedings: {total_proceedings:,}")
    print(f"Total label instances: {sum(label_counts.values()):,}")
    print("=" * 80)

    # Sort by count (descending)
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Label':<50} {'Count':<10} {'Percentage':<12}")
    print("-" * 72)

    for label, count in sorted_labels:
        percentage = (count / total_proceedings) * 100
        print(f"{label:<50} {count:<10,} {percentage:<12.2f}%")

def get_labels_for_cases(cases, cached_labels, data):
    """Extract labels for a given list of cases."""
    return {  # Creates and returns dict in one expression
        proc_id: cached_labels[proc_id]
        for mag_num, case_id, def_id in tqdm(cases, desc="Processing cases")
        for case in get_proceeding_sequence_for_defendant(def_id, mag_num, case_id, data)
        for proc in case['proceedings']
        for proc_id in [proc.get('proceeding_id')]
        if proc_id in cached_labels
    }