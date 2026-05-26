from collections import Counter
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from collections import defaultdict

from data.sequence_builder import get_proceeding_sequence_for_defendant
from config import get_model_config

# Label Frequency Analysis

def analyze_raw_label_frequencies_combined(train_sequences_and_labels, val_sequences_and_labels):
    """Analyze label frequencies from raw sequences (before transformation)."""
    model_configuration = get_model_config()
    
    print("\n" + "=" * 80)
    print("RAW LABEL FREQUENCY ANALYSIS")
    print("=" * 80)
    
    # Combine all sequences
    all_sequences = train_sequences_and_labels + val_sequences_and_labels
    
    # Count all labels
    label_counts = Counter()
    total_proceedings = 0
    
    for seq, labels_seq in tqdm(all_sequences, desc="Counting raw labels"):
        total_proceedings += len(labels_seq)
        for label_set in labels_seq:
            # label_set is a list of string labels
            for label in label_set:
                if label not in model_configuration.EXCLUDED_LABELS:
                    label_counts[label] += 1
    
    print(f"Total proceedings: {total_proceedings:,}")
    print(f"Total label instances: {sum(label_counts.values()):,}")
    print(f"Unique labels: {len(label_counts)}")
    print("=" * 80)
    
    # Sort by count (descending)
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Label':<50} {'Count':<10} {'Percentage':<12}")
    print("-" * 72)
    
    for label, count in sorted_labels:
        percentage = (count / total_proceedings) * 100 if total_proceedings > 0 else 0
        print(f"{label:<50} {count:<10,} {percentage:<12.2f}%")
    
    return label_counts

def print_label_distribution_by_court(
    data,
    outcome_labels,
    label_names,
    mag_key: str = "mag",
    cdc_key: str = "cdc",
    proceeding_col: str = "proceeding_id",
):
    # Initialize counters
    label_court_counts = defaultdict(lambda: {mag_key: 0, cdc_key: 0})

    # Helper: fetch labels for a proceeding_id with light type-flex
    def _get_labels(pid):
        if pid in outcome_labels:
            return outcome_labels[pid]
        # try int and str fallbacks
        try:
            pid_int = int(pid)
            if pid_int in outcome_labels:
                return outcome_labels[pid_int]
        except Exception:
            pass
        pid_str = str(pid)
        return outcome_labels.get(pid_str, None)

    # Process each court type
    for court_key in (mag_key, cdc_key):
        if court_key not in data:
            continue
        df = data[court_key]
        if proceeding_col not in df.columns:
            continue

        for _, row in df.iterrows():
            proceeding_id = row[proceeding_col]
            labels = _get_labels(proceeding_id)
            if not labels:
                continue
            for label in labels:
                if label in label_names:
                    label_court_counts[label][court_key] += 1

    # Print results
    print("LABEL DISTRIBUTION BY COURT TYPE")
    print("=" * 80)
    header = f"{'Label':<50} {mag_key.upper():<8} {cdc_key.upper():<8} {'Total':<8} {(mag_key.upper()+' %'):<8} {(cdc_key.upper()+' %'):<8}"
    print(header)
    print("-" * 80)

    for label in label_names:
        mag_count = label_court_counts[label][mag_key]
        cdc_count = label_court_counts[label][cdc_key]
        total_count = mag_count + cdc_count
        if total_count > 0:
            mag_pct = (mag_count / total_count) * 100
            cdc_pct = (cdc_count / total_count) * 100
            print(
                f"{label:<50} "
                f"{mag_count:<8,} {cdc_count:<8,} {total_count:<8,} "
                f"{mag_pct:<8.1f} {cdc_pct:<8.1f}"
            )

    return label_court_counts

def analyze_raw_label_frequencies_singular(sequences_and_labels, title):
   """Analyze label frequencies from raw string labels"""
   print("=" * 80)
   print(title)
   print("=" * 80)
   
   all_labels = []
   total_proceedings = 0
   
   for seq, labels_seq in sequences_and_labels:
       total_proceedings += len(labels_seq)
       for step_labels in labels_seq:
           all_labels.extend(step_labels)  # step_labels is a list of strings
   
   from collections import Counter
   label_counts = Counter(all_labels)
   
   print(f"Total proceedings: {total_proceedings:,}")
   print(f"Total label instances: {len(all_labels):,}")
   print("=" * 80)
   
   sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
   
   print(f"{'Label':<50} {'Count':<10} {'Percentage':<12}")
   print("-" * 72)
   
   for label, count in sorted_labels:
       percentage = (count / total_proceedings) * 100 if total_proceedings > 0 else 0
       print(f"{label:<50} {count:<10,} {percentage:<12.2f}%")