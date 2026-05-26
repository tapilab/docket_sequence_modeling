import numpy as np
import random
from collections import Counter, defaultdict
from tqdm import tqdm
from typing import List, Tuple, Dict, Set, Any
import pandas as pd
from pathlib import Path
from notebook_helpers import extract_case_info_from_sequence


def count_label_frequencies(sequences_and_labels: List[Tuple[List[dict], List[List[str]]]]) -> Counter:
    label_counts = Counter()
    for _, labels_seq in tqdm(sequences_and_labels, desc="Counting labels"):
        for step_labels in labels_seq:
            label_counts.update(step_labels)
    return label_counts


def get_cases_by_label(sequences_and_labels: List[Tuple[List[dict], List[List[str]]]]) -> Dict[str, Set[int]]:
    label_to_cases = defaultdict(set)
    for case_idx, (_, labels_seq) in enumerate(tqdm(sequences_and_labels, desc="Indexing cases by label")):
        for step_labels in labels_seq:
            for label in step_labels:
                label_to_cases[label].add(case_idx)
    return dict(label_to_cases)

def create_balanced_dataset(
    sequences_and_labels: List[Tuple[List[dict], List[List[str]]]],
    low_frequency_threshold: int = 200,
    target_size: int = 1000,
    excluded_labels: Set[str] = None,
    random_seed: int = 42
) -> Tuple[List[Tuple[List[dict], List[List[str]]]], List[int], List[int]]:
    random.seed(random_seed)
    np.random.seed(random_seed)

    if excluded_labels is None:
        excluded_labels = {'NONE', 'NO_DECISION', 'NO_SENTENCE_WAS_IMPOSED'}

    print("\n" + "=" * 70)
    print("BALANCED DATASET CREATION")
    print("=" * 70)
    print(f"Total input cases: {len(sequences_and_labels):,}")
    print(f"Low frequency threshold: {low_frequency_threshold}")
    print(f"Target size: {target_size}")

    # Build label -> set of case indices (case-level counts throughout)
    label_to_cases = get_cases_by_label(sequences_and_labels)

    for excluded in excluded_labels:
        label_to_cases.pop(excluded, None)

    # Case-level counts: number of distinct cases that contain this label
    label_counts = {label: len(cases) for label, cases in label_to_cases.items()}

    low_freq_labels  = {label for label, count in label_counts.items() if count < low_frequency_threshold}
    high_freq_labels = {label for label, count in label_counts.items() if count >= low_frequency_threshold}

    print(f"\nLabel Distribution (case-level):")
    print(f"  - Low frequency labels (<{low_frequency_threshold}): {len(low_freq_labels)}")
    print(f"  - High frequency labels (>={low_frequency_threshold}): {len(high_freq_labels)}")

    print(f"\nLow Frequency Labels (<{low_frequency_threshold}):")
    for label in sorted(low_freq_labels):
        print(f"  {label}: {label_counts[label]:,} cases")

    print(f"\nHigh Frequency Labels (>={low_frequency_threshold}):")
    for label in sorted(high_freq_labels, key=lambda x: label_counts[x], reverse=True):
        print(f"  {label}: {label_counts[label]:,} cases")

    # ── Step 1: include ALL cases with low-frequency labels ────────────────
    # These labels have fewer than low_frequency_threshold cases total,
    # so we take everything available (use-all fallback).
    selected_indices = set()
    for label in low_freq_labels:
        selected_indices.update(label_to_cases[label])

    print(f"\nAfter including all low-frequency label cases: {len(selected_indices):,} cases")

    # ── Step 2: guarantee >= low_frequency_threshold CASES per high-freq label
    # For each high-frequency label that doesn't yet have enough cases in the
    # selected set, add the missing cases (shuffled). If the label has fewer
    # total cases than the threshold, all of them are added (use-all fallback).
    print(f"\nEnsuring >= {low_frequency_threshold} cases per high-frequency label...")
    boosted = []

    for label in sorted(high_freq_labels):
        already = len(label_to_cases[label] & selected_indices)
        need = low_frequency_threshold - already
        if need <= 0:
            continue

        boosted.append(label)
        candidates = list(label_to_cases[label] - selected_indices)
        random.shuffle(candidates)
        # Take as many as needed (or all available if fewer exist than needed)
        selected_indices.update(candidates[:need])

    if boosted:
        print(f"  Boosted {len(boosted)} label(s): {boosted}")
    else:
        print("  All high-frequency labels already met threshold.")

    print(f"After threshold guarantee: {len(selected_indices):,} cases")

    # ── Step 3: fill to target_size with random remaining cases ───────────
    if len(selected_indices) < target_size:
        remaining_indices = set(range(len(sequences_and_labels))) - selected_indices
        needed = target_size - len(selected_indices)
        available = min(needed, len(remaining_indices))
        additional_indices = random.sample(list(remaining_indices), available)
        selected_indices.update(additional_indices)
        print(f"Sampled {available:,} additional cases from remaining pool")
    elif len(selected_indices) > target_size:
        print(f"Note: Guarantees exceed target ({len(selected_indices):,} > {target_size:,}). Keeping all.")

    # ── Build final dataset ────────────────────────────────────────────────
    sampled_indices = sorted(selected_indices)
    sampled_cases   = [sequences_and_labels[i] for i in sampled_indices]
    sampled_lengths = [len(seq) for seq, _ in sampled_cases]

    balanced_label_to_cases = get_cases_by_label(sampled_cases)
    balanced_label_counts   = {label: len(cases) for label, cases in balanced_label_to_cases.items()}

    print(f"\n" + "=" * 70)
    print("FINAL BALANCED DATASET STATISTICS")
    print("=" * 70)
    print(f"Final dataset size: {len(sampled_cases):,} cases")
    total_proceedings = sum(len(labels_seq) for _, labels_seq in sampled_cases)
    print(f"Total proceedings: {total_proceedings:,}")

    print(f"\nLabel distribution in balanced dataset (case-level):")
    print(f"{'Label':<55} {'Original':>10} {'Balanced':>10} {'Ratio':>8} {'Meets?':>7}")
    print("-" * 93)

    for label in sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True):
        orig     = label_counts[label]
        balanced = balanced_label_counts.get(label, 0)
        ratio    = balanced / orig if orig > 0 else 0
        marker   = " *" if label in low_freq_labels else ""
        meets    = "YES" if balanced >= min(low_frequency_threshold, orig) else "NO "
        print(f"{label:<55} {orig:>10,} {balanced:>10,} {ratio:>7.1%}{marker}  {meets}")

    print("\n* = Low frequency label (fewer than threshold available, fully included)")

    return sampled_cases, sampled_lengths, sampled_indices, list(low_freq_labels)
    
def export_balanced_cases_to_xlsx(
    sampled_cases: List[Tuple[List[dict], List[List[str]]]],
    sampled_indices: List[int],
    output_filepath: str,
    low_freq_labels: List[str] = None,
    court_data: Dict[str, Any] = None
):
    """
    Export balanced dataset to Excel with label frequency information.
    
    Args:
        sampled_cases: List of (sequence, label_sequence) tuples
        sampled_indices: List of original indices
        output_filepath: Path to save the xlsx file
        low_freq_labels: Optional list of low-frequency label names for marking
    """
    import pandas as pd
    from pathlib import Path
    
    if low_freq_labels is None:
        low_freq_labels = []
    low_freq_set = set(low_freq_labels)
    
    output_data = []
    for seq_idx, (sequence, label_sequence) in enumerate(tqdm(sampled_cases, desc="Processing balanced cases")):
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
                low_freq_labels_in_step = [l for l in labels if l in low_freq_set]
                has_low_freq = len(low_freq_labels_in_step) > 0
            else:
                labels_str = 'NO_LABELS'
                label_count = 0
                has_low_freq = False
                low_freq_labels_in_step = []
            
            output_data.append({
                'sequence_index': seq_idx,
                'original_index': sampled_indices[seq_idx] if seq_idx < len(sampled_indices) else -1,
                'proceeding_index': proc_idx,
                'mag_num': mag_num if pd.notna(mag_num) else '',
                'case_id': case_id if pd.notna(case_id) else '',
                'defendant_id': defendant_id if pd.notna(defendant_id) else '',
                'proceeding_id': proc_id,
                'date': proc.get('date'),
                'date_month_year': proc.get('date_month_year', ''),
                'proceeding_type': proc.get('type'),
                'text': proc.get('text', '')[:1000] if proc.get('text') else '',
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
                'sequence_length': len(sequence),
                'has_low_freq_label': has_low_freq,
                'low_freq_labels': '|'.join(low_freq_labels_in_step) if low_freq_labels_in_step else ''
            })
    
    output_df = pd.DataFrame(output_data)
    output_df = output_df.sort_values(['defendant_id', 'date']).reset_index(drop=True)
    
    output_filepath = Path(output_filepath)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_excel(output_filepath, index=False)
    
    print(f"Exported balanced dataset to: {output_filepath}")
    print(f"  - {len(output_df)} rows")