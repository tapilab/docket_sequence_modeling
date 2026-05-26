TRACKING_LABELS = [
    'TRIAL_VERDICT_REACHED',
    'SENTENCE_THAT_DOES_NOT_INCLUDE_JAIL_TIME_WAS_IMPOSED'
]

def count_specific_labels_in_cache(cached_outcome_labels: dict, label_names: list = None) -> dict:
    if label_names is None:
        label_names = TRACKING_LABELS
    results = {label: 0 for label in label_names}
    for proc_id, labels in cached_outcome_labels.items():
        if labels:
            for label in labels:
                if label in results:
                    results[label] += 1
    return results

def count_labels_in_sequences(sequences, label_sequences_raw, label_names):
    seen_proc_ids = {label: set() for label in label_names}
    results = {label: 0 for label in label_names}

    for seq, label_seq in zip(sequences, label_sequences_raw):
        for step, labels in zip(seq, label_seq):
            proc_id = step['proceeding_id']
            if labels:
                for label in labels:
                    if label in results and proc_id not in seen_proc_ids[label]:
                        results[label] += 1
                        seen_proc_ids[label].add(proc_id)
    return results

def print_tracking_checkpoint(counts: dict, checkpoint_name: str, extra_info: str = ""):
    print("\\n" + "=" * 80)
    print(f"🔍 LABEL TRACKING: {checkpoint_name}")
    print("=" * 80)
    if extra_info:
        print(extra_info)
    print("-" * 80)
    print(f"{'Label':<55} {'Count':>10}")
    print("-" * 80)
    for label, count in counts.items():
        print(f"{label:<55} {count:>10,}")
    print("=" * 80 + "\\n")
    return counts
def count_labels_in_sequences_by_case(sequences, label_sequences_raw, label_names):
    """Count how many distinct cases contain each label (case-level, not proceeding-level)."""
    seen_case_indices = {label: set() for label in label_names}
    results = {label: 0 for label in label_names}

    for case_idx, (seq, label_seq) in enumerate(zip(sequences, label_sequences_raw)):
        for labels in label_seq:
            if labels:
                for label in labels:
                    if label in results and case_idx not in seen_case_indices[label]:
                        results[label] += 1
                        seen_case_indices[label].add(case_idx)

    return results