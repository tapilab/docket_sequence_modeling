import pickle
import pandas as pd
from pathlib import Path

def extract_case_identifier_from_sequences(sequences, label_sequences_raw):
    case_identifiers = []
    for seq, label_seq in zip(sequences, label_sequences_raw):
        if not seq:
            continue
        # Get identifiers from first proceeding in sequence
        first_proc = seq[0]
        mag_num = first_proc.get('mag_num', '')
        case_id = first_proc.get('case_id', '')
        defendant_id = first_proc.get('defendant_id', '')
        case_identifiers.append((mag_num, case_id, defendant_id))
    return case_identifiers

def save_case_identifiers(case_identifiers, output_path):
    output_path = Path(output_path)
    # Save as pickle for exact comparison
    pkl_path = output_path.with_suffix('.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(case_identifiers, f)
    print(f"Saved case IDs to: {pkl_path}")

    # Save as CSV for human inspection
    csv_path = output_path.with_suffix('.csv')
    df = pd.DataFrame(case_identifiers, columns=['mag_num', 'case_id', 'defendant_id'])
    df['sequence_index'] = range(len(df))
    df.to_csv(csv_path, index=False)
    print(f"Saved case IDs (readable) to: {csv_path}")
    
    return pkl_path, csv_path

def load_case_identifiers(pkl_path):
    with open(pkl_path, 'rb') as f:
        case_identifiers = pickle.load(f)
    return case_identifiers

def compare_case_identifiers(case_ids1, case_ids2):
    set1 = set(case_ids1)
    set2 = set(case_ids2)
    
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1
    in_both_sets = set1 & set2
    print(f"Cases in both sets: {len(in_both_sets)}")
    print(f"Cases only in first set: {len(only_in_1)}")
    print(f"Cases only in second set: {len(only_in_2)}")

    if case_ids1 == case_ids2:
        print("The case identifier lists are identical.")
        match_status = 'identical'
    elif set1 == set2:
        print("The case identifier lists contain the same cases but in different orders.")
        match_status = 'same_cases_different_order'
    else:
        print("The case identifier lists differ.")
        match_status = 'different'
    return {
        'only_in_1': only_in_1,
        'only_in_2': only_in_2,
        'only_in_1_count': len(only_in_1),
        'only_in_2_count': len(only_in_2),
        'in_both_sets_count': len(in_both_sets),
        'in_both_sets': in_both_sets,
        'match_status': match_status,
        'total_in_1': len(case_ids1),
        'total_in_2': len(case_ids2)
    }


def load_and_compare_case_identifiers(pkl_path1, pkl_path2):
    case_ids1 = load_case_identifiers(pkl_path1)
    case_ids2 = load_case_identifiers(pkl_path2)
    return compare_case_identifiers(case_ids1, case_ids2)
    



