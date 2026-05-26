
import pandas as pd
from pathlib import Path
import numpy as np
import random
from collections import Counter, defaultdict
from tqdm import tqdm
from typing import List, Tuple, Dict, Set, Any
import pandas as pd
from pathlib import Path
from notebook_helpers import extract_case_info_from_sequence

def export_all_cases_to_xlsx(
    sequences_and_labels: List[Tuple[List[dict], List[List[str]]]],
    output_filepath: str,
    court_data: Dict[str, Any] = None
):
    output_data = []
    for seq_idx, (sequence, label_sequence) in enumerate(tqdm(sequences_and_labels, desc="Processing cases")):
        if not sequence:
            continue
        

        case_info = extract_case_info_from_sequence(sequence, court_data=court_data)
        mag_num = case_info.get('mag_num', '')
        case_id = case_info.get('case_id', '')
        defendant_id = case_info.get('defendant_id', '')

        for proc_idx, (proc, labels) in enumerate(zip(sequence, label_sequence)):
            proc_id = proc.get('proceeding_id')

            charges_dict = proc.get('charges', {})
            charge_names = [charge_info.get('name', '') for charge_info in charges_dict.values()]
            charge_names_str = ' | '.join(charge_names) if charge_names else ''

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
                'sequence_length': len(sequence),
            })

    output_df = pd.DataFrame(output_data)
    output_df = output_df.sort_values(['defendant_id', 'date']).reset_index(drop=True)

    output_filepath = Path(output_filepath)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_excel(output_filepath, index=False)

    print(f"Exported all cases to: {output_filepath}")
    print(f"  - {len(output_df):,} rows")
    print(f"  - {output_df['sequence_index'].nunique():,} cases")