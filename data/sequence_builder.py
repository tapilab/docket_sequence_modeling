# Python File that contains functions related to building and creating the sequences

import pandas as pd
import numpy as np
from tqdm import tqdm
from regex_labels.label_generator_and_cache_manager import true_labels_generator_optimization
from collections import defaultdict

## SECTION Create a way to create sequences faster by caching and using indexes
class SequenceCacheBuilder:
    def __init__(self, data):
        self.data = data
        self._build_indexes()
        self._build_charge_lookups()
        self._build_defendant_lookups()
        self._build_proceeding_lookups()
        self.build_precomputed_defendant_prior_cases()
        
    def _build_indexes(self):
        #create dictionary based indexes for faster lookup

        # First, we will index proceedings by proceeding_id
        self.mag_by_proc_id = {}
        if 'mag' in self.data and not self.data['mag'].empty:
            for _, row in self.data['mag'].iterrows():
                proc_id = row.get('proceeding_id')
                if proc_id:
                    self.mag_by_proc_id[proc_id] = row.to_dict()

        self.cdc_by_proc_id = {}
        if 'cdc' in self.data and not self.data['cdc'].empty:
            for _, row in self.data['cdc'].iterrows():
                proc_id = row.get('proceeding_id')
                if proc_id:
                    self.cdc_by_proc_id[proc_id] = row.to_dict()
        
        # Index dockets by linker (whether mag_num or case_id)
        self.mag_dockets_by_mag_num = defaultdict(list)
        if 'mag_dockets' in self.data and not self.data['mag_dockets'].empty:
            for _, row in self.data['mag_dockets'].iterrows():
                mag_num = row.get('mag_num')
                if mag_num:
                    self.mag_dockets_by_mag_num[mag_num].append(row.to_dict())
        
        self.cdc_dockets_by_case_id = defaultdict(list)
        if 'cdc_dockets' in self.data and not self.data['cdc_dockets'].empty:
            for _, row in self.data['cdc_dockets'].iterrows():
                case_id = row.get('case_id')
                if case_id:
                    self.cdc_dockets_by_case_id[case_id].append(row.to_dict())
        
        # Index combiend_dockets by defendant_id
        self.cases_by_defendant = defaultdict(list)
        if 'combined_dockets' in self.data and not self.data['combined_dockets'].empty:
            for _, row in self.data['combined_dockets'].iterrows():
                def_id = row.get('defendant_id')
                if def_id:
                    self.cases_by_defendant[def_id].append(row.to_dict())
        
        # Print summary of indexes built
        print("Indexes built:")
        print(f"Indexed {len(self.mag_by_proc_id)} MAG proceedings")
        print(f"Indexed {len(self.cdc_by_proc_id)} CDC proceedings")
        print(f"Indexed {len(self.mag_dockets_by_mag_num)} MAG docket groups")
        print(f"Indexed {len(self.cdc_dockets_by_case_id)} CDC docket groups")
        print(f"Indexed {len(self.cases_by_defendant)} defendants")
    
    def _build_charge_lookups(self):
        # Index charges by charge_id for name lookup
        self.charge_names = {}
        if 'charges' in self.data and not self.data['charges'].empty:
            for idx, row in self.data['charges'].iterrows():
                charge_id = str(row.get('charge_id', ''))
                if charge_id:
                    self.charge_names[charge_id] = row.get('charge_name', f"Charge {charge_id}")
        
        # Index mag_charges by mag_num
        self.mag_charges_by_mag_num = defaultdict(list)
        if 'mag_charges' in self.data and not self.data['mag_charges'].empty:
            for _, row in self.data['mag_charges'].iterrows():
                mag_num = row.get('mag_num')
                if mag_num:
                    self.mag_charges_by_mag_num[mag_num].append(row.to_dict())
        
        # Index cdc_charges by case_id
        self.cdc_charges_by_case_id = defaultdict(list)
        if 'cdc_charges' in self.data and not self.data['cdc_charges'].empty:
            for _, row in self.data['cdc_charges'].iterrows():
                case_id = row.get('case_id')
                if case_id:
                    self.cdc_charges_by_case_id[case_id].append(row.to_dict())
        
        # Print summary of charge lookups built
        print("Charge lookups built:")
        print(f"Indexed {len(self.charge_names)} charge names")
        print(f"Indexed {len(self.mag_charges_by_mag_num)} MAG charge groups")
        print(f"Indexed {len(self.cdc_charges_by_case_id)} CDC charge groups")

    def _build_defendant_lookups(self):
        #Build defendant lookup by defendant_id
        self.defendants_by_id = {}
        if 'defendants' in self.data and not self.data['defendants'].empty:
            for _, row in self.data['defendants'].iterrows():
                def_id = row.get('defendant_id')
                if def_id:
                    self.defendants_by_id[def_id] = row.to_dict()
        
        # Print summary of defendant lookups built
        print("Defendant lookups built:")
        print(f"Indexed {len(self.defendants_by_id)} defendants")
    
    def _build_proceeding_lookups(self):
        # Build proceeding lookups by docket

        # Build MAG proceeding lookup by mag_num
        self.mag_proceedings_by_mag_num = defaultdict(list)
        if 'mag' in self.data and not self.data['mag'].empty:
            for proc_id, row_dict in self.mag_by_proc_id.items():
                mag_num = row_dict.get('mag_num')
                if mag_num:
                    self.mag_proceedings_by_mag_num[mag_num].append(row_dict)
        
        # Build CDC proceeding lookup by case_id
        self.cdc_proceedings_by_case_id = defaultdict(list)
        if 'cdc' in self.data and not self.data['cdc'].empty:
            for proc_id, row_dict in self.cdc_by_proc_id.items():
                case_id = row_dict.get('case_id')
                if case_id:
                    self.cdc_proceedings_by_case_id[case_id].append(row_dict)
        
        # Build summary of proceeding lookups built
        print("Proceeding lookups built:")
        print(f"Indexed {len(self.mag_proceedings_by_mag_num)} MAG proceeding groups")
        print(f"Indexed {len(self.cdc_proceedings_by_case_id)} CDC proceeding groups")

    def build_precomputed_defendant_prior_cases(self):
        # For each defendant, precomute the prior case information for each defendant
        
        # This means the following
        # For each defendant, create a sorted list of their cases with dates

        self.defendant_case_timeline = {}

        for def_id, cases in tqdm(self.cases_by_defendant.items(), desc="Precomputing defendant case timelines"):
            # Get unique cases with dates
            case_dates = []
            seen_cases = set()
            for case in cases:
                mag_num = case.get('mag_num')
                case_id = case.get('case_id')

                mag_str = str(mag_num) if pd.notna(mag_num) else None
                case_str = str(case_id) if pd.notna(case_id) else None
                case_key = (mag_str, case_str)
                
                if case_key in seen_cases:
                    continue
                seen_cases.add(case_key)

                # Determine case start date
                mag_date = pd.to_datetime(case.get('mag_date_of_first_proceeding'), errors='coerce')
                cdc_date = pd.to_datetime(case.get('cdc_date_of_first_proceeding'), errors='coerce')
                
                if pd.notna(mag_date) and pd.notna(cdc_date):
                    case_start_date = min(mag_date, cdc_date)
                elif pd.notna(mag_date):
                    case_start_date = mag_date
                elif pd.notna(cdc_date):
                    case_start_date = cdc_date
                else:
                    case_start_date = None
                
                #Pre-aggregate charges for the case
                case_charges = {}
                if pd.notna(mag_num):
                    for charge_row in self.mag_charges_by_mag_num.get(mag_num, []):
                        charge_id = charge_row.get('charge_id')
                        if charge_id:
                            charge_id_str = str(charge_id)
                            if charge_id_str not in case_charges:
                                case_charges[charge_id_str] = 0
                            case_charges[charge_id_str] += charge_row.get('count', 0)
                
                if pd.notna(case_id):
                    for charge_row in self.cdc_charges_by_case_id.get(case_id, []):
                        charge_id = charge_row.get('charge_id')
                        if charge_id:
                            charge_id_str = str(charge_id)
                            if charge_id_str not in case_charges:
                                case_charges[charge_id_str] = 0
                            case_charges[charge_id_str] += charge_row.get('count', 0)
                
                case_dates.append({
                    'case_key': case_key,
                    'start_date': case_start_date,
                    'charges': case_charges
                })
            # Sort cases by date
            case_dates.sort(key=lambda x: x['start_date'] if pd.notna(x['start_date']) else pd.Timestamp.max)
            self.defendant_case_timeline[def_id] = case_dates

    
    def get_defendant_prior_cases(self, defendant_id: str, current_date: pd.Timestamp, current_mag_num=None, current_case_id=None) -> dict:
        if defendant_id not in self.defendant_case_timeline:
            return {'prior_case_count': 0, 'all_prior_charges': {}}
        
        if pd.isna(current_date):
            return {'prior_case_count': 0, 'all_prior_charges': {}}
        
        #prior case count
        prior_case_count = 0
        all_prior_charges = {}

        for case_info in self.defendant_case_timeline[defendant_id]:
            case_key = case_info['case_key']
            case_start = case_info['start_date']
            if case_key == (current_mag_num, current_case_id):
                continue
            if pd.notna(case_start) and case_start < current_date:
                prior_case_count += 1
                for charge_id, count in case_info['charges'].items():
                    if charge_id not in all_prior_charges:
                        all_prior_charges[charge_id] = {'count': 0}
                    all_prior_charges[charge_id]['count'] += count
        return {
            'prior_case_count': prior_case_count,
            'all_prior_charges': all_prior_charges
        }


## SECTION Defining race and gender mapping constants
RACE_MAP = {'B': 0, 'W': 1, 'H': 2, 'A': 3, 'I': 4, 'U': 5}
GENDER_MAP = {'M': 0, 'F': 1}

## SECTION Assemble proceeding dictionary with defendant demographics, charges, and case information

def create_dict_for_proceeding(proceeding_id: str, proceeding_type: str, cache) -> dict:
    if proceeding_type == 'mag':
        row = cache.mag_by_proc_id.get(proceeding_id)
        link = 'mag_num'
        dockets_lookup = cache.mag_dockets_by_mag_num
        charges_lookup = cache.mag_charges_by_mag_num
    elif proceeding_type == 'cdc':
        row = cache.cdc_by_proc_id.get(proceeding_id)
        link = 'case_id'
        dockets_lookup = cache.cdc_dockets_by_case_id
        charges_lookup = cache.cdc_charges_by_case_id
    else:
        raise ValueError(f"Unknown type {proceeding_type}")
    
    if not row:
        return {}
    
    proceeding_date = pd.to_datetime(row.get('date', pd.NaT), errors='coerce')
    out = {
        'proceeding_id': proceeding_id,
        'type': proceeding_type,
        'date': proceeding_date,
        'text': row.get('text',''),
        'docket_id': row.get(link)
    }

    #date month year
    if pd.notna(proceeding_date):
        out['date_month_year'] = f"{proceeding_date.year}-{proceeding_date.month:02d}"  # 2023-01, 2023-12
    else:
        out['date_month_year'] = None
    # docket extraction
    did = None
    docket_id = out['docket_id']
    if docket_id and pd.notna(docket_id):
        docket_rows = dockets_lookup.get(docket_id, [])
        if docket_rows:
            docket_row = docket_rows[0]
            did = docket_row.get('defendant_id')
            og_bond = docket_row.get('original_bond')
            sum_of_bonds = docket_row.get('sum_of_bonds')
            if og_bond:
                out['original_bond'] = og_bond
            if sum_of_bonds:
                out['sum_of_bonds'] = sum_of_bonds
    
    if did:
        def_row = cache.defendants_by_id.get(did)
        if def_row:
            out.update({
                'defendant_id': did,
                'race_id': RACE_MAP.get(def_row.get('race','U'),5),
                'gender_id': GENDER_MAP.get(def_row.get('sex'),2),
            })
            dob = pd.to_datetime(def_row.get('dob'), format='%Y', errors='coerce')
            if pd.notna(out['date']) and pd.notna(dob):
                age = out['date'].year - dob.year
                out['age'] = age if 0<=age<=110 else None
    
    out['charges'] = {}
    if docket_id and pd.notna(docket_id):
        for charge_row in charges_lookup.get(docket_id, []):
            cid = charge_row.get('charge_id')
            if cid:
                cid_str = str(cid)
                charge_name = cache.charge_names.get(cid_str, f"Charge {cid_str}")
                out['charges'][cid] = {
                    'name': charge_name,
                    'count': charge_row.get('count', 1),
                    'bond': charge_row.get('bond')
                }
    out['magistrate_section'] = row.get('magistrate_section','unknown')
    return out

## Retrieves chronologically sorted proceeding sequences for a defendant across magistrate and CDC cases
def get_proceeding_sequence_for_defendant(def_id: str, mag_num: int, case_id: int, cache):
    proceedings_by_case = {}

    defendant_cases = cache.cases_by_defendant.get(def_id, [])

    for case_row in defendant_cases:
        row_mag = case_row.get('mag_num')
        row_case = case_row.get('case_id')
        
        #check if this row matches the target criterion
        mag_match = (pd.isna(mag_num) and pd.isna(row_mag)) or (row_mag == mag_num)
        case_match = (pd.isna(case_id) and pd.isna(row_case)) or (row_case == case_id)

        if not(mag_match and case_match):
            continue

        mag_val = row_mag if pd.notna(row_mag) else None
        case_val = row_case if pd.notna(row_case) else None
        case_key = (mag_val, case_val)

        if case_key in proceedings_by_case:
            continue

        proceedings_by_case[case_key] = {
            'case_info': {'mag_num': mag_val, 'case_id': case_val},
            'proceedings': []
        }

        # Add MAG proceedings
        if mag_val is not None:
            for proc_row in cache.mag_proceedings_by_mag_num.get(mag_val, []):
                proc_id = proc_row.get('proceeding_id')
                if proc_id:
                    proc_dict = create_dict_for_proceeding(proc_id, 'mag', cache)
                    if proc_dict:
                        proceedings_by_case[case_key]['proceedings'].append(proc_dict)
        
        # Add CDC proceedings
        if case_val is not None:
            for proc_row in cache.cdc_proceedings_by_case_id.get(case_val, []):
                proc_id = proc_row.get('proceeding_id')
                if proc_id:
                    proc_dict = create_dict_for_proceeding(proc_id, 'cdc', cache)
                    if proc_dict:
                        proceedings_by_case[case_key]['proceedings'].append(proc_dict)
        
        #Sort proceedings by date
        proceedings_by_case[case_key]['proceedings'].sort(
            key=lambda x: x['date'] if pd.notna(x.get('date')) else pd.Timestamp.min
        )

    matched_case_ids = {
        case_val
        for (mag_val, case_val) in proceedings_by_case
        if mag_val is not None and case_val is not None
    }

    keys_to_remove = []
    for (mag_val, case_val) in list(proceedings_by_case.keys()):
        if mag_val is None and case_val in matched_case_ids:
            keys_to_remove.append((mag_val, case_val))

    for key in keys_to_remove:
        del proceedings_by_case[key]

    all_cases = list(proceedings_by_case.values())
    all_cases.sort(
        key=lambda x: x['proceedings'][0]['date'] if x['proceedings'] and pd.notna(x['proceedings'][0].get('date')) else pd.Timestamp.max
    )   
    return all_cases


def create_enhanced_training_sequences_cached(case_sample, data, cached_labels, skip_duplicate_labels = False, cache = None):
    if cache is None:
        print("Building sequence cache...")
        cache = SequenceCacheBuilder(data)    
    
    sequences = []
    label_sequences_raw = []  # Store raw string labels for MLB
    
    total_skipped = 0
    total_proceedings = 0
    
    for mag_num, case_id, def_id in tqdm(case_sample, desc="Processing cases"):
        if case_id == "556090":
            print(mag_num, case_id, def_id) 
        case_seq = get_proceeding_sequence_for_defendant(def_id, mag_num, case_id, cache)
        if case_id == "556090":
            print(f"Case sequence for defendant {def_id}, mag_num {mag_num}, case_id {case_id}:")
            for case in case_seq:
                print(f"  Case: {case['case_info']}")
                for proc in case['proceedings']:
                    print(f"    Proceeding: {proc['proceeding_id']}, Date: {proc['date']}, Type: {proc['type']}")
        for case in case_seq:
            proceedings = case['proceedings']
            if not proceedings:
                continue

            proceedings_with_past = []
            labels_per_step_raw = []
            for i, proc in enumerate(proceedings):
                total_proceedings += 1
                proc_id = proc.get('proceeding_id','')
                proc_date = proc.get('date', pd.NaT)
                # Get raw string labels (instead of encoded binary)
                true_labels = true_labels_generator_optimization(
                    proc.get('text',''),
                    proc_id,
                    cached_labels
                )

                if skip_duplicate_labels and i > 0:
                    prev_labels = labels_per_step_raw[-1] if labels_per_step_raw else []
                    current_labels_set = set(true_labels)
                    prev_labels_set = set(prev_labels)
                    if current_labels_set == prev_labels_set and current_labels_set:
                        total_skipped += 1
                        continue
                
                prior_info = cache.get_defendant_prior_cases(def_id, proc_date, current_mag_num = mag_num, current_case_id = case_id)
                enhanced = proc.copy()
                enhanced.update(prior_info)
                proceedings_with_past.append(enhanced)
                labels_per_step_raw.append(true_labels)

            if len(proceedings_with_past) >= 2 and len(labels_per_step_raw) >= 2:
                has_meaningful_next_labels = any(
                    len(labels_per_step_raw[i+1]) > 0  # Check if list of labels is not empty
                    for i in range(len(labels_per_step_raw) - 1)
                )
                if has_meaningful_next_labels:
                    sequences.append(proceedings_with_past)
                    label_sequences_raw.append(labels_per_step_raw)
    
    if skip_duplicate_labels:
        skip_pct = (total_skipped / total_proceedings * 100) if total_proceedings > 0 else 0
        print("Duplicate label skipping:")
        print(f"Total proceedings processed: {total_proceedings:,}")
        print(f"Proceedings skipped (duplicate labels): {total_skipped:,}")
        print(f"Skip percentage: {skip_pct:.2f}%")

    return sequences, label_sequences_raw