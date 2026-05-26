import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from config import us_states  # Import the state list from config

# Python file that has functions related to filtering and processing the data

# SECTION Filtering the data to retain only magistrate/CDC rows for defendants with full demographics.

def append_prefix_to_proceeding_ids(data: dict) -> dict:
    for key in ['mag', 'cdc']:
        if key in data and not data[key].empty:
            data[key]['proceeding_id'] = f"{key}_" + data[key]['proceeding_id'].astype(str)
    return data

def filter_proceedings(data):
    defs = data['defendants']
    valid = defs.dropna(subset=['name','race','sex','dob'])['defendant_id']
    mag_nums = data['mag_dockets'][data['mag_dockets']['defendant_id'].isin(valid)]['mag_num']
    case_ids = data['cdc_dockets'][data['cdc_dockets']['defendant_id'].isin(valid)]['case_id']
    data['mag'] = data['mag'][data['mag']['mag_num'].isin(mag_nums)]
    data['cdc'] = data['cdc'][data['cdc']['case_id'].isin(case_ids)]
    data['mag_dockets'] = data['mag_dockets'][data['mag_dockets']['defendant_id'].isin(valid)]
    data['cdc_dockets'] = data['cdc_dockets'][data['cdc_dockets']['defendant_id'].isin(valid)]
    return data

# SECTION Removing Multi-Defendant Cases

## Getting the multi-defendant cases including those with anonymized defendants (xxxxx) from proceedings

def get_multi_defendant_cases(data: dict, crawled_path: Path, min_x_count: int = 3) -> dict:
    # Load crawled dockets
    mag_dockets = pd.read_csv(crawled_path / "dockets.csv", dtype=str)
    cdc_dockets = pd.read_csv(crawled_path / "criminal_dockets.csv", dtype=str)

    # Count defendants per case from crawled dockets
    cdc_counts = cdc_dockets[['case_id', 'defendant']].drop_duplicates().groupby('case_id').size()
    mag_counts = mag_dockets[['mag_id', 'defendant']].drop_duplicates().groupby('mag_id').size()

    # Check proceedings for anonymized defendants (xxxxx)
    anonymized_pattern = re.compile(r'x{' + str(min_x_count) + r',}', re.IGNORECASE)

    # Find cases with anonymized defendants in proceedings text
    anonymized_cdc = set()
    anonymized_mag = set()

    if 'cdc' in data:
        for _, row in data['cdc'].iterrows():
            if anonymized_pattern.search(str(row.get('text', ''))):
                anonymized_cdc.add(str(row.get('case_id', '')))

    if 'mag' in data:
        for _, row in data['mag'].iterrows():
            if anonymized_pattern.search(str(row.get('text', ''))):
                anonymized_mag.add(str(row.get('mag_num', '')))

    # Combine: multi-defendant from crawled + cases with anonymized defendants
    cdc_multi = set(cdc_counts[cdc_counts > 1].index.astype(str)) | anonymized_cdc
    mag_multi = set(mag_counts[mag_counts > 1].index.astype(str)) | anonymized_mag

    return {'cdc_multi': list(cdc_multi), 'mag_multi': list(mag_multi)}

## Remove multi-defendant cases from data

def remove_multi_defendant_cases(data: dict, crawled_path: Path = None) -> dict:
    if crawled_path is None:
        raise ValueError("crawled_path must be provided to load crawled dockets for multi-defendant case identification.")
    # Get multi-defendant case lists
    multi_cases = get_multi_defendant_cases(data, crawled_path)
    cdc_multi_set = set(multi_cases['cdc_multi'])
    mag_multi_set = set(multi_cases['mag_multi'])

    # Find corresponding cases to also remove
    # If mag case is multi-defendant, also remove corresponding cdc case
    # If cdc case is multi-defendant, also remove corresponding mag case

    if 'mag_dockets' in data and 'cdc_dockets' in data:
        # Get mag_num -> case_id mappings
        mag_to_cdc = {}
        cdc_to_mag = {}

        for _, row in data['mag_dockets'].iterrows():
            # Handle None/NaN values properly
            mag_num = row.get('mag_num')
            case_id = row.get('case_id')

            # Convert to string if not None/NaN, otherwise skip
            mag_num_str = str(mag_num) if pd.notna(mag_num) and mag_num is not None else 'None'
            case_id_str = str(case_id) if pd.notna(case_id) and case_id is not None else 'None'

            # Only add mapping if both values are valid
            if (mag_num_str and case_id_str and
                mag_num_str not in ['nan', 'None', 'NaT', ''] and
                case_id_str not in ['nan', 'None', 'NaT', '']):

                mag_to_cdc[mag_num_str] = case_id_str
                cdc_to_mag[case_id_str] = mag_num_str

        # Add corresponding cases to removal sets
        for mag_num in list(mag_multi_set):
            if mag_num in mag_to_cdc:
                cdc_multi_set.add(mag_to_cdc[mag_num])

        for case_id in list(cdc_multi_set):
            if case_id in cdc_to_mag:
                mag_multi_set.add(cdc_to_mag[case_id])

    # Filter data
    filtered_data = data.copy()

    # Filter proceedings
    if 'mag' in filtered_data:
        mask = ~filtered_data['mag']['mag_num'].astype(str).isin(mag_multi_set)
        filtered_data['mag'] = filtered_data['mag'][mask]

    if 'cdc' in filtered_data:
        mask = ~filtered_data['cdc']['case_id'].astype(str).isin(cdc_multi_set)
        filtered_data['cdc'] = filtered_data['cdc'][mask]

    # Filter related tables
    for table, col, multi_set in [
        ('mag_dockets', 'mag_num', mag_multi_set),
        ('cdc_dockets', 'case_id', cdc_multi_set),
        ('mag_charges', 'mag_num', mag_multi_set),
        ('cdc_charges', 'case_id', cdc_multi_set)
    ]:
        if table in filtered_data:
            mask = ~filtered_data[table][col].astype(str).isin(multi_set)
            filtered_data[table] = filtered_data[table][mask]

    # Remove combined_dockets for recreation
    if 'combined_dockets' in filtered_data:
        del filtered_data['combined_dockets']

    print(f"Removed {len(cdc_multi_set)} CDC and {len(mag_multi_set)} mag cases (including corresponding matches)")
    return filtered_data, (mag_multi_set, cdc_multi_set)

# SECTION Create combined dockets dataframe with mag and cdc information

def create_combined_dockets(data: dict) -> pd.DataFrame:

    # Get the magistrate sections from the proceedings data
    mag_sections = data['mag'][['mag_num', 'magistrate_section']].drop_duplicates()
    mag_sections = mag_sections.groupby('mag_num').first().reset_index()

    # Rename columns in mag_dockets to add mag_ prefix
    mag_dockets_renamed = data['mag_dockets'].rename(columns={
        'date_of_first_proceeding': 'mag_date_of_first_proceeding',
        'original_bond': 'mag_original_bond',
        'sum_of_bonds': 'mag_sum_of_bonds'
    })

    # Add mag_section by merging with mag_sections
    mag_dockets_renamed = mag_dockets_renamed.merge(
        mag_sections,
        on='mag_num',
        how='left'
    ).rename(columns={'magistrate_section': 'mag_section'})

    # Rename columns in cdc_dockets to add cdc_ prefix
    cdc_dockets_renamed = data['cdc_dockets'].rename(columns={
        'date_of_first_proceeding': 'cdc_date_of_first_proceeding',
        'original_bond': 'cdc_original_bond',
        'sum_of_bonds': 'cdc_sum_of_bonds',
        'section': 'cdc_section'
    })

    # matched magistrate + CDC rows
    dockets_cases = (
        data['mag_dockets']
        .loc[data['mag_dockets']['case_id'].notna(),
             ['mag_num', 'case_id', 'defendant_id']]
        .drop_duplicates()
    )

    # magistrate-only rows
    mag_only_cases = (
        data['mag_dockets']
        .loc[data['mag_dockets']['case_id'].isna(),
             ['mag_num', 'defendant_id']]
        .drop_duplicates()
    )
    mag_only_cases['case_id'] = np.nan
    mag_only_cases = mag_only_cases[['mag_num', 'case_id', 'defendant_id']]

    matched_case_ids = set(dockets_cases['case_id'].dropna().astype(str).unique())

    # CDC-only rows
    cdc_only_cases = (
        data['cdc_dockets']
        .loc[~data['cdc_dockets']['case_id'].astype(str).isin(matched_case_ids),
             ['case_id', 'defendant_id']]
        .drop_duplicates()
    )

    cdc_only_cases['mag_num'] = np.nan
    cdc_only_cases = cdc_only_cases[['mag_num', 'case_id', 'defendant_id']]

    # Combine all unique cases
    unique_cases = pd.concat([dockets_cases, mag_only_cases, cdc_only_cases], ignore_index=True).drop_duplicates()

    # Find (case_id, defendant_id) pairs that already have a MAG-CDC match (non-null mag_num).
    # These are the only ones where a CDC-only duplicate row should be dropped.
    matched_pairs = set(
        zip(
            dockets_cases['case_id'].astype(str),
            dockets_cases['defendant_id'].astype(str)
        )
    )

    # A row is a spurious CDC-only duplicate if:
    # 1. It has no mag_num (it came from cdc_only_cases), AND
    # 2. Its (case_id, defendant_id) pair already exists in a MAG-CDC row
    is_spurious_cdc_only = (
        unique_cases['mag_num'].isna() &
        unique_cases.apply(
            lambda r: (str(r['case_id']), str(r['defendant_id'])) in matched_pairs, axis=1
        )
    )

    unique_cases = unique_cases[~is_spurious_cdc_only].reset_index(drop=True)

    # Merge with magistrate data
    combined_df = unique_cases.merge(
        mag_dockets_renamed[['mag_num', 'defendant_id', 'mag_date_of_first_proceeding',
                            'mag_original_bond', 'mag_sum_of_bonds', 'mag_section']],
        on=['mag_num', 'defendant_id'],
        how='left'
    )

    # Merge with CDC data
    combined_df = combined_df.merge(
        cdc_dockets_renamed[['case_id', 'defendant_id', 'cdc_date_of_first_proceeding',
                            'cdc_original_bond', 'cdc_sum_of_bonds', 'cdc_section']],
        on=['case_id', 'defendant_id'],
        how='left'
    )

    # Create the key column
    combined_df['key'] = (
        combined_df['mag_num'].astype(str).fillna('') + ',' +
        combined_df['case_id'].astype(str).fillna('') + ',' +
        combined_df['defendant_id'].astype(str)
    )

    # Reorder columns as requested
    combined_df = combined_df[[
        'key', 'mag_num', 'case_id', 'defendant_id',
        'mag_date_of_first_proceeding', 'cdc_date_of_first_proceeding',
        'mag_original_bond', 'cdc_original_bond',
        'mag_sum_of_bonds', 'cdc_sum_of_bonds',
        'mag_section', 'cdc_section'
    ]]

    return combined_df

def filter_to_magistrate_started_cases(data: dict) -> dict:
    """
    Filter data to only include cases that started in magistrate court.
    Should be called after create_combined_dockets() has been run.
    
    Args:
        data: Dictionary containing court data with combined_dockets already created
        
    Returns:
        Filtered data dictionary containing only magistrate-started cases
    """
    if 'combined_dockets' not in data:
        raise ValueError("combined_dockets must exist in data. Run create_combined_dockets() first.")
    
    combined_df = data['combined_dockets']
    print(f"Starting with {len(combined_df)} total cases in combined_dockets")
    
    # Filter to magistrate-started cases only
    # Cases that start in magistrate are those where:
    # 1. mag_date_of_first_proceeding exists, AND
    # 2. Either no CDC date exists OR magistrate date is before/equal to CDC date
    mag_started_mask = (
        combined_df['mag_date_of_first_proceeding'].notna() &
        (
            combined_df['cdc_date_of_first_proceeding'].isna() |
            (combined_df['mag_date_of_first_proceeding'] <= combined_df['cdc_date_of_first_proceeding'])
        )
    )
    
    mag_started_cases = combined_df[mag_started_mask]
    print(f"Found {len(mag_started_cases)} magistrate-started cases")
    
    # Get the IDs we want to keep
    valid_mag_nums = set(mag_started_cases['mag_num'].dropna().astype(str))
    valid_case_ids = set(mag_started_cases['case_id'].dropna().astype(str))
    valid_defendant_ids = set(mag_started_cases['defendant_id'].dropna().astype(str))
    
    # Create filtered data copy
    filtered_data = data.copy()
    filtered_data['combined_dockets'] = mag_started_cases
    
    # Filter magistrate tables
    if 'mag' in filtered_data and not filtered_data['mag'].empty:
        original_count = len(filtered_data['mag'])
        filtered_data['mag'] = filtered_data['mag'][
            filtered_data['mag']['mag_num'].astype(str).isin(valid_mag_nums)
        ]
        print(f"  Filtered mag proceedings: {original_count} -> {len(filtered_data['mag'])}")
    
    if 'mag_dockets' in filtered_data and not filtered_data['mag_dockets'].empty:
        original_count = len(filtered_data['mag_dockets'])
        filtered_data['mag_dockets'] = filtered_data['mag_dockets'][
            filtered_data['mag_dockets']['mag_num'].astype(str).isin(valid_mag_nums)
        ]
        print(f"  Filtered mag_dockets: {original_count} -> {len(filtered_data['mag_dockets'])}")
    
    if 'mag_charges' in filtered_data and not filtered_data['mag_charges'].empty:
        original_count = len(filtered_data['mag_charges'])
        filtered_data['mag_charges'] = filtered_data['mag_charges'][
            filtered_data['mag_charges']['mag_num'].astype(str).isin(valid_mag_nums)
        ]
        print(f"  Filtered mag_charges: {original_count} -> {len(filtered_data['mag_charges'])}")
    
    # Filter CDC tables (only keep CDC cases that are linked to magistrate-started cases)
    if 'cdc' in filtered_data and not filtered_data['cdc'].empty:
        original_count = len(filtered_data['cdc'])
        filtered_data['cdc'] = filtered_data['cdc'][
            filtered_data['cdc']['case_id'].astype(str).isin(valid_case_ids)
        ]
        print(f"  Filtered cdc proceedings: {original_count} -> {len(filtered_data['cdc'])}")
    
    if 'cdc_dockets' in filtered_data and not filtered_data['cdc_dockets'].empty:
        original_count = len(filtered_data['cdc_dockets'])
        filtered_data['cdc_dockets'] = filtered_data['cdc_dockets'][
            filtered_data['cdc_dockets']['case_id'].astype(str).isin(valid_case_ids)
        ]
        print(f"  Filtered cdc_dockets: {original_count} -> {len(filtered_data['cdc_dockets'])}")
    
    if 'cdc_charges' in filtered_data and not filtered_data['cdc_charges'].empty:
        original_count = len(filtered_data['cdc_charges'])
        filtered_data['cdc_charges'] = filtered_data['cdc_charges'][
            filtered_data['cdc_charges']['case_id'].astype(str).isin(valid_case_ids)
        ]
        print(f"  Filtered cdc_charges: {original_count} -> {len(filtered_data['cdc_charges'])}")
    
    # Filter defendants table to only include defendants with magistrate-started cases
    if 'defendants' in filtered_data and not filtered_data['defendants'].empty:
        original_count = len(filtered_data['defendants'])
        filtered_data['defendants'] = filtered_data['defendants'][
            filtered_data['defendants']['defendant_id'].astype(str).isin(valid_defendant_ids)
        ]
        print(f"  Filtered defendants: {original_count} -> {len(filtered_data['defendants'])}")
    
    print(f"Filtering complete. Dataset now contains only magistrate-started cases.")
    return filtered_data

# SECTION Remove Extradition Cases
def remove_extradition_cases(data: dict) -> dict:
    
    non_louisiana_states = [state for state in us_states if state != r'LOUISIANA']
    
    print(f"Starting extradition case removal...")
    
    mag_nums_to_remove = set()
    case_ids_to_remove = set()
    
    if 'mag' in data and not data['mag'].empty:
        for idx, row in data['mag'].iterrows():
            text = str(row.get('text', '')).upper()
            mag_num = str(row.get('mag_num', ''))
            if 'EXTRAD' in text:
                mag_nums_to_remove.add(mag_num)
                continue
            for state_pattern in non_louisiana_states:
                if re.search(state_pattern, text, re.IGNORECASE):
                    mag_nums_to_remove.add(mag_num)
                    break
        print(f"  Found {len(mag_nums_to_remove)} magistrate cases with extradition indicators")
    
    if 'cdc' in data and not data['cdc'].empty:
        for idx, row in data['cdc'].iterrows():
            text = str(row.get('text', '')).upper()
            case_id = str(row.get('case_id', ''))
            if 'EXTRAD' in text:
                case_ids_to_remove.add(case_id)
                continue
            for state_pattern in non_louisiana_states:
                if re.search(state_pattern, text, re.IGNORECASE):
                    case_ids_to_remove.add(case_id)
                    break
        print(f"  Found {len(case_ids_to_remove)} CDC cases with extradition indicators")
    
    if 'mag_dockets' in data and not data['mag_dockets'].empty:
        mag_to_cdc = {}
        cdc_to_mag = {}
        for _, row in data['mag_dockets'].iterrows():
            mag_num = str(row.get('mag_num')) if pd.notna(row.get('mag_num')) else None
            case_id = str(row.get('case_id')) if pd.notna(row.get('case_id')) else None
            if mag_num and case_id and mag_num != 'nan' and case_id != 'nan':
                mag_to_cdc[mag_num] = case_id
                cdc_to_mag[case_id] = mag_num
        for mag_num in list(mag_nums_to_remove):
            if mag_num in mag_to_cdc:
                case_ids_to_remove.add(mag_to_cdc[mag_num])
        for case_id in list(case_ids_to_remove):
            if case_id in cdc_to_mag:
                mag_nums_to_remove.add(cdc_to_mag[case_id])
    
    filtered_data = data.copy()
    
    if 'mag' in filtered_data and not filtered_data['mag'].empty:
        original_count = len(filtered_data['mag'])
        filtered_data['mag'] = filtered_data['mag'][
            ~filtered_data['mag']['mag_num'].astype(str).isin(mag_nums_to_remove)
        ]
        print(f"  Filtered mag proceedings: {original_count} -> {len(filtered_data['mag'])}")
    
    if 'cdc' in filtered_data and not filtered_data['cdc'].empty:
        original_count = len(filtered_data['cdc'])
        filtered_data['cdc'] = filtered_data['cdc'][
            ~filtered_data['cdc']['case_id'].astype(str).isin(case_ids_to_remove)
        ]
        print(f"  Filtered cdc proceedings: {original_count} -> {len(filtered_data['cdc'])}")
    
    if 'mag_dockets' in filtered_data and not filtered_data['mag_dockets'].empty:
        original_count = len(filtered_data['mag_dockets'])
        filtered_data['mag_dockets'] = filtered_data['mag_dockets'][
            ~filtered_data['mag_dockets']['mag_num'].astype(str).isin(mag_nums_to_remove)
        ]
        print(f"  Filtered mag_dockets: {original_count} -> {len(filtered_data['mag_dockets'])}")
    
    if 'cdc_dockets' in filtered_data and not filtered_data['cdc_dockets'].empty:
        original_count = len(filtered_data['cdc_dockets'])
        filtered_data['cdc_dockets'] = filtered_data['cdc_dockets'][
            ~filtered_data['cdc_dockets']['case_id'].astype(str).isin(case_ids_to_remove)
        ]
        print(f"  Filtered cdc_dockets: {original_count} -> {len(filtered_data['cdc_dockets'])}")
    
    if 'mag_charges' in filtered_data and not filtered_data['mag_charges'].empty:
        original_count = len(filtered_data['mag_charges'])
        filtered_data['mag_charges'] = filtered_data['mag_charges'][
            ~filtered_data['mag_charges']['mag_num'].astype(str).isin(mag_nums_to_remove)
        ]
        print(f"  Filtered mag_charges: {original_count} -> {len(filtered_data['mag_charges'])}")
    
    if 'cdc_charges' in filtered_data and not filtered_data['cdc_charges'].empty:
        original_count = len(filtered_data['cdc_charges'])
        filtered_data['cdc_charges'] = filtered_data['cdc_charges'][
            ~filtered_data['cdc_charges']['case_id'].astype(str).isin(case_ids_to_remove)
        ]
        print(f"  Filtered cdc_charges: {original_count} -> {len(filtered_data['cdc_charges'])}")
    
    if 'defendants' in filtered_data and not filtered_data['defendants'].empty:
        valid_defendant_ids = set()
        if 'mag_dockets' in filtered_data:
            valid_defendant_ids.update(filtered_data['mag_dockets']['defendant_id'].dropna().astype(str))
        if 'cdc_dockets' in filtered_data:
            valid_defendant_ids.update(filtered_data['cdc_dockets']['defendant_id'].dropna().astype(str))
        original_count = len(filtered_data['defendants'])
        filtered_data['defendants'] = filtered_data['defendants'][
            filtered_data['defendants']['defendant_id'].astype(str).isin(valid_defendant_ids)
        ]
        print(f"  Filtered defendants: {original_count} -> {len(filtered_data['defendants'])}")
    
    # Filter combined_dockets in place instead of deleting it
    # This preserves the magistrate-started constraint applied earlier
    if 'combined_dockets' in filtered_data and not filtered_data['combined_dockets'].empty:
        original_count = len(filtered_data['combined_dockets'])
        filtered_data['combined_dockets'] = filtered_data['combined_dockets'][
            ~filtered_data['combined_dockets']['mag_num'].astype(str).isin(mag_nums_to_remove) &
            ~filtered_data['combined_dockets']['case_id'].astype(str).isin(case_ids_to_remove)
        ]
        print(f"  Filtered combined_dockets: {original_count} -> {len(filtered_data['combined_dockets'])}")
    
    print(f"Extradition case removal complete.")
    print(f"  Removed {len(mag_nums_to_remove)} magistrate cases")
    print(f"  Removed {len(case_ids_to_remove)} CDC cases")
    
    return filtered_data