import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import os
import gc
import warnings
warnings.filterwarnings('ignore')

from config import get_model_config

# Ngram on Complete Cases Proceedings With No Matches
def get_unmatched_proceedings_from_completed_cases(completed_cases, cached_labels, data, case_sequences_dict):
    unmatched_proceedings = []
    print("Finding unmatched proceedings from completed cases...")
    
    for mag_num, case_id, def_id in tqdm(completed_cases, desc="Processing completed cases"):
        case_key = f"{mag_num}_{case_id}_{def_id}"
        # Get sequences from the precomputed dictionary
        case_sequences = case_sequences_dict.get(case_key, [])
        
        for case in case_sequences:
            proceedings = case.get('proceedings', [])
            for proc in proceedings:
                proceeding_id = proc.get('proceeding_id')
                if proceeding_id and proceeding_id not in cached_labels:
                    text = proc.get('text', '')
                    # Check if this proceeding has no match in cached labels
                    if text and isinstance(text, str):                   
                        unmatched_proceedings.append({
                            'proceeding_id': proceeding_id,
                            'case_key': case_key,
                            'text': text,
                            'type': proc.get('type', 'unknown'),
                            'date': proc.get('date'),
                            'mag_num': mag_num,
                            'case_id': case_id,
                            'defendant_id': def_id,
                            'magistrate_section': proc.get('magistrate_section', 'unknown'),
                            'text_length': len(text)
                        })
    
    print(f"Found {len(unmatched_proceedings)} unmatched proceedings from {len(completed_cases)} completed cases")
    return unmatched_proceedings
def process_batch_with_memory_management(texts, text_to_case_batch, n, existing_results=None):
    """Process a batch of texts with memory management"""
    if existing_results is None:
        existing_results = defaultdict(lambda: defaultdict(int))
    
    print(f"    Processing batch of {len(texts)} texts for {n}-grams...")
    
    # Use CountVectorizer on this batch
    vectorizer = CountVectorizer(
        ngram_range=(n, n),
        stop_words=None,
        token_pattern=r'(?u)\S+',
        lowercase=True
    )
    
    try:
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get all n-grams (no filtering for general analysis)
        X_coo = X.tocoo()
        
        # Build results from sparse matrix
        for row, col, count in zip(X_coo.row, X_coo.col, X_coo.data):
            case_key = text_to_case_batch[row]
            ngram = feature_names[col]
            existing_results[ngram][case_key] += count
        
        # Clean up
        del X, vectorizer
        gc.collect()
        
    except Exception as e:
        print(f"    Error processing batch: {e}")
    
    return existing_results
def batch_extract_ngrams_memory_optimized(unmatched_proceedings, n):
    """
    MEMORY-OPTIMIZED: Process texts in batches to avoid memory issues
    """
    print(f"Processing {n}-grams with memory optimization...")
    
    model_configuration = get_model_config()
    # Prepare data for batch processing
    all_texts = []
    text_to_case = []
    
    for proc in unmatched_proceedings:
        text = proc['text']
        case_key = proc['case_key']
        
        if text and isinstance(text, str):
            all_texts.append(text)
            text_to_case.append(case_key)
    
    if not all_texts:
        return pd.DataFrame()
    
    print(f"  Processing {len(all_texts)} texts across {len(set(text_to_case))} cases")
    
    # Process in batches to manage memory
    all_results = defaultdict(lambda: defaultdict(int))
    
    for i in range(0, len(all_texts), model_configuration.NGRAM_BATCH_SIZE):
        batch_end = min(i + model_configuration.NGRAM_BATCH_SIZE, len(all_texts))
        texts_batch = all_texts[i:batch_end]
        text_to_case_batch = text_to_case[i:batch_end]
        
        print(f"  Processing batch {i//model_configuration.NGRAM_BATCH_SIZE + 1}/{(len(all_texts) + model_configuration.NGRAM_BATCH_SIZE - 1)//model_configuration.NGRAM_BATCH_SIZE}")
        
        all_results = process_batch_with_memory_management(
            texts_batch, text_to_case_batch, n, all_results
        )
        
        # Force garbage collection between batches
        gc.collect()
    
    # Convert results to DataFrame
    if all_results:
        results = []
        for ngram, case_counts in all_results.items():
            for case_key, count in case_counts.items():
                results.append({
                    'MatchingNgram': ngram,
                    'CaseKey': case_key,
                    'Count': count
                })
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(['MatchingNgram', 'Count'], ascending=[True, False])
        
        print(f"  Found {len(df_results)} unique n-gram/case combinations")
        return df_results
    
    return pd.DataFrame()
def get_file_path_with_number(base_path, file_number):
    """Generate file path with number suffix"""
    if file_number == 1:
        return base_path
    
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    return parent / f"{stem}_{file_number}{suffix}"

def split_dataframe_for_sheets(df, max_rows=get_model_config().SAFE_ROWS_PER_SHEET):
    """Split a large dataframe into chunks that fit in Excel sheets"""
    chunks = []
    for i in range(0, len(df), max_rows):
        chunk = df.iloc[i:i + max_rows].copy()
        chunks.append(chunk)
    return chunks

def process_unmatched_ngrams_memory_optimized(unmatched_proceedings, base_output_path, dataset_name):
    """Memory-optimized processing with batch processing"""
    model_configuration = get_model_config()
    
    print(f"Processing n-grams for {len(unmatched_proceedings)} unmatched proceedings in {dataset_name}")
    
    if not unmatched_proceedings:
        print("No unmatched proceedings to process")
        return False
    
    try:
        all_results = {}
        
        # Process each n-gram size with memory-optimized batch processing
        for n in range(1, 11):
            print(f"\n--- Processing {n}-grams ---")
            df_counts = batch_extract_ngrams_memory_optimized(unmatched_proceedings, n)
            
            if not df_counts.empty:
                all_results[f'{n}_grams'] = df_counts
                print(f"  ✓ {n}-grams: {len(df_counts)} results")
            else:
                print(f"  ✗ No matching {n}-grams found")
            
            # Force garbage collection after each n-gram size
            gc.collect()
        
        # Write to Excel
        if not all_results:
            with pd.ExcelWriter(base_output_path, engine='xlsxwriter') as writer:
                pd.DataFrame({'Note': ['No matching n-grams found']}).to_excel(writer, sheet_name='Results', index=False)
            print(f"No results found. Empty file created: {base_output_path}")
            return False
        
        # Excel writing logic
        file_number = 1
        current_file_path = get_file_path_with_number(base_output_path, file_number)
        sheets_in_current_file = 0
        total_sheets_created = 0
        current_writer = None
        
        sorted_ngram_types = sorted(all_results.keys(), key=lambda x: int(x.split('_')[0]))
        
        print(f"\nWriting results to Excel...")
        
        for ngram_type in sorted_ngram_types:
            df_counts = all_results[ngram_type]
            data_chunks = split_dataframe_for_sheets(df_counts)
            
            for chunk_idx, chunk in enumerate(data_chunks):
                if sheets_in_current_file >= model_configuration.SAFE_SHEETS_PER_FILE:
                    if current_writer is not None:
                        current_writer.close()
                    file_number += 1
                    current_file_path = get_file_path_with_number(base_output_path, file_number)
                    sheets_in_current_file = 0
                    current_writer = None
                    print(f"  Starting new file: {os.path.basename(current_file_path)}")
                
                sheet_name = ngram_type if len(data_chunks) == 1 else f"{ngram_type}_part{chunk_idx + 1}"
                
                if current_writer is None:
                    current_writer = pd.ExcelWriter(current_file_path, engine='xlsxwriter')
                    workbook = current_writer.book
                    header_format = workbook.add_format({
                        'bold': True,
                        'text_wrap': True,
                        'valign': 'top',
                        'fg_color': '#D7E4BC',
                        'border': 1
                    })
                
                chunk.to_excel(current_writer, sheet_name=sheet_name, index=False)
                
                worksheet = current_writer.sheets[sheet_name]
                for col_num, value in enumerate(chunk.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                for i, col in enumerate(chunk.columns):
                    max_length = max(chunk[col].astype(str).str.len().max(), len(col))
                    worksheet.set_column(i, i, min(max_length + 2, 50))
                
                sheets_in_current_file += 1
                total_sheets_created += 1
                print(f"  {sheet_name}: {len(chunk):,} rows")
        
        if current_writer is not None:
            current_writer.close()
            print(f"  Saved file: {os.path.basename(current_file_path)}")
        
        print(f"\nExcel processing complete for {dataset_name}")
        print(f"Created {file_number} file(s) with {total_sheets_created} total sheets")
        return True
        
    except Exception as e:
        print(f"Error saving Excel file for {dataset_name}: {e}")
        if 'current_writer' in locals() and current_writer is not None:
            try:
                current_writer.close()
            except:
                pass
        return False

def consolidate_ngram_results(input_path, output_path):
    """Consolidate n-gram results by case key"""
    print(f"Consolidating n-gram results from: {input_path}")
    
    try:
        # Load all n-gram data
        xlsx_file = pd.ExcelFile(input_path)
        sheet_names = xlsx_file.sheet_names
        print(f"Found {len(sheet_names)} sheets: {sheet_names}")
        
        all_data = []
        
        for sheet_name in sheet_names:
            print(f"  Loading sheet: {sheet_name}")
            try:
                df_sheet = pd.read_excel(input_path, sheet_name=sheet_name)
                
                # Add n-gram type information
                if '_grams' in sheet_name:
                    ngram_size = sheet_name.split('_')[0]
                    df_sheet['NgramSize'] = int(ngram_size)
                else:
                    # Handle part sheets
                    parts = sheet_name.split('_')
                    if len(parts) >= 2 and parts[1] == 'grams':
                        df_sheet['NgramSize'] = int(parts[0])
                    else:
                        print(f"    Warning: Could not determine n-gram size for sheet {sheet_name}")
                        continue
                
                all_data.append(df_sheet)
                print(f"    Loaded {len(df_sheet)} rows")
                
            except Exception as e:
                print(f"    Error loading sheet {sheet_name}: {e}")
                continue
        
        if not all_data:
            print("No data could be loaded from the Excel file")
            return False
        
        # Combine all sheets
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total combined data: {len(combined_df)} rows")
        
        # Sort and add rank column
        combined_df = combined_df.sort_values(['CaseKey', 'Count', 'MatchingNgram'], ascending=[True, False, True]).reset_index(drop=True)
        combined_df['RankInCase'] = combined_df.groupby('CaseKey').cumcount() + 1
        
        # Save consolidated results
        return save_consolidated_results(combined_df, output_path)
        
    except Exception as e:
        print(f"Error consolidating results: {e}")
        return False
def save_consolidated_results(df_all_ngrams, output_path):
    """Save consolidated n-grams by case to Excel with case integrity preserved"""
    print(f"Saving consolidated results to: {output_path}")
    
    try:
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            number_format = workbook.add_format({'num_format': '#,##0'})
            
            # Split data preserving case integrity
            print("  Creating sheets with all n-grams by case...")
            all_ngrams_chunks = split_data_by_case_preserving_integrity(df_all_ngrams)
            
            for i, chunk in enumerate(all_ngrams_chunks, 1):
                sheet_name = f'All_Ngrams_Part{i}' if len(all_ngrams_chunks) > 1 else 'All_Ngrams'
                chunk.to_excel(writer, sheet_name=sheet_name, index=False)
                
                worksheet = writer.sheets[sheet_name]
                for col_num, value in enumerate(chunk.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Auto-adjust column widths
                for j, col in enumerate(chunk.columns):
                    max_length = max(
                        chunk[col].astype(str).str.len().max() if len(chunk) > 0 else 0,
                        len(col)
                    )
                    worksheet.set_column(j, j, min(max_length + 2, 50))
                
                # Apply number formatting
                if 'Count' in chunk.columns:
                    count_col_idx = chunk.columns.get_loc('Count')
                    worksheet.set_column(count_col_idx, count_col_idx, None, number_format)
                if 'RankInCase' in chunk.columns:
                    rank_col_idx = chunk.columns.get_loc('RankInCase')
                    worksheet.set_column(rank_col_idx, rank_col_idx, None, number_format)
                
                cases_in_chunk = chunk['CaseKey'].unique()
                print(f"  ✓ {sheet_name}: {len(chunk)} rows, {len(cases_in_chunk)} cases")
        
        print(f"Successfully saved consolidated results to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error saving consolidated results: {e}")
        return False
def split_data_by_case_preserving_integrity(df, max_rows_per_sheet=get_model_config().SAFE_ROWS_PER_SHEET):
    """Split dataframe into sheets ensuring complete cases stay together"""
    if 'CaseKey' not in df.columns:
        # If no CaseKey, just split normally
        chunks = []
        for i in range(0, len(df), max_rows_per_sheet):
            chunk = df.iloc[i:i + max_rows_per_sheet].copy()
            chunks.append(chunk)
        return chunks
    
    chunks = []
    current_chunk = []
    current_rows = 0
    
    # Sort by CaseKey to group cases together
    df_sorted = df.sort_values(['CaseKey', 'Count', 'MatchingNgram'], ascending=[True, False, True]).reset_index(drop=True)
    
    # Group by case and process each case as a unit
    for case_key, case_group in df_sorted.groupby('CaseKey'):
        case_size = len(case_group)
        
        # If adding this case would exceed the limit, start a new chunk
        if current_rows > 0 and (current_rows + case_size) > max_rows_per_sheet:
            # Save current chunk and start new one
            if current_chunk:
                chunks.append(pd.concat(current_chunk, ignore_index=True))
            current_chunk = [case_group]
            current_rows = case_size
        else:
            # Add case to current chunk
            current_chunk.append(case_group)
            current_rows += case_size
        
        # If a single case is larger than max_rows_per_sheet, it gets its own sheet
        if case_size > max_rows_per_sheet:
            print(f"  Warning: Case {case_key} has {case_size} rows, exceeds sheet limit")
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(pd.concat(current_chunk, ignore_index=True))
    
    return chunks
def analyze_unmatched_proceedings_summary(unmatched_proceedings):
    """Analyze summary statistics of unmatched proceedings"""
    print("\n" + "=" * 80)
    print("UNMATCHED PROCEEDINGS ANALYSIS SUMMARY")
    print("=" * 80)
    
    if not unmatched_proceedings:
        print("No unmatched proceedings to analyze")
        return
    
    # Basic statistics
    total_proceedings = len(unmatched_proceedings)
    unique_cases = len(set(proc['case_key'] for proc in unmatched_proceedings))
    
    print(f"Total unmatched proceedings: {total_proceedings:,}")
    print(f"Unique cases: {unique_cases:,}")
    print(f"Average proceedings per case: {total_proceedings/unique_cases:.1f}")
    
    # Type analysis
    type_counts = Counter(proc['type'] for proc in unmatched_proceedings)
    print(f"\nBy court type:")
    for court_type, count in type_counts.items():
        percentage = (count / total_proceedings) * 100
        print(f"  {court_type}: {count:,} ({percentage:.1f}%)")
    
    # Text length analysis
    text_lengths = [proc['text_length'] for proc in unmatched_proceedings]
    print(f"\nText length statistics:")
    print(f"  Average: {np.mean(text_lengths):.1f} characters")
    print(f"  Median: {np.median(text_lengths):.1f} characters")
    print(f"  Range: {min(text_lengths):,} - {max(text_lengths):,} characters")
    
    # Magistrate section analysis
    section_counts = Counter(proc['magistrate_section'] for proc in unmatched_proceedings)
    print(f"\nTop 10 magistrate sections:")
    for section, count in section_counts.most_common(10):
        percentage = (count / total_proceedings) * 100
        print(f"  {section}: {count:,} ({percentage:.1f}%)")

def run_complete_unmatched_ngram_analysis(completed_cases, cached_labels, data, case_sequences_dict, cache_manager = None, cache_dir: Path = None):
    """
    Main function to run complete n-gram analysis on unmatched proceedings.
    This function will be called from your main method.
    """
    print("=" * 80)
    print("STARTING COMPLETE N-GRAM ANALYSIS FOR UNMATCHED PROCEEDINGS")
    print("=" * 80)
    
    try:
        # Step 1: Get unmatched proceedings
        unmatched_proceedings = get_unmatched_proceedings_from_completed_cases(
            completed_cases, cached_labels, data, case_sequences_dict
        )
        
        if not unmatched_proceedings:
            print("No unmatched proceedings found in completed cases")
            return False
        
        # Step 2: Analyze summary statistics
        analyze_unmatched_proceedings_summary(unmatched_proceedings)
        
        # Step 3: Set up output paths
        if cache_dir is None:
            cache_dir = cache_manager.get_cache_dir("labels")
        
        output_folder = cache_dir / "unmatched_ngrams"
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Step 4: Process n-grams
        raw_output_path = output_folder / "unmatched_ngrams_raw_1to10.xlsx"
        consolidated_output_path = output_folder / "unmatched_ngrams_consolidated_by_case.xlsx"
        
        print(f"\n" + "=" * 80)
        print("PROCESSING N-GRAMS FOR UNMATCHED PROCEEDINGS")
        print("=" * 80)
        
        success = process_unmatched_ngrams_memory_optimized(
            unmatched_proceedings, 
            raw_output_path, 
            "Unmatched proceedings"
        )
        
        if not success:
            print("N-gram processing failed")
            return False
        
        # Step 5: Consolidate results
        print(f"\n" + "=" * 80)
        print("CONSOLIDATING N-GRAM RESULTS")
        print("=" * 80)
        
        consolidation_success = consolidate_ngram_results(
            raw_output_path,
            consolidated_output_path
        )
        
        # Step 6: Final summary
        print(f"\n" + "=" * 80)
        print("UNMATCHED PROCEEDINGS N-GRAM ANALYSIS COMPLETE")
        print("=" * 80)
        
        if consolidation_success:
            print(f"✓ Analysis completed successfully")
            print(f"  Proceedings analyzed: {len(unmatched_proceedings):,}")
            print(f"  Cases analyzed: {len(set(proc['case_key'] for proc in unmatched_proceedings)):,}")
            print(f"  Raw output: {raw_output_path}")
            print(f"  Consolidated output: {consolidated_output_path}")
            print(f"  Output directory: {output_folder}")
            
            # Save summary report
            summary_file = output_folder / "analysis_summary.txt"
            with open(summary_file, 'w') as f:
                f.write("UNMATCHED PROCEEDINGS N-GRAM ANALYSIS SUMMARY\n")
                f.write("=" * 50 + "\n")
                f.write(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total unmatched proceedings: {len(unmatched_proceedings):,}\n")
                f.write(f"Unique cases: {len(set(proc['case_key'] for proc in unmatched_proceedings)):,}\n")
                f.write(f"Raw output file: {raw_output_path.name}\n")
                f.write(f"Consolidated output file: {consolidated_output_path.name}\n")
            
            print(f"  Summary report: {summary_file}")
            
        else:
            print("✗ Consolidation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error in complete n-gram analysis: {e}")
        import traceback
        traceback.print_exc()
        return False