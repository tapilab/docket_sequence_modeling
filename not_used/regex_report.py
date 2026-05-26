# Python File that has functions related to generating and caching the labels that were made from regular expression

import pickle
import re
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from datetime import datetime

from regex_labels.patterns_and_extractors import (
    extract_first_appearance_hearing, determine_bond_status, should_include_ror,
    extract_the_court_appointed_the_orleans_public_defender, extract_ankle_monitoring,
    extract_domestic_stay_away_order, determine_charge_status, extract_gwens_law_hearing,
    has_probable_cause, extract_defendant_in_custody_and_not_in_court,
    extract_the_state_was_not_present, extract_pre_trial_conference,
    determine_continuance, plea_status, guilty_or_no_guilty, sentence_jail_status
)
from config import EXCLUDED_LABELS
from sklearn.feature_extraction.text import CountVectorizer
import json
import pandas as pd

# Generate comprehensive regex report using cached labels and save to file

def generate_and_save_regex_report(data: dict, cached_labels, cache_dir, n_top_ngrams: int = 10, max_ngram_length: int = 10) -> None:
    # Combine all proceedings text
    all_proceedings = []
    # Get magistrate proceedings
    for _, row in data['mag'].iterrows():
        text = row.get('text', '')
        if text and isinstance(text, str):
            all_proceedings.append({
                'proceeding_id': row['proceeding_id'],
                'text': text,
                'type': 'magistrate'
            })
    # Get CDC proceedings
    for _, row in data['cdc'].iterrows():
        text = row.get('text', '')
        if text and isinstance(text, str):
            all_proceedings.append({
                'proceeding_id': row['proceeding_id'],
                'text': text,
                'type': 'cdc'
            })
    # Define label mapping to regex functions (same as before)
    label_functions = {
        "FIRST_APPEARANCE_HEARING_OCCURRED": extract_first_appearance_hearing,
        "BOND_SET": lambda text: 'BOND_SET' if determine_bond_status(text) == 'BOND_SET' else 'NONE',
        "BOND_NOT_SET": lambda text: 'BOND_NOT_SET' if determine_bond_status(text) == 'BOND_NOT_SET' else 'NONE',
        "DEFENDANT_RELEASED_ON_OWN_RECOGNIZANCE": lambda text: 'DEFENDANT_RELEASED_ON_OWN_RECOGNIZANCE' if should_include_ror(text) == 'DEFENDANT_RELEASED_ON_OWN_RECOGNIZANCE' else 'NONE',
        "THE_COURT_APPOINTED_THE_ORLEANS_PUBLIC_DEFENDER": extract_the_court_appointed_the_orleans_public_defender,
        "ANKLE_MONITORING_IMPOSED": extract_ankle_monitoring,
        "DOMESTIC_STAY_AWAY_ORDER_IMPOSED": extract_domestic_stay_away_order,
        "CHARGES_ACCEPTED": lambda text: 'CHARGES_ACCEPTED' if determine_charge_status(text) == 'CHARGES_ACCEPTED' else 'NONE',
        "CHARGES_REFUSED": lambda text: 'CHARGES_REFUSED' if determine_charge_status(text) == 'CHARGES_REFUSED' else 'NONE',
        #"GWENS_LAW_HEARING_OCCURRED": extract_gwens_law_hearing,
        "PROBABLE_CAUSE_FOUND": lambda text: 'PROBABLE_CAUSE_FOUND' if has_probable_cause(text) == 'PROBABLE_CAUSE_FOUND' else 'NONE',
        "PROBABLE_CAUSE_NOT_FOUND": lambda text: 'PROBABLE_CAUSE_NOT_FOUND' if has_probable_cause(text) == 'PROBABLE_CAUSE_NOT_FOUND' else 'NONE',
        "DEFENDANT_IN_CUSTODY_AND_NOT_IN_COURT": extract_defendant_in_custody_and_not_in_court,
        "THE_STATE_WAS_NOT_PRESENT": extract_the_state_was_not_present,
        "PRE_TRIAL_CONFERENCE_HELD": extract_pre_trial_conference,
        "CONTINUANCE_OCCURRED": determine_continuance,
        "PLED_GUILTY_TO_CHARGES": lambda text: 'PLED_GUILTY_TO_CHARGES' if plea_status(text) == 'PLED_GUILTY_TO_CHARGES' else 'NONE',
        "PLED_NOT_GUILTY_TO_CHARGES": lambda text: 'PLED_NOT_GUILTY_TO_CHARGES' if plea_status(text) == 'PLED_NOT_GUILTY_TO_CHARGES' else 'NONE',
        "FOUND_GUILTY_BY_JUDGE_OR_JURY": lambda text: 'FOUND_GUILTY_BY_JUDGE_OR_JURY' if guilty_or_no_guilty(text) == 'FOUND_GUILTY_BY_JUDGE_OR_JURY' else 'NONE',
        "FOUND_NOT_GUILTY_BY_JUDGE_OR_JURY": lambda text: 'FOUND_NOT_GUILTY_BY_JUDGE_OR_JURY' if guilty_or_no_guilty(text) == 'FOUND_NOT_GUILTY_BY_JUDGE_OR_JURY' else 'NONE',
        "SENTENCE_THAT_INCLUDES_JAIL_TIME_WAS_IMPOSED": lambda text: 'SENTENCE_THAT_INCLUDES_JAIL_TIME_WAS_IMPOSED' if sentence_jail_status(text) == 'SENTENCE_THAT_INCLUDES_JAIL_TIME_WAS_IMPOSED' else 'NONE',
        "SENTENCE_THAT_DOES_NOT_INCLUDE_JAIL_TIME_WAS_IMPOSED": lambda text: 'SENTENCE_THAT_DOES_NOT_INCLUDE_JAIL_TIME_WAS_IMPOSED' if sentence_jail_status(text) == 'SENTENCE_THAT_DOES_NOT_INCLUDE_JAIL_TIME_WAS_IMPOSED' else 'NONE'
    }
    # Create comprehensive report
    report_lines = []
    report_data = {}
    report_lines.append("COMPREHENSIVE REGEX ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total proceedings analyzed: {len(all_proceedings)}")
    report_lines.append("=" * 80)
    # Process each label
    for label_name, label_function in tqdm(label_functions.items(), desc="Generating regex report"):
        report_lines.append(f"\n{label_name}")
        report_lines.append("-" * len(label_name))
        # Find matching proceedings using cached labels for efficiency when possible
        matching_texts = []
        matching_proceeding_ids = []
        # Use cached labels to find matches more efficiently
        for proceeding in all_proceedings:
            proceeding_id = proceeding['proceeding_id']
            if proceeding_id in cached_labels:
              cached_proc_labels = cached_labels[proceeding_id]
              if label_name in cached_proc_labels:
                matching_texts.append(proceeding['text'])
                matching_proceeding_ids.append(proceeding_id)
        match_count = len(matching_texts)
        match_percentage = (match_count / len(all_proceedings) * 100) if len(all_proceedings) > 0 else 0
        report_lines.append(f"Total matches: {match_count} out of {len(all_proceedings)} proceedings ({match_percentage:.2f}%)")
        # Store data for JSON export
        label_data = {
            'match_count': match_count,
            'match_percentage': match_percentage,
            'matching_proceeding_ids': matching_proceeding_ids,
            'ngrams': {},
            'example_excerpts': []
        }
        if match_count == 0:
            report_lines.append("No matches found for this label.")
            report_data[label_name] = label_data
            continue

        # Extract n-grams from matching texts
        if matching_texts:
            report_lines.append(f"\nTop {n_top_ngrams} n-grams from matching proceedings:")

            for ngram_size in range(1, max_ngram_length + 1):
                try:
                    vectorizer = CountVectorizer(
                        ngram_range=(ngram_size, ngram_size),
                        token_pattern=r'(?u)\S+',
                        stop_words=None,
                        min_df=1,
                        max_df=1.0,
                        lowercase=True
                    )

                    X = vectorizer.fit_transform(matching_texts)
                    feature_names = vectorizer.get_feature_names_out()
                    counts = X.sum(axis=0).A1

                    ngram_freq = list(zip(feature_names, counts))
                    ngram_freq.sort(key=lambda x: x[1], reverse=True)

                    ngram_type = {1: "Unigrams", 2: "Bigrams", 3: "Trigrams", 4: "4-grams", 5: "5-grams"}.get(ngram_size, f"{ngram_size}-grams")
                    report_lines.append(f"\n  {ngram_type}:")

                    ngram_list = []
                    for i, (ngram, freq) in enumerate(ngram_freq[:n_top_ngrams]):
                        percentage = (freq / match_count) * 100
                        report_lines.append(f"    {i+1:2d}. '{ngram}' (appears {freq} times, {percentage:.1f}% of matches)")
                        ngram_list.append({
                            'ngram': ngram,
                            'frequency': int(freq),
                            'percentage': round(percentage, 1)
                        })

                    label_data['ngrams'][ngram_type.lower()] = ngram_list

                except Exception as e:
                    report_lines.append(f"    Error extracting {ngram_size}-grams: {str(e)}")

        # Show example excerpts
        report_lines.append(f"\nExample excerpts (showing first 3 matches):")
        for i, text in enumerate(matching_texts[:3]):
            excerpt = ' '.join(text.split())
            if len(excerpt) > 200:
                excerpt = excerpt[:200] + "..."
            report_lines.append(f"  {i+1}. {excerpt}")
            label_data['example_excerpts'].append(excerpt)

        report_data[label_name] = label_data
        report_lines.append("\n" + "="*80)

    # Save text report
    report_text_file = cache_dir / "regex_analysis_report.txt"
    with open(report_text_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    # Save JSON data
    report_json_file = cache_dir / "regex_analysis_data.json"
    with open(report_json_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # Save summary CSV
    summary_data = []
    for label_name, data in report_data.items():
        summary_data.append({
            'label': label_name,
            'match_count': data['match_count'],
            'match_percentage': data['match_percentage']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_csv_file = cache_dir / "regex_analysis_summary.csv"
    summary_df.to_csv(summary_csv_file, index=False)

    print(f"\nRegex analysis report saved to:")
    print(f"  Text report: {report_text_file}")
    print(f"  JSON data: {report_json_file}")
    print(f"  Summary CSV: {summary_csv_file}")

    # Print summary to console
    print(f"\nREGEX ANALYSIS SUMMARY")
    print("-" * 50)
    print(f"{'Label':<50} {'Matches':<8} {'%':<6}")
    print("-" * 64)
    for _, row in summary_df.sort_values('match_count', ascending=False).iterrows():
        print(f"{row['label']:<50} {row['match_count']:<8} {row['match_percentage']:<6.1f}")