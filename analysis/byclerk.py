import os
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import pandas as pd


def load_and_merge_proceedings_with_clerks(court_data: Dict):
    """
    Load v1 clerk data and merge with existing court_data following your pattern
    """
    # Load v1 data with clerk info
    mag_proceedings_v1_path = r"/data/krosen5/gdrive-all/tulane-courts-shared/crawled/proceedings.csv"
    cdc_proceedings_v1_path = r"/data/krosen5/gdrive-all/tulane-courts-shared/crawled/criminal_proceedings.csv"
    
    mag_proceedings_v1 = pd.read_csv(mag_proceedings_v1_path, dtype={'mag_id':str,'date': str, 'clerk':str, 'text': str})
    cdc_proceedings_v1 = pd.read_csv(cdc_proceedings_v1_path, dtype={'case_id':str,'date': str, 'clerk':str, 'text': str})
    
    def date_fixer(df):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    
    mag_proceedings_v1 = date_fixer(mag_proceedings_v1)
    cdc_proceedings_v1 = date_fixer(cdc_proceedings_v1)
    
    # Get existing v2 data
    mag_proceedings_v2 = court_data['mag'].copy()
    cdc_proceedings_v2 = court_data['cdc'].copy()
    
    mag_proceedings_v2 = date_fixer(mag_proceedings_v2)
    cdc_proceedings_v2 = date_fixer(cdc_proceedings_v2)
    
    # Merge on mag_num, date, and text
    mag_merged = mag_proceedings_v2.merge(
        mag_proceedings_v1, 
        left_on=['mag_num', 'date', 'text'], 
        right_on=['mag_id', 'date', 'text'], 
        how='inner', 
        suffixes=('_v2', '_v1')
    )
    
    # Merge on case_id, date, and text  
    cdc_merged = cdc_proceedings_v2.merge(
        cdc_proceedings_v1,
        left_on=['case_id', 'date', 'text'],
        right_on=['case_id', 'date', 'text'],
        how='inner',
        suffixes=('_v2', '_v1')
    )
    
    return mag_merged, cdc_merged

class ClerkLabelAnalyzer:
    """
    Simple analyzer for creating clerk by label count code tables
    """
    
    def __init__(self, court_data: Dict, cached_outcome_labels: Dict, 
                 valid_completed_defendant_cases: List[Tuple] = None):
        """
        Initialize the analyzer
        
        Args:
            court_data: Dictionary containing court dataframes
            cached_outcome_labels: Dictionary mapping proceeding_id to labels
            valid_completed_defendant_cases: List of (mag_num, case_id, defendant_id) tuples to filter to
        """
        self.cached_outcome_labels = cached_outcome_labels
        self.valid_completed_defendant_cases = valid_completed_defendant_cases
        self.clerk_table = None
        
        # Load and merge with clerk data
        self.mag_merged, self.cdc_merged = load_and_merge_proceedings_with_clerks(court_data)
        
        # Filter to valid completed cases if provided
        if valid_completed_defendant_cases is not None:
            self._filter_to_valid_cases()
        
    
    def _filter_to_valid_cases(self):
        valid_mag_nums = {case[0] for case in self.valid_completed_defendant_cases if pd.notna(case[0])}
        valid_case_ids = {case[1] for case in self.valid_completed_defendant_cases if pd.notna(case[1])}
        valid_defendant_ids = {case[2] for case in self.valid_completed_defendant_cases if pd.notna(case[2])}
        
        # Filter magistrate data
        if not self.mag_merged.empty:
            original_mag_count = len(self.mag_merged)
            
            # Filter by mag_num and defendant_id if available
            mag_filter = self.mag_merged['mag_num'].isin(valid_mag_nums)
            if 'defendant_id' in self.mag_merged.columns:
                mag_filter = mag_filter & self.mag_merged['defendant_id'].isin(valid_defendant_ids)
            
            self.mag_merged = self.mag_merged[mag_filter]
            print(f"Filtered magistrate: {original_mag_count} -> {len(self.mag_merged)} records")
        
        # Filter CDC data
        if not self.cdc_merged.empty:
            original_cdc_count = len(self.cdc_merged)
            
            # Filter by case_id and defendant_id if available
            cdc_filter = self.cdc_merged['case_id'].isin(valid_case_ids)
            if 'defendant_id' in self.cdc_merged.columns:
                cdc_filter = cdc_filter & self.cdc_merged['defendant_id'].isin(valid_defendant_ids)
            
            self.cdc_merged = self.cdc_merged[cdc_filter]
            print(f"Filtered CDC: {original_cdc_count} -> {len(self.cdc_merged)} records")
        
    def create_clerk_label_table(self) -> pd.DataFrame:
        """
        Create clerk by label count table
        
        Returns:
            DataFrame with clerk, label, count information
        """
        results = []
        
        # Process magistrate data
        results.extend(self._process_court_data(self.mag_merged, 'MAGISTRATE'))
        
        # Process CDC data
        results.extend(self._process_court_data(self.cdc_merged, 'CDC'))
        
        # Create DataFrame
        columns = ['clerk_name', 'court_type', 'label', 'count', 'percentage', 'total_proceedings']
        self.clerk_table = pd.DataFrame(results, columns=columns)
        
        return self.clerk_table
    
    def _process_court_data(self, merged_df: pd.DataFrame, court_type: str) -> List[List]:
        """
        Process merged dataframe to extract clerk label data
        """
        results = []
        
        if merged_df.empty:
            return results
        
        # Group by clerk and analyze labels
        for clerk_name, group in merged_df.groupby('clerk'):
            if pd.isna(clerk_name):
                continue
                
            # Get all proceeding IDs for this clerk
            proceeding_ids = group['proceeding_id'].dropna().tolist()
            
            # Count labels for these proceedings
            label_counts = self._count_labels_for_proceedings(proceeding_ids)
            total_proceedings = len(proceeding_ids)
            
            if total_proceedings == 0 or not label_counts:
                continue
            
            # Create rows for each label
            for label, count in label_counts.items():
                percentage = (count / total_proceedings) * 100
                
                row = [
                    str(clerk_name),
                    court_type,
                    label,
                    count,
                    round(percentage, 2),
                    total_proceedings
                ]
                
                results.append(row)
        
        return results
    
    def _count_labels_for_proceedings(self, proceeding_ids: List) -> Counter:
        """
        Count labels for a list of proceeding IDs
        """
        label_counts = Counter()
        
        for pid in proceeding_ids:
            # Try different formats for proceeding ID lookup
            labels = []
            if pid in self.cached_outcome_labels:
                labels = self.cached_outcome_labels[pid]
            elif str(pid) in self.cached_outcome_labels:
                labels = self.cached_outcome_labels[str(pid)]
            
            for label in labels:
                if label:
                    label_counts[label] += 1
        
        return label_counts
    
    def create_clerk_label_matrix(self) -> pd.DataFrame:
        """
        Create a matrix with clerks as rows and labels as columns
        
        Returns:
            DataFrame matrix with counts
        """
        if self.clerk_table is None or self.clerk_table.empty:
            print("No clerk table available. Run create_clerk_label_table() first.")
            return pd.DataFrame()
        
        # Clerks as rows, Labels as columns
        matrix = self.clerk_table.pivot_table(
            values='count',
            index='clerk_name',
            columns='label',
            aggfunc='sum',
            fill_value=0
        )
        
        return matrix
    
    def create_clerk_label_matrix_by_court(self, court_type: str = 'all') -> Dict[str, pd.DataFrame]:
        """
        Create matrices separated by court type (clerks as rows, labels as columns)
        
        Args:
            court_type: 'magistrate', 'cdc', or 'all'
            
        Returns:
            Dictionary of matrices by court type
        """
        if self.clerk_table is None or self.clerk_table.empty:
            print("No clerk table available. Run create_clerk_label_table() first.")
            return {}
        
        matrices = {}
        
        if court_type == 'all':
            court_types = self.clerk_table['court_type'].unique()
        else:
            court_types = [court_type.upper()]
        
        for ct in court_types:
            court_data = self.clerk_table[self.clerk_table['court_type'] == ct]
            if not court_data.empty:
                # Clerks as rows, labels as columns
                matrix = court_data.pivot_table(
                    values='count',
                    index='clerk_name',
                    columns='label',
                    aggfunc='sum',
                    fill_value=0
                )
                matrices[ct] = matrix
        
        return matrices

def analyze_clerk_data(court_data: Dict, cached_outcome_labels: Dict, 
                      valid_completed_defendant_cases: List[Tuple] = None,
                      output_dir: str = "./output") -> ClerkLabelAnalyzer:
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    analyzer = ClerkLabelAnalyzer(court_data, cached_outcome_labels, valid_completed_defendant_cases)
    clerk_table = analyzer.create_clerk_label_table()
    clerk_table.to_csv(os.path.join(output_dir, "clerk_label_table.csv"), index=False)

    matrices = analyzer.create_clerk_label_matrix()
    matrices.to_csv(os.path.join(output_dir, "clerk_label_matrix.csv"))

    matrices_by_court = analyzer.create_clerk_label_matrix_by_court('all')
    for court_type, matrix in matrices_by_court.items():
        matrix.to_csv(os.path.join(output_dir, f"clerk_label_matrix_{court_type.lower()}.csv"))
        