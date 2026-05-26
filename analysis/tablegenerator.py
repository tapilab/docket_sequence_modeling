import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from notebook_helpers import evaluate_case_level
from models.transformer_creation import create_model_from_data
import torch

class TableGenerator:    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'main_model': None,
            'baseline_1': None,
            'baseline_2': None,
            'transformer_only_completion': None,
            'baseline_1_only_completion': None,
            'baseline_1_folds': [],
            'baseline_2_folds': [],
            'transformer_only_completion_folds': [],
            'baseline_1_only_completion_folds': [],
            'k_fold_folds': [],
            'ablation_studies': {}
        }
        
    def add_main_model_results(self, 
                               step_metrics: Dict,
                               case_results: List[Dict],
                               model_name: str = "Main Transformer"):
        self.results['main_model'] = {
            'name': model_name,
            'type': 'Primary',
            'step_metrics': step_metrics,
            'case_results': case_results
        }
    
        
    def add_baseline_results(self,
                            step_metrics: Dict,
                            case_results: List[Dict],
                            model_name: str = "Baseline Model", 
                            type_of_model: str = 'Baseline Model',
                            results_key: str = 'baseline_1'):
        self.results[results_key] = {
            'name': model_name,
            'type': type_of_model,
            'step_metrics': step_metrics,
            'case_results': case_results
        }
    
    def add_baseline_fold_results(self,
                                  fold_idx: int,
                                  step_metrics: Dict,
                                  case_results: List[Dict] = None,
                                  model_name: str = "Baseline Model", 
                                  type_of_model: str = 'Baseline Model',
                                  results_key: str = 'baseline_1_folds'):
        self.results[results_key].append({
            'name': f"{model_name} (Fold {fold_idx + 1})",
            'type': type_of_model,
            'fold_idx': fold_idx,
            'step_metrics': step_metrics,
            'case_results': case_results
        })
    
    def add_k_fold_results(self,
                          fold_idx: int,
                          step_metrics: Dict,
                          case_results: List[Dict] = None,
                          train_size: int = None,
                          val_size: int = None):
        self.results['k_fold_folds'].append({
            'name': f"K-Fold Fold {fold_idx + 1}",
            'type': 'K-Fold',
            'fold_idx': fold_idx,
            'step_metrics': step_metrics,
            'case_results': case_results,
            'train_size': train_size,
            'val_size': val_size
        })
    
    def add_k_fold_results_from_cv_runner(self, cv_runner, label_processor, device):

        print("Adding K-Fold results from cv_runner...")
        
        for fold_idx, fold_result in enumerate(cv_runner.fold_results):
            print(f"  Processing Fold {fold_idx + 1}/{len(cv_runner.fold_results)}...")
            
            # Get step-level metrics (already computed)
            step_metrics = fold_result['metrics']
            
            # Compute case-level metrics for this fold
            fold_model = cv_runner.model_creation_func()
            fold_model.load_state_dict(fold_result['model_state'])
            fold_model.to(device)
            fold_model.eval()
            
            # Evaluate case-level
            case_results = evaluate_case_level(
                fold_model,
                fold_result['val_data'],
                label_processor
            )
            
            # Add to results
            self.add_k_fold_results(
                fold_idx=fold_idx,
                step_metrics=step_metrics,
                case_results=case_results,
                train_size=fold_result.get('train_size'),
                val_size=fold_result.get('val_size')
            )
            
            # Clean up
            del fold_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"✅ Added {len(cv_runner.fold_results)} K-Fold results")
        
    def add_ablation_results(self,
                            config_name: str,
                            ablated_features: List[str],
                            step_metrics: Dict,
                            case_results: List[Dict] = None):  
        self.results['ablation_studies'][config_name] = {
            'name': f"Ablation: {config_name}",
            'type': 'Ablation',
            'ablated_features': ablated_features,
            'step_metrics': step_metrics,
            'case_results': case_results
        }
    
        
    def generate_table_1_generalized(self, baseline_1_type_of_model='Baseline Model', baseline_2_type_of_model='Random Baseline') -> pd.DataFrame:
        rows = []
        
        # Add main model
        if self.results['main_model']:
            rows.append(self._extract_generalized_row(self.results['main_model']))
        
        # Add baseline
        if self.results['baseline_1']:
            rows.append(self._extract_generalized_row(self.results['baseline_1']))
        
        if self.results['baseline_2']:
            rows.append(self._extract_generalized_row(self.results['baseline_2']))
        
        if self.results['baseline_1_folds']:
            for baseline_fold in self.results['baseline_1_folds']:
                rows.append(self._extract_generalized_row(baseline_fold))
        
        if self.results['baseline_2_folds']:
            for baseline_fold in self.results['baseline_2_folds']:
                rows.append(self._extract_generalized_row(baseline_fold))

        if self.results['baseline_1_folds']:
            # Add Baseline Fold Average
            rows.append(self._compute_baseline_fold_average(
                baseline_1_type_of_model,
                'baseline_1_folds'
            ))
            
            # Add Baseline Best Fold
            rows.append(self._compute_baseline_fold_best(
                baseline_1_type_of_model,
                'baseline_1_folds'
            ))
        
        if self.results['baseline_2_folds']:
            # Add Baseline Fold Average
            rows.append(self._compute_baseline_fold_average(
                baseline_2_type_of_model,
                'baseline_2_folds'
            ))
            #Add baseline best fold
            rows.append(self._compute_baseline_fold_best(
                baseline_2_type_of_model,
                'baseline_2_folds'))

        if self.results['transformer_only_completion']:
            rows.append(self._extract_generalized_row(self.results['transformer_only_completion']))
        if self.results['baseline_1_only_completion']:
            rows.append(self._extract_generalized_row(self.results['baseline_1_only_completion']))
        
        if self.results['transformer_only_completion_folds']:
            for transformer_fold in self.results['transformer_only_completion_folds']:
                rows.append(self._extract_generalized_row(transformer_fold))
        
        if self.results['transformer_only_completion_folds']:
            # Add Transformer Only Completion Fold Average
            rows.append(self._compute_baseline_fold_average(
                'Transformer Model Only Completion',
                'transformer_only_completion_folds'
            ))
            # Add Transformer Only Completion Best Fold
            rows.append(self._compute_baseline_fold_best(
                'Transformer Model Only Completion',
                'transformer_only_completion_folds'
            ))
        if self.results['baseline_1_only_completion_folds']:
            for baseline1_completion_fold in self.results['baseline_1_only_completion_folds']:
                rows.append(self._extract_generalized_row(baseline1_completion_fold))
        
        if self.results['baseline_1_only_completion_folds']:
            # Add Baseline 1 Only Completion Fold Average
            rows.append(self._compute_baseline_fold_average(
                'Baseline Model 1 Only Completion',
                'baseline_1_only_completion_folds'
            ))
            # Add Baseline 1 Only Completion Best Fold
            rows.append(self._compute_baseline_fold_best(
                'Baseline Model 1 Only Completion',
                'baseline_1_only_completion_folds'
            ))
        
        # Compute the difference between the only completion models averages
        if self.results['transformer_only_completion'] and self.results['baseline_1_only_completion']:
            transformer_avg = self._extract_generalized_row(self.results['transformer_only_completion'])
            baseline1_avg = self._extract_generalized_row(self.results['baseline_1_only_completion'])
            diff_row = {
                'Model': 'Δ (Transformer - Baseline 1)',
                'Type': 'Only Completion Comparison'
            }
            for metric in ['Micro_F1', 'Micro_Precision', 'Micro_Recall',
                           'Macro_F1', 'Weighted_F1',
                           'Case_Avg_F1', 'Case_Avg_Precision', 'Case_Avg_Recall',
                           'Exact_Match_Pct', 'Avg_True_Labels_Per_Case', 'Avg_Pred_Labels_Per_Case']:
                if metric in transformer_avg and metric in baseline1_avg:
                    try:
                        transformer_value = float(transformer_avg[metric])
                        baseline1_value = float(baseline1_avg[metric])
                        diff_value = transformer_value - baseline1_value
                        diff_row[metric] = f"{diff_value:.4f}"
                    except:
                        diff_row[metric] = "N/A"
            rows.append(diff_row)
        
        elif self.results['transformer_only_completion_folds'] and self.results['baseline_1_only_completion_folds']:
            transformer_avg = self._compute_baseline_fold_average(
                'Transformer Model Only Completion',
                'transformer_only_completion_folds'
            )
            baseline1_avg = self._compute_baseline_fold_average(
                'Baseline Model 1 Only Completion',
                'baseline_1_only_completion_folds'
            )
            diff_row = {
                'Model': 'Δ (Transformer Avg - Baseline 1 Avg)',
                'Type': 'Only Completion Comparison'
            }
            for metric in ['Micro_F1', 'Micro_Precision', 'Micro_Recall',
                           'Macro_F1', 'Weighted_F1',
                           'Case_Avg_F1', 'Case_Avg_Precision', 'Case_Avg_Recall',
                           'Exact_Match_Pct', 'Avg_True_Labels_Per_Case', 'Avg_Pred_Labels_Per_Case']:
                if metric in transformer_avg and metric in baseline1_avg:
                    try:
                        transformer_value = float(transformer_avg[metric].split(' ± ')[0])
                        baseline1_value = float(baseline1_avg[metric].split(' ± ')[0])
                        diff_value = transformer_value - baseline1_value
                        diff_row[metric] = f"{diff_value:.4f}"
                    except:
                        diff_row[metric] = "N/A"
            rows.append(diff_row)

        # Add individual K-Fold folds
        for fold_result in self.results['k_fold_folds']:
            rows.append(self._extract_generalized_row(fold_result))
        
        # Add K-Fold average
        if self.results['k_fold_folds']:
            rows.append(self._compute_k_fold_average())
        
        # Add K-Fold best
        if self.results['k_fold_folds']:
            rows.append(self._compute_k_fold_best())
        
        # Add ablation studies
        for ablation_result in self.results['ablation_studies'].values():
            rows.append(self._extract_generalized_row(ablation_result))
        
        if self.results['ablation_studies']:
            # Group ablation studies by base config name
            grouped_ablation = self._group_ablation_studies()
            
            for base_name, studies in grouped_ablation.items():
                if len(studies) > 1:
                    # Add Ablation Average
                    rows.append(self._compute_ablation_average(base_name, studies))
                    # Add Ablation Best
                    rows.append(self._compute_ablation_best(base_name, studies))
                
        df = pd.DataFrame(rows)
        
        # Reorder columns
        column_order = [
            'Model', 'Type',
            'Micro_F1', 'Micro_Precision', 'Micro_Recall',
            'Macro_F1', 'Weighted_F1',
            'Case_Avg_F1', 'Case_Avg_Precision', 'Case_Avg_Recall',
            'Exact_Match_Pct', 'Total_Cases',
            'Avg_True_Labels_Per_Case', 'Avg_Pred_Labels_Per_Case'
        ]
        
        # Only include columns that exist
        column_order = [col for col in column_order if col in df.columns]
        df = df[column_order]
        
        return df
    
    def _extract_generalized_row(self, result: Dict) -> Dict:
        row = {
            'Model': result['name'],
            'Type': result['type']
        }
        
        # Extract step-level metrics
        if result['step_metrics'] and 'SUMMARY' in result['step_metrics']:
            summary = result['step_metrics']['SUMMARY']
            row['Micro_F1'] = summary['micro_avg']['f1']
            row['Micro_Precision'] = summary['micro_avg']['precision']
            row['Micro_Recall'] = summary['micro_avg']['recall']
            row['Macro_F1'] = summary['macro_avg']['f1']
            row['Weighted_F1'] = summary['weighted_avg']['f1']
            row['Total_Cases'] = summary.get('total_samples', 0)
        
        # Extract case-level metrics
        if result.get('case_results'):
            case_results = result['case_results']
            row['Case_Avg_F1'] = np.mean([c['f1'] for c in case_results])
            row['Case_Avg_Precision'] = np.mean([c['precision'] for c in case_results])
            row['Case_Avg_Recall'] = np.mean([c['recall'] for c in case_results])
            row['Exact_Match_Pct'] = np.mean([c['exact_match'] for c in case_results]) * 100
            row['Avg_True_Labels_Per_Case'] = np.mean([c['num_true'] for c in case_results])
            row['Avg_Pred_Labels_Per_Case'] = np.mean([c['num_pred'] for c in case_results])
        
        return row
    
    def _compute_k_fold_average(self) -> Dict:
        folds = self.results['k_fold_folds']
        
        row = {
            'Model': 'K-Fold Average',
            'Type': 'K-Fold'
        }
        
        # Collect metrics from all folds
        metrics = {
            'Micro_F1': [],
            'Micro_Precision': [],
            'Micro_Recall': [],
            'Macro_F1': [],
            'Weighted_F1': [],
            'Case_Avg_F1': [],
            'Case_Avg_Precision': [],
            'Case_Avg_Recall': [],
            'Exact_Match_Pct': [],
            'Avg_True_Labels_Per_Case': [],
            'Avg_Pred_Labels_Per_Case': []
        }
        
        for fold in folds:
            if fold['step_metrics'] and 'SUMMARY' in fold['step_metrics']:
                summary = fold['step_metrics']['SUMMARY']
                metrics['Micro_F1'].append(summary['micro_avg']['f1'])
                metrics['Micro_Precision'].append(summary['micro_avg']['precision'])
                metrics['Micro_Recall'].append(summary['micro_avg']['recall'])
                metrics['Macro_F1'].append(summary['macro_avg']['f1'])
                metrics['Weighted_F1'].append(summary['weighted_avg']['f1'])
            
            if fold.get('case_results'):
                case_results = fold['case_results']
                metrics['Case_Avg_F1'].append(np.mean([c['f1'] for c in case_results]))
                metrics['Case_Avg_Precision'].append(np.mean([c['precision'] for c in case_results]))
                metrics['Case_Avg_Recall'].append(np.mean([c['recall'] for c in case_results]))
                metrics['Exact_Match_Pct'].append(np.mean([c['exact_match'] for c in case_results]) * 100)
                metrics['Avg_True_Labels_Per_Case'].append(np.mean([c['num_true'] for c in case_results]))
                metrics['Avg_Pred_Labels_Per_Case'].append(np.mean([c['num_pred'] for c in case_results]))
        
        # Compute mean ± std for each metric
        for metric_name, values in metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                row[metric_name] = f"{mean_val:.4f} ± {std_val:.4f}"
        
        return row
    
    def _compute_k_fold_best(self) -> Dict:
        folds = self.results['k_fold_folds']
        
        best_fold = max(folds, 
                       key=lambda f: f['step_metrics']['SUMMARY']['micro_avg']['f1'] 
                       if f['step_metrics'] and 'SUMMARY' in f['step_metrics'] else 0)
        
        row = self._extract_generalized_row(best_fold)
        row['Model'] = 'K-Fold Best Fold'
        row['Type'] = 'K-Fold'
        
        return row
        

    def _compute_baseline_fold_average(self, type_of_model, results_key) -> Dict:
        folds = self.results[results_key]
        
        row = {
            'Model': f'{type_of_model} Fold Average',
            'Type': type_of_model
        }
        
        # Collect metrics from all baseline folds
        metrics = {
            'Micro_F1': [],
            'Micro_Precision': [],
            'Micro_Recall': [],
            'Macro_F1': [],
            'Weighted_F1': [],
            'Case_Avg_F1': [],
            'Case_Avg_Precision': [],
            'Case_Avg_Recall': [],
            'Exact_Match_Pct': [],
            'Avg_True_Labels_Per_Case': [],
            'Avg_Pred_Labels_Per_Case': []
        }
        
        for fold in folds:
            if fold['step_metrics'] and 'SUMMARY' in fold['step_metrics']:
                summary = fold['step_metrics']['SUMMARY']
                metrics['Micro_F1'].append(summary['micro_avg']['f1'])
                metrics['Micro_Precision'].append(summary['micro_avg']['precision'])
                metrics['Micro_Recall'].append(summary['micro_avg']['recall'])
                metrics['Macro_F1'].append(summary['macro_avg']['f1'])
                metrics['Weighted_F1'].append(summary['weighted_avg']['f1'])
            
            if fold.get('case_results'):
                case_results = fold['case_results']
                metrics['Case_Avg_F1'].append(np.mean([c['f1'] for c in case_results]))
                metrics['Case_Avg_Precision'].append(np.mean([c['precision'] for c in case_results]))
                metrics['Case_Avg_Recall'].append(np.mean([c['recall'] for c in case_results]))
                metrics['Exact_Match_Pct'].append(np.mean([c['exact_match'] for c in case_results]) * 100)
                metrics['Avg_True_Labels_Per_Case'].append(np.mean([c['num_true'] for c in case_results]))
                metrics['Avg_Pred_Labels_Per_Case'].append(np.mean([c['num_pred'] for c in case_results]))
        
        # Compute mean ± std for each metric
        for metric_name, values in metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                row[metric_name] = f"{mean_val:.4f} ± {std_val:.4f}"
        
        return row
    
    def _compute_baseline_fold_best(self, type_of_model, results_key) -> Dict:
        """Find the best performing baseline fold (for K-Fold)."""
        folds = self.results[results_key]
        
        best_fold = max(folds, 
                       key=lambda f: f['step_metrics']['SUMMARY']['micro_avg']['f1'] 
                       if f['step_metrics'] and 'SUMMARY' in f['step_metrics'] else 0)
        
        row = self._extract_generalized_row(best_fold)
        row['Model'] = f'{type_of_model} Best Fold'
        row['Type'] = type_of_model
        
        return row    
    
    def _group_ablation_studies(self) -> Dict[str, List[Dict]]:
        groups = {}
        
        for config_name, ablation_result in self.results['ablation_studies'].items():
            # Extract base config name by removing " (Fold X)" suffix
            base_name = config_name
            if ' (Fold ' in config_name:
                base_name = config_name.split(' (Fold ')[0]
            
            if base_name not in groups:
                groups[base_name] = []
            
            groups[base_name].append(ablation_result)
        
        return groups
    
    def _compute_ablation_average(self, base_name: str, ablation_studies: List[Dict]) -> Dict:
        row = {
            'Model': f'Ablation: {base_name} Average',
            'Type': 'Ablation'
        }
        
        # Collect metrics from all ablation folds
        metrics = {
            'Micro_F1': [],
            'Micro_Precision': [],
            'Micro_Recall': [],
            'Macro_F1': [],
            'Weighted_F1': [],
            'Case_Avg_F1': [],
            'Case_Avg_Precision': [],
            'Case_Avg_Recall': [],
            'Exact_Match_Pct': [],
            'Avg_True_Labels_Per_Case': [],
            'Avg_Pred_Labels_Per_Case': []
        }
        
        for study in ablation_studies:
            if study['step_metrics'] and 'SUMMARY' in study['step_metrics']:
                summary = study['step_metrics']['SUMMARY']
                metrics['Micro_F1'].append(summary['micro_avg']['f1'])
                metrics['Micro_Precision'].append(summary['micro_avg']['precision'])
                metrics['Micro_Recall'].append(summary['micro_avg']['recall'])
                metrics['Macro_F1'].append(summary['macro_avg']['f1'])
                metrics['Weighted_F1'].append(summary['weighted_avg']['f1'])
            
            if study.get('case_results'):
                case_results = study['case_results']
                metrics['Case_Avg_F1'].append(np.mean([c['f1'] for c in case_results]))
                metrics['Case_Avg_Precision'].append(np.mean([c['precision'] for c in case_results]))
                metrics['Case_Avg_Recall'].append(np.mean([c['recall'] for c in case_results]))
                metrics['Exact_Match_Pct'].append(np.mean([c['exact_match'] for c in case_results]) * 100)
                metrics['Avg_True_Labels_Per_Case'].append(np.mean([c['num_true'] for c in case_results]))
                metrics['Avg_Pred_Labels_Per_Case'].append(np.mean([c['num_pred'] for c in case_results]))
        
        # Compute mean ± std for each metric
        for metric_name, values in metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                row[metric_name] = f"{mean_val:.4f} ± {std_val:.4f}"
        
        return row

    def _compute_ablation_best(self, base_name: str, ablation_studies: List[Dict]) -> Dict:
        best_study = max(ablation_studies, 
                        key=lambda s: s['step_metrics']['SUMMARY']['micro_avg']['f1'] 
                        if s['step_metrics'] and 'SUMMARY' in s['step_metrics'] else 0)
        
        row = self._extract_generalized_row(best_study)
        row['Model'] = f'Ablation: {base_name} Best'
        row['Type'] = 'Ablation'
        
        return row    
    def generate_table_2_per_label(self, label_names: List[str], baseline_1_type_of_model='Baseline Model', baseline_2_type_of_model='Random Baseline') -> pd.DataFrame:
        rows = []
        
        # Add main model
        if self.results['main_model']:
            rows.extend(self._extract_per_label_rows(self.results['main_model'], label_names))
        
        # Add baseline
        if self.results['baseline_1']:
            rows.extend(self._extract_per_label_rows(self.results['baseline_1'], label_names))

        if self.results['baseline_2']:
            rows.extend(self._extract_per_label_rows(self.results['baseline_2'], label_names))

        if self.results['baseline_1_folds']:
            for baseline_fold in self.results['baseline_1_folds']:
                rows.extend(self._extract_per_label_rows(baseline_fold, label_names))
        
        if self.results['baseline_2_folds']:
            for baseline_fold in self.results['baseline_2_folds']:
                rows.extend(self._extract_per_label_rows(baseline_fold, label_names))
        
        if self.results['baseline_1_folds']:
            rows.extend(self._compute_baseline_fold_per_label_average(label_names, baseline_1_type_of_model, 'baseline_1_folds'))
        
        if self.results['baseline_2_folds']:
            rows.extend(self._compute_baseline_fold_per_label_average(label_names, baseline_2_type_of_model, 'baseline_2_folds'))
        
        if self.results['transformer_only_completion']:
            rows.extend(self._extract_per_label_rows(self.results['transformer_only_completion'], label_names))
        if self.results['baseline_1_only_completion']:
            rows.extend(self._extract_per_label_rows(self.results['baseline_1_only_completion'], label_names))
        
        if self.results['transformer_only_completion_folds']:
            for transformer_fold in self.results['transformer_only_completion_folds']:
                rows.extend(self._extract_per_label_rows(transformer_fold, label_names))

        if self.results['transformer_only_completion_folds']:
            rows.extend(self._compute_baseline_fold_per_label_average(label_names, 'Transformer Model Only Completion', 'transformer_only_completion_folds'))
        
        if self.results['baseline_1_only_completion_folds']:
            for baseline1_completion_fold in self.results['baseline_1_only_completion_folds']:
                rows.extend(self._extract_per_label_rows(baseline1_completion_fold, label_names))
        
        if self.results['baseline_1_only_completion_folds']:
            rows.extend(self._compute_baseline_fold_per_label_average(label_names, 'Baseline Model 1 Only Completion', 'baseline_1_only_completion_folds'))
        
        if self.results['transformer_only_completion'] and self.results['baseline_1_only_completion']:
            transformer_rows = self._extract_per_label_rows(self.results['transformer_only_completion'], label_names)
            baseline1_rows = self._extract_per_label_rows(self.results['baseline_1_only_completion'], label_names)
            baseline_by_label_index = {row['Label']: row for row in baseline1_rows}
            for trans_row in transformer_rows:
                label = trans_row['Label']
                baseline_row = baseline_by_label_index.get(label)
                if baseline_row is None:
                    continue
                diff_row = {
                    'Model': f'Δ (Transformer - Baseline 1)',
                    'Type': 'Only Completion Comparison',
                    'Label': label
                }

                for metric in ['Precision', 'Recall', 'F1_Score', 'Support',
                            'TP', 'FP', 'FN',
                            'Total_Pred', 'Total_True']:
                    try:
                        transformer_value = float(trans_row.get(metric))
                        baseline_value = float(baseline_row.get(metric))
                        diff_value = transformer_value - baseline_value

                        if metric in ['Support', 'TP', 'FP', 'FN', 'Total_Pred', 'Total_True']:
                            diff_row[metric] = str(int(round(diff_value)))
                        else:
                            diff_row[metric] = f"{diff_value:.4f}"

                    except Exception:
                        diff_row[metric] = "N/A"

                rows.append(diff_row)
        elif self.results['transformer_only_completion_folds'] and self.results['baseline_1_only_completion_folds']:
            transformer_rows = self._compute_baseline_fold_per_label_average(label_names, 'Transformer Model Only Completion', 'transformer_only_completion_folds')
            baseline1_rows = self._compute_baseline_fold_per_label_average(label_names, 'Baseline Model 1 Only Completion', 'baseline_1_only_completion_folds')
            baseline_by_label_index = {row['Label']: row for row in baseline1_rows}
            for trans_row in transformer_rows:
                label = trans_row['Label']
                baseline_row = baseline_by_label_index.get(label)
                if baseline_row is None:
                    continue
                diff_row = {
                    'Model': f'Δ (Transformer Avg - Baseline 1 Avg)',
                    'Type': 'Only Completion Comparison',
                    'Label': label
                }
                for metric in ['Precision', 'Recall', 'F1_Score', 'Support',
                            'TP', 'FP', 'FN',
                            'Total_Pred', 'Total_True']:
                    try:
                        transformer_value = float(trans_row.get(metric).split(' ± ')[0])
                        baseline_value = float(baseline_row.get(metric).split(' ± ')[0])
                        diff_value = transformer_value - baseline_value
                        if metric in ['Support', 'TP', 'FP', 'FN', 'Total_Pred', 'Total_True']:
                            diff_row[metric] = str(int(round(diff_value)))
                        else:
                            diff_row[metric] = f"{diff_value:.4f}"
                    except Exception:
                        diff_row[metric] = "N/A"
                rows.append(diff_row)
        # Add individual K-Fold folds
        for fold_result in self.results['k_fold_folds']:
            rows.extend(self._extract_per_label_rows(fold_result, label_names))
        
        # Add K-Fold average
        if self.results['k_fold_folds']:
            rows.extend(self._compute_k_fold_per_label_average(label_names))
        
        # Add ablation studies
        for ablation_result in self.results['ablation_studies'].values():
            rows.extend(self._extract_per_label_rows(ablation_result, label_names))
        
        if self.results['ablation_studies']:
            # Group ablation studies by base config name
            grouped_ablation = self._group_ablation_studies()
            for base_name, studies in grouped_ablation.items():
                if len(studies) > 1:
                    rows.extend(self._compute_ablation_per_label_average(base_name, studies, label_names))
        
        df = pd.DataFrame(rows)
        
        # Reorder columns
        column_order = [
            'Model', 'Type', 'Label',
            'Precision', 'Recall', 'F1_Score', 'Support',
            'TP', 'FP', 'FN',
            'Total_Pred', 'Total_True'
        ]
        
        column_order = [col for col in column_order if col in df.columns]
        df = df[column_order]
        
        return df
    
    def _extract_per_label_rows(self, result: Dict, label_names: List[str]) -> List[Dict]:
        rows = []
        
        if not result['step_metrics'] or 'INDIVIDUAL_LABELS' not in result['step_metrics']:
            return rows
        
        individual_labels = result['step_metrics']['INDIVIDUAL_LABELS']
        
        for label in label_names:
            if label not in individual_labels:
                continue
            
            label_metrics = individual_labels[label]
            
            row = {
                'Model': result['name'],
                'Type': result['type'],
                'Label': label,
                'Precision': label_metrics['precision'],
                'Recall': label_metrics['recall'],
                'F1_Score': label_metrics['f1'],
                'Support': label_metrics['support'],
                'TP': label_metrics.get('tp', 0),
                'FP': label_metrics.get('fp', 0),
                'FN': label_metrics.get('fn', 0),
            }
            
            # Calculate totals
            row['Total_Pred'] = row['TP'] + row['FP']
            row['Total_True'] = row['Support']
            
            rows.append(row)
        
        return rows
    
    def _compute_k_fold_per_label_average(self, label_names: List[str]) -> List[Dict]:
        folds = self.results['k_fold_folds']
        rows = []
        
        for label in label_names:
            label_metrics = {
                'Precision': [],
                'Recall': [],
                'F1_Score': [],
                'Support': [],
                'TP': [],
                'FP': [],
                'FN': []
            }
            
            for fold in folds:
                if (fold['step_metrics'] and 
                    'INDIVIDUAL_LABELS' in fold['step_metrics'] and
                    label in fold['step_metrics']['INDIVIDUAL_LABELS']):
                    
                    lm = fold['step_metrics']['INDIVIDUAL_LABELS'][label]
                    label_metrics['Precision'].append(lm['precision'])
                    label_metrics['Recall'].append(lm['recall'])
                    label_metrics['F1_Score'].append(lm['f1'])
                    label_metrics['Support'].append(lm['support'])
                    label_metrics['TP'].append(lm.get('tp', 0))
                    label_metrics['FP'].append(lm.get('fp', 0))
                    label_metrics['FN'].append(lm.get('fn', 0))
            
            if not label_metrics['F1_Score']:
                continue
            
            row = {
                'Model': 'K-Fold Average',
                'Type': 'K-Fold',
                'Label': label
            }
            
            for metric_name, values in label_metrics.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    row[metric_name] = f"{mean_val:.4f} ± {std_val:.2f}" if metric_name in ['Precision', 'Recall', 'F1_Score'] else f"{mean_val:.1f} ± {std_val:.1f}"
            
            # Calculate totals
            if label_metrics['TP'] and label_metrics['FP']:
                total_pred_mean = np.mean([tp + fp for tp, fp in zip(label_metrics['TP'], label_metrics['FP'])])
                total_pred_std = np.std([tp + fp for tp, fp in zip(label_metrics['TP'], label_metrics['FP'])])
                row['Total_Pred'] = f"{total_pred_mean:.1f} ± {total_pred_std:.1f}"
            
            if label_metrics['Support']:
                row['Total_True'] = f"{np.mean(label_metrics['Support']):.1f} ± {np.std(label_metrics['Support']):.1f}"
            
            rows.append(row)
        
        return rows
    
    def _compute_baseline_fold_per_label_average(self, label_names: List[str], type_of_model, results_key) -> List[Dict]:
        folds = self.results[results_key]
        rows = []
        
        for label in label_names:
            label_metrics = {
                'Precision': [],
                'Recall': [],
                'F1_Score': [],
                'Support': [],
                'TP': [],
                'FP': [],
                'FN': []
            }
            
            for fold in folds:
                if (fold['step_metrics'] and 
                    'INDIVIDUAL_LABELS' in fold['step_metrics'] and
                    label in fold['step_metrics']['INDIVIDUAL_LABELS']):
                    
                    lm = fold['step_metrics']['INDIVIDUAL_LABELS'][label]
                    label_metrics['Precision'].append(lm['precision'])
                    label_metrics['Recall'].append(lm['recall'])
                    label_metrics['F1_Score'].append(lm['f1'])
                    label_metrics['Support'].append(lm['support'])
                    label_metrics['TP'].append(lm.get('tp', 0))
                    label_metrics['FP'].append(lm.get('fp', 0))
                    label_metrics['FN'].append(lm.get('fn', 0))
            
            if not label_metrics['F1_Score']:
                continue
            
            row = {
                'Model': f'{type_of_model} Fold Average',
                'Type': type_of_model,
                'Label': label
            }
            
            for metric_name, values in label_metrics.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    row[metric_name] = f"{mean_val:.4f} ± {std_val:.2f}" if metric_name in ['Precision', 'Recall', 'F1_Score'] else f"{mean_val:.1f} ± {std_val:.1f}"
            
            # Calculate totals
            if label_metrics['TP'] and label_metrics['FP']:
                total_pred_mean = np.mean([tp + fp for tp, fp in zip(label_metrics['TP'], label_metrics['FP'])])
                total_pred_std = np.std([tp + fp for tp, fp in zip(label_metrics['TP'], label_metrics['FP'])])
                row['Total_Pred'] = f"{total_pred_mean:.1f} ± {total_pred_std:.1f}"
            
            if label_metrics['Support']:
                row['Total_True'] = f"{np.mean(label_metrics['Support']):.1f} ± {np.std(label_metrics['Support']):.1f}"
            
            rows.append(row)
        
        return rows
        
    def _compute_ablation_per_label_average(self, base_name: str, ablation_studies: List[Dict], label_names: List[str]) -> List[Dict]:
        rows = []
        
        for label in label_names:
            label_metrics = {
                'Precision': [],
                'Recall': [],
                'F1_Score': [],
                'Support': [],
                'TP': [],
                'FP': [],
                'FN': []
            }
            
            for study in ablation_studies:
                if (study['step_metrics'] and 'INDIVIDUAL_LABELS' in study['step_metrics'] and label in study['step_metrics']['INDIVIDUAL_LABELS']):
                                        
                    lm = study['step_metrics']['INDIVIDUAL_LABELS'][label]
                    label_metrics['Precision'].append(lm['precision'])
                    label_metrics['Recall'].append(lm['recall'])
                    label_metrics['F1_Score'].append(lm['f1'])
                    label_metrics['Support'].append(lm['support'])
                    label_metrics['TP'].append(lm.get('tp', 0))
                    label_metrics['FP'].append(lm.get('fp', 0))
                    label_metrics['FN'].append(lm.get('fn', 0))
            
            if not label_metrics['F1_Score']:
                continue
            
            row = {
                'Model': f'Ablation: {base_name} Average',
                'Type': 'Ablation',
                'Label': label
            }
            
            for metric_name, values in label_metrics.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    row[metric_name] = f"{mean_val:.4f} ± {std_val:.2f}" if metric_name in ['Precision', 'Recall', 'F1_Score'] else f"{mean_val:.1f} ± {std_val:.1f}"
            
            # Calculate totals
            if label_metrics['TP'] and label_metrics['FP']:
                total_pred_mean = np.mean([tp + fp for tp, fp in zip(label_metrics['TP'], label_metrics['FP'])])
                total_pred_std = np.std([tp + fp for tp, fp in zip(label_metrics['TP'], label_metrics['FP'])])
                row['Total_Pred'] = f"{total_pred_mean:.1f} ± {total_pred_std:.1f}"
            
            if label_metrics['Support']:
                row['Total_True'] = f"{np.mean(label_metrics['Support']):.1f} ± {np.std(label_metrics['Support']):.1f}"
            
            rows.append(row)
        
        return rows
            
    
    
    def save_to_excel(self, 
                      table1_df: pd.DataFrame, 
                      table2_df: pd.DataFrame,
                      filename: str = "comprehensive_model_comparison.xlsx"):
        output_path = self.output_dir / filename
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Table 1: Generalized summary
            table1_df.to_excel(writer, sheet_name='Summary_All_Models', index=False)
            
            # Table 2: Per-label (long format)
            table2_df.to_excel(writer, sheet_name='PerLabel_Long_Format', index=False)
            
            # Create pivoted views for each label
            if not table2_df.empty:
                unique_labels = table2_df['Label'].unique()
                
                # Limit to first 20 labels to avoid too many sheets
                for label in unique_labels[:20]:
                    label_df = table2_df[table2_df['Label'] == label].copy()
                    label_df = label_df.drop('Label', axis=1)
                    
                    # Clean sheet name
                    sheet_name = f"Label_{label[:25]}"
                    sheet_name = sheet_name.replace('/', '_').replace('\\', '_')
                    
                    label_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Summary statistics
            self._create_summary_stats_sheet(writer, table1_df, table2_df)
            
            # Feature importance (from ablation)
            if self.results['ablation_studies']:
                self._create_feature_importance_sheet(writer)
        
        print(f"✅ Comprehensive comparison tables saved to: {output_path}")
        return output_path
    
    def _create_summary_stats_sheet(self, writer, table1_df, table2_df):
        summary_rows = []
        
        # Best models by different metrics
        if not table1_df.empty:
            for metric in ['Micro_F1', 'Macro_F1', 'Case_Avg_F1', 'Exact_Match_Pct']:
                if metric in table1_df.columns:
                    # Filter out rows with ± (averages)
                    numeric_df = table1_df[~table1_df[metric].astype(str).str.contains('±', na=False)]
                    if not numeric_df.empty:
                        best_idx = numeric_df[metric].astype(float).idxmax()
                        worst_idx = numeric_df[metric].astype(float).idxmin()
                        
                        summary_rows.append({
                            'Metric': f'Best {metric}',
                            'Model': numeric_df.loc[best_idx, 'Model'],
                            'Value': numeric_df.loc[best_idx, metric]
                        })
                        
                        summary_rows.append({
                            'Metric': f'Worst {metric}',
                            'Model': numeric_df.loc[worst_idx, 'Model'],
                            'Value': numeric_df.loc[worst_idx, metric]
                        })
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    
    def _create_feature_importance_sheet(self, writer):
        """Create feature importance sheet from ablation studies."""
        if 'no_ablation' not in self.results['ablation_studies']:
            return
        
        control_f1 = self.results['ablation_studies']['no_ablation']['step_metrics']['SUMMARY']['micro_avg']['f1']
        
        importance_rows = []
        for name, result in self.results['ablation_studies'].items():
            if name == 'no_ablation':
                continue
            
            ablated_f1 = result['step_metrics']['SUMMARY']['micro_avg']['f1']
            delta = ablated_f1 - control_f1
            
            ablated_features = result['ablated_features']
            feature_name = ablated_features[0] if len(ablated_features) == 1 else ', '.join(ablated_features)
            
            importance_rows.append({
                'Feature': feature_name,
                'Control_F1': control_f1,
                'Ablated_F1': ablated_f1,
                'Delta_F1': delta,
                'Impact_Percentage': (delta / control_f1) * 100 if control_f1 > 0 else 0,
                'Importance': 'High' if abs(delta) > 0.05 else 'Medium' if abs(delta) > 0.02 else 'Low'
            })
        
        if importance_rows:
            importance_df = pd.DataFrame(importance_rows)
            importance_df = importance_df.sort_values('Delta_F1', ascending=True)
            importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)