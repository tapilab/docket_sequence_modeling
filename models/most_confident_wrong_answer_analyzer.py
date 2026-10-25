import torch
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models.interpretability import CaseTransformerInterpretability
from training.teacher_forcing import build_teacher_forced_sequences


class MostConfidentWrongAnswerAnalyzer:
    def __init__(
        self,
        model,
        label_processor,
        device,
        verbose: bool = True
    ):
        self.model = model
        self.label_processor = label_processor
        self.device = device
        self.verbose = verbose
        self.label_names = label_processor.label_names

        self.interpreter = CaseTransformerInterpretability(
            model=model,
            label_processor=label_processor,
            verbose=verbose,
            device=device
        )

        print("MostConfidentWrongAnswerAnalyzer initialized.")

    def analyze_mistakes_per_label(self, dataloader, threshold, top_k, top_n_per_label):
        self.model.eval()
        mistakes_by_label = {label: [] for label in self.label_names}
        if self.verbose:
            print("Starting per-label analysis of most confident wrong answers...")
            print(f"Analyzing {len(dataloader)} sequences...")
            print(f"Confidence threshold: {threshold}")
            print(f"Will analyze up to top {top_n_per_label} confident mistakes per label.")
        with torch.no_grad():
            for case_idx, (seq, labels_seq) in enumerate(tqdm(dataloader, disable = not self.verbose)):
                case_id = self.extract_case_id(seq)
                #convert labels to binary format if needed
                if isinstance(labels_seq[0], list):
                    labels_seq_binary = self.label_processor.transform_sequence(labels_seq)
                else:
                    labels_seq_binary = labels_seq

                labels_binaries_list = []
                for lbl in labels_seq_binary:
                    if isinstance(lbl, torch.Tensor):
                        labels_binaries_list.append(lbl.cpu().numpy())
                    else:
                        labels_binaries_list.append(np.array(lbl))
                
                teacher_seq, _ = build_teacher_forced_sequences(seq, labels_binaries_list, self.label_processor, include_initial_state=True)

                logits = self.model(teacher_seq, return_all_timesteps=True)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                for i in range(len(labels_binaries_list)):
                    target_idx = i
                    true_binary = labels_binaries_list[target_idx]
                    k = int(true_binary.sum())
                    if k == 0:
                        continue

                    if i < len(seq):
                        proceeding_id = seq[i].get('proceeding_id', f'proc_{i}')
                    else:
                        proceeding_id = f'proc_{i}'

                    probabilities_vec = probs[target_idx]
                    all_indices = np.argsort(probabilities_vec)[::-1]
                    all_probabilities_sorted = probabilities_vec[all_indices]
                    for pred_idx, pred_prob in zip(all_indices, all_probabilities_sorted):
                        is_wrong = true_binary[pred_idx] == 0
                        is_confident = pred_prob >= threshold
                        if is_wrong and is_confident:
                            wrong_label = self.label_names[pred_idx]
                            true_labels = [self.label_names[j] for j in range(len(true_binary)) if true_binary[j] == 1]
                            top_k_indices = all_indices[:top_k]
                            top_k_probs = all_probabilities_sorted[:top_k]
                            number_of_predictions_to_consider = min(k, top_k)
                            predicted_labels = [self.label_names[idx] for idx in top_k_indices[:number_of_predictions_to_consider]]
                            predicted_confidences = [float(conf) for conf in top_k_probs[:number_of_predictions_to_consider]]
                            mistake_info = ({
                                'case_id': case_id,
                                'case_idx': case_idx,
                                'proceeding_id': proceeding_id,
                                'step_idx': target_idx,
                                'true_labels': true_labels,
                                'predicted_labels': predicted_labels,
                                'confidences': predicted_confidences,
                                'wrong_label': wrong_label,
                                'wrong_label_confidence': float(pred_prob),
                                'sequence_length': len(seq),
                                'num_true_labels': int(k),
                                '_sequence': seq,
                                '_teacher_forced_sequence': teacher_seq,
                                '_true_binary': true_binary
                            })
                            mistakes_by_label[wrong_label].append(mistake_info)
        all_mistakes = []

        for label, mistakes in mistakes_by_label.items():
            if mistakes:
                sorted_mistakes = sorted(mistakes, key=lambda x: x['wrong_label_confidence'], reverse=True)
                top_mistakes = sorted_mistakes[:top_n_per_label]
                all_mistakes.extend(top_mistakes)
                if self.verbose:
                    print(f"Label '{label}': Found {len(mistakes)} confident mistakes, keeping top {len(top_mistakes)}")
        mistakes_df = pd.DataFrame(all_mistakes)
        if not mistakes_df.empty:
            mistakes_df = mistakes_df.sort_values(
                by=['wrong_label', 'wrong_label_confidence'], ascending=[True, False]
            ).reset_index(drop=True)
            if self.verbose:
                print(f"Per-label analysis complete. Found total {len(mistakes_df)} most confident wrong answers across labels.")
        else:
            if self.verbose:
                print("No confident wrong answers found in per-label analysis.")
        return mistakes_df

    def interpret_mistakes(self, mistakes_df) -> pd.DataFrame:
        if mistakes_df.empty:
            if self.verbose:
                print("No mistakes to interpret.")
            return mistakes_df
            
        if self.verbose:
            print("Starting interpretation of mistakes...")
        
        interpreted_rows = []


        for idx, row in tqdm(mistakes_df.iterrows(), total=len(mistakes_df), disable=not self.verbose):
            try:
                teacher_seq = row['_teacher_forced_sequence']
                wrong_label = row['wrong_label']
                target_class_idx = self.label_names.index(wrong_label)
                target_proceeding_idx = row['step_idx']
                
                input_to_explain = teacher_seq[:target_proceeding_idx + 1]  # Adjusting for initial state
                                    
                
                target_labels = np.zeros(len(self.label_names))
                target_labels[target_class_idx] = 1


                attributions, component_attrs = self.interpreter.attribute_sequence_features(
                    seq=input_to_explain,
                    target_class_idx=target_class_idx,
                    target_proceeding_idx=target_proceeding_idx,
                    method="IntegratedGradients",
                    n_steps=20,
                    return_component_attributions=True
                )
                proceeding = {}
                if input_to_explain:
                    proceeding = input_to_explain[-1]
                


                top_features_list = self.extract_top_features(
                    attributions=component_attrs, # Pass raw component_attrs
                    proceeding=proceeding,
                    top_n=5 # Keep top 5
                )

                row_dict = row.to_dict()
                row_dict['top_features'] = top_features_list
                row_dict['top_feature_scores'] = [f['attribution'] for f in top_features_list]
                row_dict['component_importance'] = {f['name']: f['attribution'] for f in top_features_list}
                interpreted_rows.append(row_dict)
            except Exception as e:
                if self.verbose:
                    print(f"Error during interpretation: {e}")
                row_dict = row.to_dict()
                row_dict['top_features'] = []
                row_dict['top_feature_scores'] = []
                row_dict['component_importance'] = {}
                interpreted_rows.append(row_dict)
        return pd.DataFrame(interpreted_rows)

        
    def agg_by_case(self, mistakes_df) -> pd.DataFrame:
        if mistakes_df.empty:
            if self.verbose:
                print("No mistakes to aggregate.")
            return mistakes_df
            
        if self.verbose:
            print("Aggregating mistakes by case...")
            
        agg_data = []
        grouped = mistakes_df.groupby('case_id')
        for case_id, group in tqdm(grouped, disable=not self.verbose):
            wrong_predictions = []
            for _, row in group.iterrows():
                wrong_predictions.append(row['wrong_label'])

            prediction_counts = Counter(wrong_predictions)

            all_true_labels = []
            for _, row in group.iterrows():
                all_true_labels.extend(row['true_labels'])

            missed_label_counts = Counter(all_true_labels)
            common_features = []
            if 'top_features' in group.columns and group['top_features'].notna().any():
                all_features = []
                for features in group['top_features'].dropna():
                    if features:
                        all_features.extend([f['name'] for f in features])
                feature_counts = Counter(all_features)
                common_features = feature_counts.most_common(10)
            average_confidence = group['wrong_label_confidence'].mean()
            case_summary = {
                'case_id': case_id,
                'num_mistakes': len(group),
                'avg_confidence': float(average_confidence),
                'most_common_wrong_predictions': prediction_counts.most_common(5),
                'most_missed_true_labels': missed_label_counts.most_common(5),
                'common_contributing_features': common_features,
                'proceeding_ids': list(group['proceeding_id'].values)
            }
            agg_data.append(case_summary)
        agg_df = pd.DataFrame(agg_data)
        agg_df = agg_df.sort_values(by='num_mistakes', ascending=False).reset_index(drop=True)
        if self.verbose:
            print("Aggregation complete.")
        return agg_df

    def save_report(self, mistake_df, case_report_path):
        if mistake_df.empty:
            if self.verbose:
                print("No mistakes to save.")
            return

        save_path = Path(case_report_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_df = mistake_df.copy()

        # drop internal columns
        internal_cols = [col for col in save_df.columns if col.startswith('_')]
        save_df.drop(columns=internal_cols, inplace=True, errors="ignore")

        # flatten list-ish cols
        for col in ['true_labels', 'predicted_labels', 'confidences']:
            if col in save_df.columns:
                save_df[col] = save_df[col].apply(
                    lambda x: ' | '.join(map(str, x)) if isinstance(x, list) else x
                )

        if 'top_features' in save_df.columns:
                    # This will now save the rich value, e.g., "Case_Embedding(Charges: ... (0.42))"
                    save_df['top_features'] = save_df['top_features'].apply(
                        lambda features: ' | '.join([f"{f['name']}({f['value']}) ({f['attribution']:.4f})" for f in features[:5]]) if isinstance(features, list) and features else ''
                    )

        save_df.to_csv(save_path, index=False)
        if self.verbose:
            print(f"Mistake report saved to {save_path}")
            print(f"Total mistakes saved: {len(save_df)}")

        # also write by-case if we have enough rows
        if len(mistake_df) >= 5:
            case_df = self.agg_by_case(mistake_df)
            case_output_path = save_path.parent / (save_path.stem + "_by_case.csv")
            case_output_path.parent.mkdir(parents=True, exist_ok=True)
            case_save_df = case_df.copy()
            for col in case_save_df.columns:
                if case_save_df[col].dtype == object:
                    case_save_df[col] = case_save_df[col].apply(
                        lambda x: str(x) if not isinstance(x, (list, dict)) else str(x)
                    )
            case_save_df.to_csv(case_output_path, index=False)
            if self.verbose:
                print(f"Case summary report saved to {case_output_path}")
                print(f"Total cases summarized: {len(case_save_df)}")

    def plot_confidences(self, mistakes_df, save_path):
        if mistakes_df.empty:
            print("No data to plot.")
            return
        confidence_column = 'wrong_label_confidence'
        wrong_prediction_column = 'wrong_label'

        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax = axes[0, 0]
        ax.hist(mistakes_df[confidence_column], bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(mistakes_df[confidence_column].mean(), color='red', linestyle='--', 
                  label=f"Mean: {mistakes_df[confidence_column].mean():.3f}")
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Confidence Scores for Wrong Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        wrong_pred_counts = Counter(mistakes_df[wrong_prediction_column].values)
        top_wrong = wrong_pred_counts.most_common(10)
        if top_wrong:
            labels, counts = zip(*top_wrong)
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, counts, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Frequency')
            ax.set_title('Top 10 Most Common Wrong Predictions')
            ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        ax.scatter(mistakes_df['num_true_labels'], mistakes_df[confidence_column],
                  alpha=0.5, s=30)
        ax.set_xlabel('Number of True Labels (k)')
        ax.set_ylabel('Confidence Score')
        ax.set_title('Confidence vs. Complexity (Number of True Labels)')
        ax.grid(True, alpha=0.3)
        if len(mistakes_df['num_true_labels'].unique()) > 1:
            try:
                z = np.polyfit(mistakes_df['num_true_labels'], mistakes_df[confidence_column], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(mistakes_df['num_true_labels'].min(),
                                        mistakes_df['num_true_labels'].max(), 100)
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, label='Trend')
                ax.legend()
            except Exception as e:
                if self.verbose:
                    print(f"Could not plot trend line: {e}")
        
        ax = axes[1, 1]
        ax.scatter(mistakes_df['step_idx'], mistakes_df[confidence_column],
                  alpha=0.5, s=30)
        ax.set_xlabel('Step Index in Sequence')
        ax.set_ylabel('Confidence Score')
        ax.set_title('Confidence vs. Position in Sequence')
        ax.grid(True, alpha=0.3)
        
        if len(mistakes_df) > 10 and len(mistakes_df['step_idx'].unique()) > 1:
            sorted_df = mistakes_df.sort_values('step_idx')
            rolling_mean = sorted_df.groupby('step_idx')[confidence_column].mean()
            if len(rolling_mean) > 1:
                ax.plot(rolling_mean.index, rolling_mean.values, 'r-', alpha=0.5, 
                    label='Average by Step')
                ax.legend()
        
        plt.suptitle(f'Confident Misclassification Analysis (n={len(mistakes_df)})', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def extract_case_id(self, seq):
        """Extract case identifier from sequence"""
        from notebook_helpers import extract_case_info_from_sequence
        
        case_info = extract_case_info_from_sequence(seq, court_data=None)
        mag_num = case_info.get('mag_num', 'unknown')
        case_id = case_info.get('case_id', 'unknown')
        defendant_id = case_info.get('defendant_id', 'unknown')
        
        return f"mag_{mag_num}_case_{case_id}_def_{defendant_id}"
    
    def extract_top_features(self, attributions, proceeding, top_n=10):
            features = []

            # stuff we can pull from the proceeding to make the output readable
            feature_mapping = {
                'defendant_race': self._get_race_name(proceeding.get('race_id', 5)),
                'defendant_gender': self._get_gender_name(proceeding.get('gender_id', 2)),
                'defendant_age': proceeding.get('age', 25),
                'charges': self._format_charges(proceeding.get('charges', {})),
                'proceeding_text': proceeding.get('text', ''),
                'proceeding_id': proceeding.get('proceeding_id', 'unknown'),
                'magistrate_section': proceeding.get('magistrate_section', 'unknown'),
                'court_type': proceeding.get('type', 'mag'),
                'original_bond': proceeding.get('original_bond'),
                'sum_of_bonds': proceeding.get('sum_of_bonds'),
                'prior_case_count': proceeding.get('prior_case_count', 0),
                'prior_charges': self._format_charges(proceeding.get('all_prior_charges', {})),
                'prior_labels': self._format_prior_labels(proceeding.get('true_prior_labels', [])),
                'proceeding_date': str(proceeding.get('date', 'unknown')),
            }

            # case 1: we actually got component attributions from interpreter
            if isinstance(attributions, dict):
                component_names = [
                    'Defendant_Embedding',
                    'Case_Embedding',
                    'BERT_Text_Embedding',
                    'Magistrate_Embedding',
                    'Prior_Label_Embedding',
                    'Prior_Charge_Embedding',
                    'Court_Type',
                    'Original_Bond',
                    'Sum_Of_Bonds',
                    'Prior_Case_Count',
                    'Temporal_Features',
                ]

                # get value or 0 if missing
                scored = []
                for name in component_names:
                    val = attributions.get(name, 0)
                    if isinstance(val, (int, float, np.floating)):
                        score = float(val)
                    else:
                        # sometimes captum gives arrays; take mean abs
                        try:
                            arr = np.asarray(val)
                            score = float(np.mean(np.abs(arr)))
                        except Exception:
                            score = 0.0
                    scored.append((name, score))

                # sort by absolute importance
                scored = sorted(scored, key=lambda x: abs(x[1]), reverse=True)[:top_n]

                for comp_name, score in scored:
                    comp_value = self._get_component_value(comp_name, proceeding, feature_mapping)
                    features.append({
                        'name': comp_name,
                        'value': comp_value,
                        'attribution': score,
                    })

            else:
                # fallback: no structured attributions, just return metadata
                for key in ['court_type', 'original_bond', 'charges', 'magistrate_section', 'prior_case_count']:
                    if key in feature_mapping:
                        features.append({
                            'name': key,
                            'value': feature_mapping[key],
                            'attribution': 0.0
                        })
                    if len(features) >= top_n:
                        break

            return features

    def _get_race_name(self, race_id: int) -> str:
            race_map = {0: 'Black', 1: 'White', 2: 'Hispanic', 3: 'Asian', 4: 'Indigenous', 5: 'Unknown'}
            return race_map.get(race_id, 'Unknown')
    
    def _get_gender_name(self, gender_id: int) -> str:
        gender_map = {0: 'Male', 1: 'Female', 2: 'Unknown'}
        return gender_map.get(gender_id, 'Unknown')

    def _format_charges(self, charges: dict) -> str:
        if not charges:
            return 'No charges'
        charge_list = []
        for charge_id, charge_info in list(charges.items()):
            if isinstance(charge_info, dict):
                name = charge_info.get('name', f'Charge {charge_id}')
                count = charge_info.get('count', 1)
                charge_list.append(f"{name} (x{count})")
            else:
                charge_list.append(str(charge_id))
        return ' | '.join(charge_list)
    
    def _format_prior_labels(self, prior_labels: list) -> str:
        if not prior_labels or not any(prior_labels):
            return 'None'
        active_labels = set()
        for idx, val in enumerate(prior_labels):
            if val == 1 and idx < len(self.label_names):
                active_labels.add(self.label_names[idx])
        return ' | '.join(active_labels) if active_labels else 'None'
    
    def _get_component_value(self, comp_name: str, proceeding: dict, feature_mapping: dict):
        component_to_features = {
            'Defendant_Embedding': f"{feature_mapping['defendant_race']}, {feature_mapping['defendant_gender']}, Age {feature_mapping['defendant_age']}",
            'Case_Embedding': feature_mapping['charges'],
            'BERT_Text_Embedding': feature_mapping['proceeding_text'],
            'Magistrate_Embedding': feature_mapping['magistrate_section'],
            'Prior_Label_Embedding': feature_mapping['prior_labels'],
            'Prior_Charge_Embedding': feature_mapping['prior_charges'],
            'Court_Type': feature_mapping['court_type'],
            'Original_Bond': feature_mapping['original_bond'],
            'Sum_Of_Bonds': feature_mapping['sum_of_bonds'],
            'Prior_Case_Count': f"{feature_mapping['prior_case_count']} prior cases",
            'Temporal_Features': feature_mapping['proceeding_date']
        }
        return component_to_features.get(comp_name, 'N/A')
    
    def get_summary_statistics(self, mistakes_df) -> dict:
        """Get summary statistics organized by label"""
        if mistakes_df.empty:
            return {}
        
        stats = {
            'total_mistakes': len(mistakes_df),
            'unique_cases': mistakes_df['case_id'].nunique(),
            'labels_with_mistakes': mistakes_df['wrong_label'].nunique(),
        }
        
        by_label = {}
        grouped = mistakes_df.groupby('wrong_label')
        for label, group in grouped:
            by_label[label] = {
                'count': len(group),
                'avg_confidence': group['wrong_label_confidence'].mean(),
                'max_confidence': group['wrong_label_confidence'].max(),
                'min_confidence': group['wrong_label_confidence'].min(),
            }
        stats['by_label'] = by_label
        
        stats['avg_confidence'] = mistakes_df['wrong_label_confidence'].mean()
        stats['std_confidence'] = mistakes_df['wrong_label_confidence'].std()
        stats['min_confidence'] = mistakes_df['wrong_label_confidence'].min()
        stats['max_confidence'] = mistakes_df['wrong_label_confidence'].max()
        
        wrong_pred_counts = Counter(mistakes_df['wrong_label'].values)
        stats['most_common_wrong_predictions'] = wrong_pred_counts.most_common(5)
        
        all_true = []
        for labels in mistakes_df['true_labels']:
            all_true.extend(labels)
        true_label_counts = Counter(all_true)
        stats['most_missed_true_labels'] = true_label_counts.most_common(5)
        
        stats['avg_num_true_labels'] = float(mistakes_df['num_true_labels'].mean())
        stats['confidence_by_step'] = mistakes_df.groupby('step_idx')['wrong_label_confidence'].agg(['mean', 'count']).to_dict()

        if self.verbose:
            print("\n" + "=" * 80)
            print("SUMMARY STATISTICS (PER-LABEL ANALYSIS)")
            print("=" * 80)
            print(f"Total confident mistakes: {stats['total_mistakes']}")
            print(f"Unique cases affected: {stats['unique_cases']}")
            print(f"Labels with mistakes: {stats['labels_with_mistakes']}/{len(self.label_names)}")
            print(f"Average confidence on mistakes: {stats['avg_confidence']:.3f} +/- {stats['std_confidence']:.3f}")
            print(f"Confidence range: [{stats['min_confidence']:.3f}, {stats['max_confidence']:.3f}]")
            print(f"\nTop 5 wrong predictions (by label):")
            for label, count in stats['most_common_wrong_predictions']:
                print(f"  - {label}: {count} times")
            print(f"\nTop 5 missed true labels:")
            for label, count in stats['most_missed_true_labels']:
                print(f"  - {label}: {count} times")
        
        return stats

# ---------- Top-level function to run the analysis ----------
def run_confident_misclassification_analysis(
    model,
    label_processor,
    device,
    dataloader,
    save_dir,
    threshold: float = 0.8,
    top_k: int = 5,
    top_n_per_label: int = 20,
    max_sequences: int | None = 100,   # set None to use all
    verbose: bool = True,):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Build analyzer
    analyzer = MostConfidentWrongAnswerAnalyzer(
        model=model,
        label_processor=label_processor,
        device=device,
        verbose=verbose,
    )
    data_iter = dataloader
    if max_sequences is not None:
        try:
            # Works if it's a list/Sequence
            data_iter = dataloader[:max_sequences]
        except TypeError:
            # Fallback for generic iterables
            from itertools import islice
            data_iter = list(islice(dataloader, max_sequences))
    
    # ---------- Run analysis ----------
    mistakes_report = analyzer.analyze_mistakes_per_label(
        dataloader=data_iter,
        threshold=threshold,
        top_k=top_k,
        top_n_per_label=top_n_per_label,
    )

    results = {
        'mistakes_report': mistakes_report,
        'interpreted_mistakes': None,
        'case_patterns': None,
        'summary_stats': None,
        'report_path': None,
        'plot_path': None,
        'by_case_path': None,
    }
    if mistakes_report is None or getattr(mistakes_report, "empty", True):
        if verbose:
            print("No confident misclassifications found with the current settings.")
        return results
    if verbose:
        print("\n" + "="*80)
        print("TOP CONFIDENT MISCLASSIFICATIONS PER LABEL")
        print("="*80)
        
        for label in label_processor.label_names:
            label_mistakes = mistakes_report[mistakes_report['wrong_label'] == label]
            if not label_mistakes.empty:
                print(f"\n{label} (top {min(3, len(label_mistakes))}):")
                display_cols = ['case_id', 'proceeding_id', 'wrong_label_confidence', 'true_labels']
                cols = [c for c in display_cols if c in label_mistakes.columns]
                print(label_mistakes[cols].head(3).to_string(index=False))    
    try:
        summary_stats = analyzer.get_summary_statistics(mistakes_report)
    except AttributeError:
        summary_stats = None
    results['summary_stats'] = summary_stats

    # Plot confidences
    plot_path = save_dir / "confident_mistakes_plot.png"
    try:
        analyzer.plot_confidences(mistakes_report, save_path=plot_path)
        results['plot_path'] = plot_path
    except Exception as e:
        if verbose:
            print(f"(warn) Could not plot confidences: {e}")
    
    # ---------- Interpretability ----------
    if verbose:
        print("Running interpretability analysis on confident mistakes...")
    try:
        interpreted_mistakes = analyzer.interpret_mistakes(mistakes_report)
        results['interpreted_mistakes'] = interpreted_mistakes
    except Exception as e:
        interpreted_mistakes = None
        if verbose:
            print(f"(warn) Interpretability step failed: {e}")

    if verbose and interpreted_mistakes is not None and not interpreted_mistakes.empty:
        if 'top_features' in interpreted_mistakes.columns:
            print("\n" + "="*80)
            print("EXAMPLES WITH CONTRIBUTING FEATURES")
            print("="*80)
            top_interpreted = interpreted_mistakes.sort_values(by='wrong_label_confidence', ascending=False)
            for _, row in top_interpreted.head(3).iterrows():
                print(f"\nCase: {row.get('case_id')}")
                print(f"Proceeding: {row.get('proceeding_id')}")
                mc = row.get('wrong_label_confidence')
                if mc is not None:
                    print(f"Confidence: {mc:.3f}")
                print(f"Wrong Prediction: {row.get('wrong_label')}")
                print(f"True Labels: {row.get('true_labels')}")
                feats = row.get('top_features') or []
                if feats:
                    print("Top Contributing Features:")
                    # This print loop will now show the rich values
                    for feat in feats[:3]:
                        name = feat.get('name')
                        val = feat.get('value')
                        attr = feat.get('attribution')
                        if isinstance(attr, (float, int)):
                            print(f"  - {name}: {val} (attribution: {attr:.3f})")
                        else:
                            print(f"  - {name}: {val} (attribution: {attr})")
                print("-" * 40)
    
    # ---------- Aggregate by case ----------
    case_patterns = None
    df_to_agg = interpreted_mistakes if interpreted_mistakes is not None and not interpreted_mistakes.empty else mistakes_report

    try:
        case_patterns = analyzer.agg_by_case(df_to_agg)
        results['case_patterns'] = case_patterns
        if verbose and not case_patterns.empty:
            print("\n" + "="*80)
            print("CASES WITH MOST CONFIDENT MISTAKES")
            print("="*80)
            to_show = min(5, len(case_patterns))
            for _, case in case_patterns.head(to_show).iterrows():
                print(f"\nCase ID: {case.get('case_id')}")
                print(f"Number of Mistakes: {case.get('num_mistakes')}")
                ac = case.get('avg_confidence')
                if isinstance(ac, (float, int)):
                    print(f"Average Confidence: {ac:.3f}")
                mcwp = case.get('most_common_wrong_predictions') or []
                if mcwp:
                    print("Most Common Wrong Predictions:")
                    for label, count in mcwp[:3]:
                        print(f"  - {label}: {count} times")
                ccf = case.get('common_contributing_features') or []
                if ccf:
                    print("Common Contributing Features:")
                    for feat, count in ccf[:3]:
                        print(f"  - {feat}: appeared {count} times")
                print("-" * 40)
    except Exception as e:
        if verbose:
            print(f"(warn) Aggregation by case failed: {e}")
    # ---------- Save detailed report(s) ----------
    report_path = save_dir / "confident_misclassifications_report.csv"
    try:
        df_to_save = interpreted_mistakes if interpreted_mistakes is not None and not interpreted_mistakes.empty else mistakes_report
        analyzer.save_report(df_to_save, case_report_path=str(report_path))
        results['report_path'] = report_path
    except Exception as e:
        if verbose:
            import traceback
            print(f"(warn) Could not save main report: {e}")
            print("Hitting traceback:")
            traceback.print_exc()
    
    by_case_path = None
    if case_patterns is not None and not case_patterns.empty:
        by_case_path = report_path.parent / (report_path.stem + "_by_case.csv")
        try:
            case_patterns.to_csv(by_case_path, index=False)
            results['by_case_path'] = by_case_path
        except Exception as e:
            if verbose:
                print(f"(warn) Could not save by-case report: {e}")
    # ---------- Insights ----------
    if verbose and results['summary_stats']:
        summary_stats = results['summary_stats']
        print("\n" + "="*80)
        print("KEY INSIGHTS FROM CONFIDENT MISCLASSIFICATION ANALYSIS")
        print("="*80)
        # Systematic label issue
        top_wrong = summary_stats.get('most_common_wrong_predictions')
        if top_wrong:
            top_label, top_count = top_wrong[0]
            total = len(mistakes_report)
            if total:
                frac = top_count / total
                if frac > 0.2:
                    print("SYSTEMATIC BIAS DETECTED:")
                    print(f"   Frequent high-confidence wrong label: '{top_label}'.")
                    print(f"   {top_count} of {total} mistakes ({frac*100:.1f}%).")
                    print("   Recommendation: re-check label distribution and hard-negatives for this label.")
        # Sequence position effect
        conf_by_step = summary_stats.get('confidence_by_step', {})
        means = conf_by_step.get('mean') if conf_by_step else None
        if means:
            early = [v for k, v in means.items() if k <= 2]
            late = [v for k, v in means.items() if k > 5]
            if early and late:
                early = [v for k,v in means.items() if k <=2]
                late = [v for k,v in means.items() if k >5]
                if early and late:
                    early_avg = sum(early) / len(early)
                    late_avg = sum(late) / len(late)
                    if abs(early_avg - late_avg) > 0.1:
                        print("\n SEQUENCE POSITION EFFECT:")
                        print(f"   Early steps (0-2): {early_avg:.3f}  |  Late steps (>5): {late_avg:.3f}")
                        if early_avg > late_avg:
                            print("   Model is more confidently wrong early in sequences.")
                        else:
                            print("   Model gets more confidently wrong later in sequences.")
        # Complexity (multi-label density)
        avg_k = summary_stats.get('avg_num_true_labels', 0)
        if isinstance(avg_k, (float, int)) and avg_k > 3:
            print("\n COMPLEXITY CHALLENGE:")
            print(f"   Mistakes skew toward complex steps (avg {avg_k:.1f} true labels).")
            print("   Recommendation: augment multi-label examples / adjust loss for label density.")
        print("\n" + "="*80)
    # Final console summary
    if verbose:
        print(f"\nAnalysis complete.")
        print(f"Found mistakes for {summary_stats['labels_with_mistakes']}/{len(analyzer.label_names)} labels.")
        print(f"Found {len(mistakes_report)} confident misclassifications.")
        if results['report_path']:
            print(f"Detailed report: {results['report_path']}")
        if results['by_case_path']:
            print(f"Case summary:   {results['by_case_path']}")
        if results['plot_path']:
            print(f"Confidence plot: {results['plot_path']}")
    return results