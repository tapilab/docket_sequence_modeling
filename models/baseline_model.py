import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import pandas as pd

from config import COMPLETION_LABELS

# Baseline Model
class BaselineNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class BaselineOutcomePredictor:
    def __init__(self, label_processor, device=None):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.label_processor = label_processor
        
        # Create label name to index mapping from label_processor
        self.label_to_idx = {name: i for i, name in enumerate(self.label_processor.label_names)}
        
        # Filter completion labels to only those that exist in label_processor
        self.completion_labels = [label for label in COMPLETION_LABELS 
                                if label in self.label_to_idx]
        
        print(f"🎯 Baseline model will predict {len(self.completion_labels)} completion outcomes:")
        for i, label in enumerate(self.completion_labels):
            mlb_idx = self.label_to_idx[label]
            print(f"   {i}: {label} (MLB index: {mlb_idx})")
    
    def _get_final_outcome(self, labels_seq):
        """
        Extract the final completion outcome from a sequence of binary tensors
        Uses label_processor's internal mapping to find completion labels in the binary vectors
        """
        # Iterate backwards through the sequence to find the final outcome
        for step_labels in reversed(labels_seq):
            # Convert tensor to numpy if needed
            if hasattr(step_labels, 'cpu'):
                step_labels = step_labels.cpu().numpy()
            elif isinstance(step_labels, torch.Tensor):
                step_labels = step_labels.numpy()
            
            # Ensure it's a flat numpy array
            step_labels = np.array(step_labels).flatten()
            
            # Check for any completion label using label_processor's indexing
            for completion_label in self.completion_labels:
                label_idx = self.label_to_idx[completion_label]
                if label_idx < len(step_labels) and step_labels[label_idx] == 1:
                    return completion_label
        return None

    def extract_features_for_baseline_model(self, sequence, labels_seq):
        """
        Extract features from first proceeding and determine final completion outcome
        """
        if not sequence or not labels_seq:
            return None, None
        
        initial_proceeding = sequence[0]
        final_outcome = self._get_final_outcome(labels_seq)
        if final_outcome is None:
            return None, None

        features = {}
        features['defendant_age'] = initial_proceeding.get('age', 25.0)
        features['defendant_race_id'] = initial_proceeding.get('race_id', 5)
        features['defendant_gender_id'] = initial_proceeding.get('gender_id', 2)
        features['court_type'] = 1 if initial_proceeding.get('type') == 'cdc' else 0
        features['magistrate_section'] = str(initial_proceeding.get('magistrate_section', 'unknown'))
        features['prior_case_count'] = float(initial_proceeding.get('prior_case_count', 0))

        # charges
        charges = initial_proceeding.get('charges', {})
        features['num_charges'] = len(charges)
        features['total_charge_count'] = sum(charge.get('count', 1) for charge in charges.values())

        # priors
        prior_charges = initial_proceeding.get('all_prior_charges', {})
        features['num_prior_charges'] = len(prior_charges)
        features['total_prior_charge_count'] = sum(charge.get('count', 0) for charge in prior_charges.values())

        return features, final_outcome

    def prepare_training_data(self, sequences):
        """
        Prepare features and labels from sequence data
        sequences: list of (seq, labels_seq_tensors) tuples
        """
        features_list, labels_list = [], []
        skipped_no_outcome = 0
        skipped_no_features = 0
        
        for seq, labels_seq in tqdm(sequences, desc="Extracting baseline features"):
            features, outcome = self.extract_features_for_baseline_model(seq, labels_seq)
            if features is None:
                skipped_no_features += 1
                continue
            if outcome is None:
                skipped_no_outcome += 1
                continue
            
            features_list.append(features)
            labels_list.append(outcome)
        
        print(f"📊 Feature extraction results:")
        print(f"   Valid samples: {len(features_list)}")
        print(f"   Skipped (no features): {skipped_no_features}")
        print(f"   Skipped (no completion outcome): {skipped_no_outcome}")
        
        if not features_list:
            raise ValueError("No valid training samples found! Check if your sequences have completion outcomes.")
        
        features_df = pd.DataFrame(features_list).fillna(0)

        # Handle categorical encoding for magistrate_section
        if 'magistrate_section' in features_df.columns:
            if 'magistrate_section' not in self.label_encoders:
                self.label_encoders['magistrate_section'] = LabelEncoder()
                # CRITICAL FIX: Ensure 'unknown' is always a valid category
                all_values = features_df['magistrate_section'].fillna('unknown')
                unique_categories = list(all_values.unique())
                if 'unknown' not in unique_categories:
                    unique_categories.append('unknown')
                
                self.label_encoders['magistrate_section'].fit(unique_categories)
                features_df['magistrate_section'] = self.label_encoders['magistrate_section'].transform(
                    features_df['magistrate_section'].fillna('unknown')
                )
            else:
                # Handle unseen categories during prediction
                unknown_mask = ~features_df['magistrate_section'].fillna('unknown').isin(
                    self.label_encoders['magistrate_section'].classes_
                )
                features_df.loc[unknown_mask, 'magistrate_section'] = 'unknown'
                features_df['magistrate_section'] = self.label_encoders['magistrate_section'].transform(
                    features_df['magistrate_section'].fillna('unknown')
                )
        
        self.feature_names = list(features_df.columns)
        
        # Show outcome distribution
        from collections import Counter
        outcome_counts = Counter(labels_list)
        print(f"\n📈 Completion outcome distribution:")
        for outcome, count in sorted(outcome_counts.items()):
            pct = count / len(labels_list) * 100
            print(f"   {outcome}: {count} ({pct:.1f}%)")
        
        return features_df.values, np.array(labels_list)

    def train(self, sequences, epochs=20, batch_size=32, lr=1e-3):
        """Train the baseline model"""
        print(f"🚀 Training baseline model on {len(sequences)} sequences...")
        
        X, y = self.prepare_training_data(sequences)
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Unique outcomes: {len(set(y))}")
        
        # Map string outcomes to indices using our completion_labels order
        class_to_idx = {label: i for i, label in enumerate(self.completion_labels)}
        
        # Filter out any outcomes not in our completion set (shouldn't happen)
        valid_mask = [outcome in class_to_idx for outcome in y]
        if not all(valid_mask):
            print(f"⚠️  Warning: {sum(~np.array(valid_mask))} samples have outcomes not in COMPLETION_LABELS")
            X = X[valid_mask]
            y = y[valid_mask]
        
        y_idx = np.array([class_to_idx[label] for label in y])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Initialize model
        input_dim = X_scaled.shape[1]
        output_dim = len(self.completion_labels)
        self.model = BaselineNN(input_dim, output_dim).to(self.device)

        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(y_idx, dtype=torch.long)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"   Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

    def predict(self, sequences):
        """Make predictions on sequences"""
        if self.model is None:
            raise ValueError("Model must be trained before prediction!")
            
        X, y = self.prepare_training_data(sequences)
        X_scaled = self.scaler.transform(X)
        
        # Map string outcomes to indices for true labels
        class_to_idx = {label: i for i, label in enumerate(self.completion_labels)}
        
        # Filter valid outcomes
        valid_mask = [outcome in class_to_idx for outcome in y]
        if not all(valid_mask):
            print(f"⚠️  Warning: {sum(~np.array(valid_mask))} test samples have invalid outcomes")
            X_scaled = X_scaled[valid_mask]
            y = y[valid_mask]
        
        y_idx = np.array([class_to_idx[label] for label in y])

        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X_scaled, dtype=torch.float32).to(self.device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        return preds, probs, y_idx

    def evaluate(self, sequences):
        """Evaluate model performance"""
        preds, probs, y_true = self.predict(sequences)
        
        # Explicitly specify ALL completion labels to get metrics for all classes
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, preds, 
            labels=list(range(len(self.completion_labels))),  # Force all classes
            average=None, 
            zero_division=0
        )
        
        macro_f1 = np.mean(f1)
        weighted_f1 = np.average(f1, weights=support)
        acc = (preds == y_true).mean()
        
        print(f"\n🎯 BASELINE MODEL RESULTS:")
        print(f"   Overall Accuracy: {acc:.3f}")
        print(f"   Macro F1: {macro_f1:.3f}")
        print(f"   Weighted F1: {weighted_f1:.3f}")
        
        print(f"\n📊 Per-Class Results (only showing classes with samples):")
        for i, (label, prec, rec, f1_score, supp) in enumerate(zip(
            self.completion_labels, precision, recall, f1, support
        )):
            if supp > 0:  # Only show classes that actually appeared
                print(f"   {label[:40]:<40}: P={prec:.3f} R={rec:.3f} F1={f1_score:.3f} N={int(supp)}")
        
        preds_binary = []
        for pred_idx in preds:
            pred_binary = np.zeros(len(self.label_processor.label_names), dtype=int)
            # Map from completion_labels index to full label_processor index
            completion_label = self.completion_labels[pred_idx]
            full_label_idx = self.label_to_idx[completion_label]
            pred_binary[full_label_idx] = 1
            preds_binary.append(pred_binary)
        
        preds_binary = np.array(preds_binary)

        return dict(
            accuracy=acc, 
            macro_f1=macro_f1, 
            weighted_f1=weighted_f1,
            precision=precision, 
            recall=recall, 
            f1=f1, 
            support=support,
            class_names=self.completion_labels,
            preds=preds,
            preds_binary=preds_binary,
            probabilities=probs,
        )


def run_baseline(train_sequences, val_sequences, label_processor, epochs=20, batch_size=32, lr=1e-3):
    """
    Main function to run baseline model training and evaluation
    
    Args:
        train_sequences: List of (sequence, labels_seq_tensors) tuples for training
        val_sequences: List of (sequence, labels_seq_tensors) tuples for validation
        label_processor: EnhancedLabelProcessor instance with fitted MultiLabelBinarizer
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        
    Returns:
        baseline: Trained BaselineOutcomePredictor
        results: Evaluation results dictionary
    """
    print(f"=" * 80)
    print(f"BASELINE MODEL COMPARISON")
    print(f"=" * 80)
    print(f"🚀 Starting baseline model training...")
    print(f"   Training sequences: {len(train_sequences)}")
    print(f"   Validation sequences: {len(val_sequences)}")
    print(f"   Label processor has {len(label_processor.label_names)} labels")
    print(f"   Will predict final case completion outcomes")
    
    # Initialize predictor with label_processor
    baseline = BaselineOutcomePredictor(label_processor)

    # Train it
    baseline.train(train_sequences, epochs=epochs, batch_size=batch_size, lr=lr)

    # Evaluate on validation
    print(f"\n📊 VALIDATION PERFORMANCE:")
    results = baseline.evaluate(val_sequences)

    print(f"\n✅ Baseline model comparison completed!")
    return baseline, results

def convert_baseline_results_to_table_format(baseline_results, total_samples=None):
    if total_samples is None:
        total_samples = int(baseline_results['support'].sum())
    
    # Calculate micro averages (weighted by support)
    support = baseline_results['support']
    total_support = support.sum()
    
    micro_precision = (baseline_results['precision'] * support).sum() / total_support if total_support > 0 else 0
    micro_recall = (baseline_results['recall'] * support).sum() / total_support if total_support > 0 else 0
    micro_f1 = (baseline_results['f1'] * support).sum() / total_support if total_support > 0 else 0
    
    # Calculate macro averages (simple mean)
    macro_precision = baseline_results['precision'].mean()
    macro_recall = baseline_results['recall'].mean()
    macro_f1 = baseline_results['macro_f1']
    
    # Calculate weighted averages
    weighted_precision = (baseline_results['precision'] * support).sum() / total_support if total_support > 0 else 0
    weighted_recall = (baseline_results['recall'] * support).sum() / total_support if total_support > 0 else 0
    weighted_f1 = baseline_results['weighted_f1']
    
    # Build the formatted result
    formatted_results = {
        'SUMMARY': {
            'micro_avg': {
                'f1': float(micro_f1),
                'precision': float(micro_precision),
                'recall': float(micro_recall)
            },
            'macro_avg': {
                'f1': float(macro_f1),
                'precision': float(macro_precision),
                'recall': float(macro_recall)
            },
            'weighted_avg': {
                'f1': float(weighted_f1),
                'precision': float(weighted_precision),
                'recall': float(weighted_recall)
            },
            'total_samples': int(total_samples)
        },
        'INDIVIDUAL_LABELS': {}
    }
    
    # Add per-class metrics
    class_names = baseline_results['class_names']
    precision = baseline_results['precision']
    recall = baseline_results['recall']
    f1 = baseline_results['f1']
    support_arr = baseline_results['support']
    
    for i, class_name in enumerate(class_names):
        # Calculate TP, FP, FN from precision, recall, and support
        true_positives = int(recall[i] * support_arr[i])
        false_negatives = int(support_arr[i] - true_positives)
        
        # From precision: TP / (TP + FP) = precision
        # So: TP + FP = TP / precision (if precision > 0)
        if precision[i] > 0:
            false_positives = int((true_positives / precision[i]) - true_positives)
        else:
            false_positives = 0
        
        formatted_results['INDIVIDUAL_LABELS'][class_name] = {
            'f1': float(f1[i]),
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'support': int(support_arr[i]),
            'tp': true_positives,
            'fp': false_positives,
            'fn': false_negatives
        }
    
    return formatted_results