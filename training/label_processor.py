# Python File that contains the class that does label processing with MultiLabelBinarizer

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import torch

class EnhancedLabelProcessor:
    def __init__(self, predefined_label_names):
        self.predefined_labels = predefined_label_names
        self.mlb = MultiLabelBinarizer(classes=predefined_label_names)
        self.label_names = predefined_label_names
        self.is_fitted = False
        
    def fit(self, label_sequences):
        """
        Fit MultiLabelBinarizer with your predefined 21 labels
        """
        # Flatten all label sets from all sequences
        all_label_sets = []
        for sequence in label_sequences:
            for step_labels in sequence:
                if isinstance(step_labels, (list, set)):
                    # Filter to only include your 21 predefined labels
                    filtered_labels = [label for label in step_labels if label in self.predefined_labels]
                    all_label_sets.append(filtered_labels)
                else:
                    all_label_sets.append([])
        
        # Fit with your predefined classes
        self.mlb.fit(all_label_sets)
        self.is_fitted = True
        print(f"✅ Fitted MultiLabelBinarizer with {len(self.label_names)} predefined classes:")
        print(f"   Labels: {self.label_names}")
        return self
    
    def transform_sequence(self, label_sequence):
        """Convert sequence of string labels to binary matrix"""
        if not self.is_fitted:
            raise ValueError("Must call fit() before transform()")
        return self.mlb.transform(label_sequence)
    
    def inverse_transform_step(self, binary_vector):
        """Convert binary vector back to label names"""
        if isinstance(binary_vector, torch.Tensor):
            binary_vector = binary_vector.cpu().numpy()
        binary_vector = binary_vector.reshape(1, -1)
        return self.mlb.inverse_transform(binary_vector)[0]