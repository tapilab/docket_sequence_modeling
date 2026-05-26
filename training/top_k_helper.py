import torch
import numpy as np

def predict_with_top_k_mlb(probs, k, label_processor):
    num_labels = len(label_processor.label_names)
    mask = np.zeros(num_labels)

    # If k > 0, pick the top-k indices
    if k > 0:
        # Convert to numpy if it's a torch tensor
        if isinstance(probs, torch.Tensor):
            probs_np = probs.cpu().numpy()
        else:
            probs_np = np.array(probs)
        
        # Get top-k indices
        k_to_select = min(k, num_labels)  # Don't select more than available labels
        top_indices = np.argsort(probs_np)[-k_to_select:]
        mask[top_indices] = 1

    return mask