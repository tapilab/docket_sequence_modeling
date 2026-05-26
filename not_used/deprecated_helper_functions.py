def build_prediction_prefix(sequence, labels_binary, step_idx, label_processor):
    if step_idx == -1:
        # Initial state prediction
        prefix = [dict(sequence[0])]
        prefix[0]['proceeding_id'] = f'initial_{sequence[0].get("defendant_id", "unknown")}'
        prefix[0]['text'] = 'INITIAL_STATE'
        prefix[0]['original_bond'] = None
        prefix[0]['sum_of_bonds'] = None
        prefix[0]['true_prior_labels'] = [0] * len(label_processor.label_names)
        target_idx = 0
        step_desc = "Initial State at index 0"
    else:
        # Regular step prediction
        prefix = [dict(p) for p in sequence[:step_idx + 1]]
        if step_idx > 0:
            prefix[-1]['true_prior_labels'] = labels_binary[step_idx - 1].tolist()
        target_idx = step_idx + 1
        step_desc = f"Step {step_idx} at index {target_idx}"
    
    return prefix, target_idx, step_desc