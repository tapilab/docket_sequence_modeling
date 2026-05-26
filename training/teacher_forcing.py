import torch

def build_teacher_forced_sequences(seq, labels_seq_binary, label_processor, include_initial_state=True):
    """
    Forecasting/No-Leak Alignment:
    Input[0] (Dummy) -> Predicts Label[0]
    Input[k] (Seq[k-1]) -> Predicts Label[k]
    """
    T = len(seq)
    if T == 0:
        return [], torch.empty(0)

    num_labels = len(label_processor.label_names)
    teacher_forced_seq = []
    targets = []
    
    def to_tensor(labels):
        if isinstance(labels, torch.Tensor):
            return labels.float()
        return torch.tensor(labels, dtype=torch.float32)
    if include_initial_state:
        initial_state = dict(seq[0])
        initial_state['proceeding_id'] = f'initial_{seq[0].get("defendant_id", "unknown")}'
        initial_state['text'] = 'INITIAL_STATE'
        initial_state['original_bond'] = None
        initial_state['sum_of_bonds'] = None
        initial_state['true_prior_labels'] = [0] * num_labels
        
        teacher_forced_seq.append(initial_state)
        targets.append(to_tensor(labels_seq_binary[0]))
    
    for k in range(T - 1):
        proceeding = dict(seq[k])
        
        # The history available at this step is the label of the event we just saw (Label k)
        prev_label = labels_seq_binary[k]
        proceeding['true_prior_labels'] = prev_label.tolist() if hasattr(prev_label, 'tolist') else list(prev_label)
        
        teacher_forced_seq.append(proceeding)
        targets.append(to_tensor(labels_seq_binary[k+1]))
        
    targets = torch.stack(targets).unsqueeze(0)  # (1, T, num_labels)
    return teacher_forced_seq, targets