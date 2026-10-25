import torch
import torch.nn as nn
import random
import json
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR


from training.evaluation import evaluate_with_teacher_forcing
from training.teacher_forcing import build_teacher_forced_sequences
class logging_tool:    
    """Logger for tracking training metrics and history."""
    
    def __init__(self):
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'micro_f1': [],        # Track micro F1 separately
            'micro_precision': [], # Track micro precision separately  
            'micro_recall': [],     # Track micro recall separately
            'learning_rate': []
        }
    
    def log_epoch(self, epoch, train_loss, val_loss=None, val_metrics=None, current_lr=None):
        """Log metrics for a single epoch."""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_metrics'].append(val_metrics)
        self.history['learning_rate'].append(current_lr)
        
        # Extract micro-averaged metrics if available
        if val_metrics and 'SUMMARY' in val_metrics:
            micro_metrics = val_metrics['SUMMARY']['micro_avg']
            self.history['micro_f1'].append(micro_metrics['f1'])
            self.history['micro_precision'].append(micro_metrics['precision'])
            self.history['micro_recall'].append(micro_metrics['recall'])
        else:
            self.history['micro_f1'].append(None)
            self.history['micro_precision'].append(None)
            self.history['micro_recall'].append(None)
    
    def save_history(self, filepath):
        """Save training history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
    
    def plot_learning_curves(self, save_path=None):
        """Plot learning curves for loss and metrics."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 14))
        fig.suptitle('Training Learning Curves (Micro-Averaged Metrics)', fontsize=16)
        
        epochs = self.history['epoch']
        
        # 1. Loss curves
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        if any(v is not None for v in self.history['val_loss']):
            val_losses = [v for v in self.history['val_loss'] if v is not None]
            val_epochs = [e for e, v in zip(epochs, self.history['val_loss']) if v is not None]
            axes[0, 0].plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. F1 Score
        if any(v is not None for v in self.history['micro_f1']):
            f1_scores = [v for v in self.history['micro_f1'] if v is not None]
            f1_epochs = [e for e, v in zip(epochs, self.history['micro_f1']) if v is not None]
            axes[0, 1].plot(f1_epochs, f1_scores, 'g-', label='Micro F1', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].set_title('Micro-Averaged F1 Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision & Recall
        if any(v is not None for v in self.history['micro_precision']):
            precision_scores = [v for v in self.history['micro_precision'] if v is not None]
            recall_scores = [v for v in self.history['micro_recall'] if v is not None]
            pr_epochs = [e for e, v in zip(epochs, self.history['micro_precision']) if v is not None]
            axes[1, 0].plot(pr_epochs, precision_scores, 'b-', label='Precision', linewidth=2)
            axes[1, 0].plot(pr_epochs, recall_scores, 'r-', label='Recall', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('Micro-Averaged Precision & Recall')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Combined micro metrics
        if any(v is not None for v in self.history['micro_f1']):
            axes[1, 1].plot(f1_epochs, f1_scores, 'g-', label='F1', linewidth=2)
            if any(v is not None for v in self.history['micro_precision']):
                axes[1, 1].plot(pr_epochs, precision_scores, 'b-', label='Precision', linewidth=2)
                axes[1, 1].plot(pr_epochs, recall_scores, 'r-', label='Recall', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('All Micro-Averaged Metrics')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Learning Rate
        if self.history['learning_rate']:
            steps = range(len(self.history['learning_rate']))
            axes[2, 0].plot(steps, self.history['learning_rate'], 'orange', linewidth=2)
            axes[2, 0].set_xlabel('Training Step')
            axes[2, 0].set_ylabel('Learning Rate')
            axes[2, 0].set_title('Learning Rate Schedule')
            #use log scale for learning rate since onecycle can vary widely
            axes[2, 0].set_yscale('log')
            axes[2, 0].grid(True, alpha=0.3)

        # 6. Difference between Train and Val Loss
        if any(v is not None for v in self.history['val_loss']):
            train_losses = [self.history['train_loss'][i] for i, v in enumerate(self.history['val_loss']) if v is not None]
            val_losses = [v for v in self.history['val_loss'] if v is not None]
            gap = [v - t for v, t in zip(val_losses, train_losses)]
            gap_epochs = [e for e, v in zip(epochs, self.history['val_loss']) if v is not None]
            
            axes[2, 1].plot(gap_epochs, gap, 'purple', linewidth=2, label='Val - Train Loss')
            axes[2, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            axes[2, 1].fill_between(gap_epochs, 0, gap, alpha=0.3, color='purple')
            axes[2, 1].set_xlabel('Epoch')
            axes[2, 1].set_ylabel('Loss Gap')
            axes[2, 1].set_title('Overfitting Monitor (Val - Train Loss)')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)      
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curves saved to: {save_path}")
        
        plt.show()

def calculate_positive_class_weights(sequences_and_labels, label_processor):
    label_counts = torch.zeros(len(label_processor.label_names))
    total_samples = 0
    for seq, labels_seq_raw in sequences_and_labels:
        labels_seq_binary = label_processor.transform_sequence(labels_seq_raw)
        for step_labels in labels_seq_binary:
            label_counts += torch.tensor(step_labels, dtype=torch.float32)
            total_samples += 1
    
    positive_weight = (total_samples - label_counts) / (label_counts + 1e-6)
    positive_weight = torch.clamp(positive_weight, min = 0.1, max = 100.0)
    return positive_weight



def train_case_transformer_with_mlb(
    model,
    sequences_and_labels,  # List of (sequence, raw_string_labels) tuples
    label_processor,       # Your EnhancedLabelProcessor instance
    val_data=None,         # Optional validation data in same format
    epochs=10,
    lr=2e-5,
    weight_decay=0.01,
    batch_size=8,
    device=None,
    directory_for_output=None,
    prediction_level='proceeding', # 'proceeding' or 'case'
    optional_tag=None,
    use_mixed_precision=False,
    use_learning_rate_scheduler=False,
    gradient_clip_norm=1.0,
    weighted_loss=False,
    cache_manager = None
):
    """
    Train the case transformer model with MultiLabelBinarizer.
    
    Args:
        model: The CaseTransformerModel to train
        sequences_and_labels: List of (sequence, raw_string_labels) tuples
        label_processor: EnhancedLabelProcessor instance
        val_data: Optional validation data in same format
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        batch_size: Batch size for training
        device: Device to train on
        directory_for_output: Directory to save outputs
    
    Returns:
        Tuple of (trained_model, logger)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)
    
    # Update model's classifier if needed to match label processor size
    if hasattr(model, 'classifier') and model.classifier is not None:
        if model.classifier.out_features != len(label_processor.label_names):
            print(f"Updating model classifier: {model.classifier.out_features} -> {len(label_processor.label_names)} labels")
            model.classifier = nn.Linear(
                model.classifier.in_features, 
                len(label_processor.label_names)
            ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if weighted_loss:
        positive_class_weights = calculate_positive_class_weights(sequences_and_labels, label_processor).to(device)
    criterion = nn.BCEWithLogitsLoss() if not weighted_loss else nn.BCEWithLogitsLoss(pos_weight=positive_class_weights)

    scaler = GradScaler() if use_mixed_precision else None

    scheduler = None
    if use_learning_rate_scheduler:
        steps_per_epoch = (len(sequences_and_labels) + batch_size - 1) // batch_size
        total_steps = steps_per_epoch * epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr * 5,  # OneCycle goes up to 5x base LR
            total_steps=total_steps,
            pct_start=0.3,  # 30% warmup
            anneal_strategy='cos',
            div_factor=25.0,  # Initial LR = max_lr / 25
            final_div_factor=1e4  # Final LR = max_lr / 10000
        )
        print(f"\nLR Scheduler: OneCycleLR")
        print(f"   Steps per epoch: {steps_per_epoch}")
        print(f"   Total steps: {total_steps}")
        print(f"   Max LR: {lr * 5:.2e}")


    logger = logging_tool()

    if directory_for_output:
        directory_for_output = Path(directory_for_output)
        directory_for_output.mkdir(parents=True, exist_ok=True)

    def print_memory(label=''):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 1024**2
            reserved = torch.cuda.memory_reserved(device) / 1024**2
            print(f"{label} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

    print_memory("Before training")
    print(f"Starting training with MultiLabelBinarizer")
    print(f"   Label processor: {len(label_processor.label_names)} labels")
    print(f"   Training sequences: {len(sequences_and_labels)}")
    print(f"   Validation sequences: {len(val_data) if val_data else 'None'}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_steps = 0
        total_timesteps = 0
        print_memory(f"Epoch {epoch}/{epochs}")

        random.shuffle(sequences_and_labels)
        epoch_pbar = tqdm(range(0, len(sequences_and_labels), batch_size), 
                         desc=f"Epoch {epoch}/{epochs}")
        
        for batch_start in epoch_pbar:
            batch = sequences_and_labels[batch_start: batch_start + batch_size]
            optimizer.zero_grad()
            batch_loss = 0.0
            batch_timesteps = 0

            for seq, labels_seq_raw in batch:
                # Convert raw string labels to binary matrix using MultiLabelBinarizer
                try:
                    labels_seq_binary = label_processor.transform_sequence(labels_seq_raw)
                except Exception as e:
                    print(f"Warning: Could not transform labels for sequence: {e}")
                    continue
                
                seq_loss = 0.0
                valid_steps = 0

                if len(seq) == 0 or len(labels_seq_binary) == 0:    
                    continue

                teacher_forced_seq, target_list = build_teacher_forced_sequences(
                    seq, labels_seq_binary, label_processor, include_initial_state=True
                )
                target_list = target_list.to(device)

                if use_mixed_precision:
                    with autocast():
                        logits = model(teacher_forced_seq, return_all_timesteps=True)
                        loss_per_position_tensor = criterion(logits, target_list)
                        seq_loss = loss_per_position_tensor.mean()
                else:
                    logits = model(teacher_forced_seq, return_all_timesteps=True)
                    loss_per_position_tensor = criterion(logits, target_list)
                    seq_loss = loss_per_position_tensor.mean()
                
                if not torch.isnan(seq_loss):
                    batch_loss += seq_loss
                    batch_timesteps += target_list.shape[1]  # T is now dim 1
                
            if batch_timesteps > 0:
                # Count valid sequences that contributed to batch_loss
                valid_batch_count = sum(1 for s in batch if len(s[0]) > 0)
                avg_batch_loss = batch_loss / max(valid_batch_count, 1)
                if use_mixed_precision:
                    scaler.scale(avg_batch_loss).backward()
                    if gradient_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    avg_batch_loss.backward()
                    if gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                    optimizer.step()
                    
                if scheduler:
                    scheduler.step()
                
                total_loss += avg_batch_loss.item()
                total_steps += 1
                total_timesteps += batch_timesteps
                
                current_lr = optimizer.param_groups[0]['lr']
                epoch_pbar.set_postfix({
                    'loss': f'{avg_batch_loss.item():.4f}',
                    'lr': f'{current_lr:.2e}',
                    'timesteps': batch_timesteps
                })

        avg_train_loss = total_loss / max(total_steps, 1)
        current_lr = optimizer.param_groups[0]['lr']


        
        # Validation evaluation
        val_loss = None
        val_metrics = None
        
        if val_data:
            model.eval()
            val_total_loss = 0.0
            val_total_steps = 0
            val_total_timesteps = 0
            
            # Prepare validation data for evaluation
            val_data_for_eval = []
            
            with torch.no_grad():
                for seq, labels_seq_raw in val_data:
                    try:
                        # Transform raw labels to binary format
                        labels_seq_binary = label_processor.transform_sequence(labels_seq_raw)
                        
                        if len(seq) == 0:
                            continue

                        label_seq_tensors = [
                            torch.tensor(step_binary_list, dtype=torch.float32, device=device)
                            for step_binary_list in labels_seq_binary
                        ]

                        val_data_for_eval.append((seq, label_seq_tensors))

                        # Build teacher-forced sequence for loss calculation
                        teacher_forced_seq, target_list = build_teacher_forced_sequences(
                            seq, labels_seq_binary, label_processor, include_initial_state=True
                        )
                        target_list = target_list.to(device)
                        # Calculate validation loss
                        if use_mixed_precision:
                            with autocast():
                                logits = model(teacher_forced_seq, return_all_timesteps=True)
                                loss_per_position_tensor = criterion(logits, target_list)
                                val_seq_loss = loss_per_position_tensor.mean()
                        else:
                            logits = model(teacher_forced_seq, return_all_timesteps=True)
                            loss_per_position_tensor = criterion(logits, target_list)
                            val_seq_loss = loss_per_position_tensor.mean()
                        if not torch.isnan(val_seq_loss):
                            # Make the loss comparable by multiplying by number of timesteps
                            val_total_loss += val_seq_loss.item() * target_list.shape[1]
                            val_total_timesteps += target_list.shape[1]
                    except Exception as e:
                        print(f"Warning: Could not process validation sequence for loss: {e}")
                        continue
            # Average validation loss over all timesteps
            val_loss = val_total_loss / max(val_total_timesteps, 1)
            val_metrics, _ = evaluate_with_teacher_forcing(
                model, 
                val_data_for_eval, 
                device,
                label_processor=label_processor,  # Pass the label_processor
                epoch=epoch,
                optional_tag=optional_tag,
                cache_manager=cache_manager
            )

            # Extract metrics for logging
            if val_metrics and 'SUMMARY' in val_metrics:
                micro_f1 = val_metrics['SUMMARY']['micro_avg']['f1']
                micro_precision = val_metrics['SUMMARY']['micro_avg']['precision']
                micro_recall = val_metrics['SUMMARY']['micro_avg']['recall']
                macro_f1 = val_metrics['SUMMARY']['macro_avg']['f1']
                weighted_f1 = val_metrics['SUMMARY']['weighted_avg']['f1']
                
                print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
                print(f"                 - Micro F1: {micro_f1:.3f}, "
                      f"Micro P: {micro_precision:.3f}, Micro R: {micro_recall:.3f}")
                print(f"                 - Macro F1: {macro_f1:.3f}, "
                      f"Weighted F1: {weighted_f1:.3f}")
            else:
                print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f}")

        # Log metrics
        logger.log_epoch(epoch, avg_train_loss, val_loss, val_metrics, current_lr)
        
        # Save intermediate results
        if directory_for_output and epoch % 5 == 0:  # Save every 5 epochs
            if optional_tag:
                logger.save_history(directory_for_output / f"training_history_mlb_{optional_tag}_epoch_{epoch}.json")
                torch.save(model.state_dict(), directory_for_output / f"model_mlb_{optional_tag}_epoch_{epoch}.pt")
            else:
                logger.save_history(directory_for_output / f"training_history_mlb_epoch_{epoch}.json")
                torch.save(model.state_dict(), directory_for_output / f"model_mlb_epoch_{epoch}.pt")

    # Final save and plot
    if directory_for_output:
        if optional_tag:
            logger.save_history(directory_for_output / f"final_training_history_mlb_{optional_tag}.json")
            torch.save(model.state_dict(), directory_for_output / f"final_model_mlb_{optional_tag}.pt")
            logger.plot_learning_curves(directory_for_output / f"learning_curves_mlb_{optional_tag}.png")
        else:
            logger.save_history(directory_for_output / "final_training_history_mlb.json")
            torch.save(model.state_dict(), directory_for_output / "final_model_mlb.pt")
            logger.plot_learning_curves(directory_for_output / "learning_curves_mlb.png")
    else:
        logger.plot_learning_curves()
    
    return model, logger