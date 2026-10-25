import pandas as pd
import torch
import torch.nn as nn
from copy import deepcopy

from config import get_model_config
from training.trainer import train_case_transformer_with_mlb
from training.evaluation import evaluate_with_teacher_forcing
from models.transformer_helpers import (
    discretize_bond_amount,
    bond_one_hot,
    encode_month_year_features,
)


class model_with_ablation(nn.Module):
    def __init__(self, base_model, ablation_config):
        super(model_with_ablation, self).__init__()
        self.base_model = base_model
        self.ablation_config = ablation_config
        self.text_emb = base_model.text_emb
        self.def_emb = base_model.def_emb
        self.case_emb = base_model.case_emb
        self.mag_emb = base_model.mag_emb

        self.num_labels = base_model.num_labels
        self.classifier = base_model.classifier

    def forward(self, sequence, return_all_timesteps=True):
        orig_encode = self.base_model.encode_proceeding
        cfg = self.ablation_config

        def ablated_encode_proceeding(pdict, proceeding_idx, hidden_features):
            device = next(self.base_model.parameters()).device
            model_cfg = get_model_config()

            # embeddings
            de = self.base_model.def_emb(
                pdict.get("race_id", 5),
                pdict.get("gender_id", 2),
                pdict.get("age", 25)
            ).to(device)

            ce = self.base_model.case_emb(pdict.get("charges", {})).to(device)

            pid = pdict.get("proceeding_id")
            te = self.base_model.text_emb(pid).to(device)

            court_type = pdict.get("type", "mag")
            court_type_feature = torch.tensor(
                [[1.0 if court_type == "cdc" else 0.0]],
                dtype=torch.float32,
                device=device
            )

            me = self.base_model.mag_emb(pdict.get("magistrate_section", "unknown")).to(device)

            # bonds
            if pid and pid.startswith('initial_'):
                orig_bin = model_cfg.BOND_BINS["not_set_or_missing"]
                sum_bin  = model_cfg.BOND_BINS["not_set_or_missing"]
            elif proceeding_idx == 0:
                orig_bin = model_cfg.BOND_BINS["not_set_or_missing"]
                sum_bin  = model_cfg.BOND_BINS["not_set_or_missing"]
            else:
                orig_bin = discretize_bond_amount(pdict.get("original_bond"))
                sum_bin  = discretize_bond_amount(pdict.get("sum_of_bonds"))

            orig_bond_feat = bond_one_hot(orig_bin, device)
            sum_bond_feat  = bond_one_hot(sum_bin,  device)

            # prior case count
            prior_case_count = torch.tensor(
                [[float(pdict.get("prior_case_count", 0))]],
                dtype=torch.float32, device=device
            )

            # prior charges
            prior_charges = pdict.get("all_prior_charges", {})
            prior_charge_emb = self.base_model.case_emb(prior_charges).to(device)

            # prior labels
            prior_labels_vec = pdict.get("true_prior_labels", None)
            if isinstance(prior_labels_vec, torch.Tensor):
                prior_labels_vec = prior_labels_vec.float().to(device)
            elif prior_labels_vec is not None:
                prior_labels_vec = torch.tensor(prior_labels_vec, dtype=torch.float32, device=device)
            else:
                prior_labels_vec = torch.zeros(self.base_model.num_labels, dtype=torch.float32, device=device)
            prior_label_feat = self.base_model.prior_label_emb(prior_labels_vec.unsqueeze(0))

            # temporal
            temporal_feats = encode_month_year_features(pdict.get("date"), device)

            # ablations
            if cfg.get('proceeding_text', False):
                te = torch.zeros_like(te)
            if cfg.get('previous_cases', False):
                prior_case_count = torch.zeros_like(prior_case_count)
                prior_charge_emb = torch.zeros_like(prior_charge_emb)
            if cfg.get('defendant_demographics', False):
                de = torch.zeros_like(de)
            if cfg.get('charges', False):
                ce = torch.zeros_like(ce)
            if cfg.get('magistrate', False):
                me = torch.zeros_like(me)
            if cfg.get('bond', False):
                orig_bond_feat = torch.zeros_like(orig_bond_feat)
                sum_bond_feat  = torch.zeros_like(sum_bond_feat)
            if cfg.get('temporal', False):
                temporal_feats = torch.zeros_like(temporal_feats)
            if cfg.get('prior_labels', False):
                prior_label_feat = torch.zeros_like(prior_label_feat)

            feature_block = torch.cat([court_type_feature, orig_bond_feat, sum_bond_feat, prior_case_count], dim=1)

            combined = torch.cat([
                de, ce, te, me, prior_label_feat, prior_charge_emb, feature_block, temporal_feats
            ], dim=1)

            return self.base_model.fuse(combined)

        self.base_model.encode_proceeding = ablated_encode_proceeding
        out = self.base_model.forward(sequence, return_all_timesteps=return_all_timesteps)
        self.base_model.encode_proceeding = orig_encode
        return out

def run_ablation_study_function(
    base_model,
    train_data,
    val_data,
    label_processor,
    device,
    ablation_configs,
    output_dir,
    epochs,
    batch_size,
    use_mixed_precision=False,
    use_learning_rate_scheduler=False,
    gradient_clip_norm=1.0,
    weighted_loss=False,
    fold_idx=None,
    cache_manager = None):

    results = {}
    for ablation_name, ablation_config in ablation_configs.items():
        ablated_features = [k for k, v in ablation_config.items() if v]
        
        print(f"Ablated features: {', '.join(ablated_features) if ablated_features else 'None'}")
        
        base_copy = deepcopy(base_model).to(device)
        ablation_model = model_with_ablation(base_copy, ablation_config).to(device)
        
        print(f"Training model...")
        print(f"Ablation name: {ablation_name}")
        if fold_idx is None:
            optional_tag = f"{ablation_name}"
        else:
            optional_tag = f"fold_{fold_idx + 1}_{ablation_name}"  

        trained_model, logger = train_case_transformer_with_mlb(
            ablation_model,
            train_data,
            label_processor,
            val_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            directory_for_output=output_dir / ablation_name if output_dir else None,
            prediction_level='proceeding',
            optional_tag=optional_tag,
            use_mixed_precision = use_mixed_precision,
            use_learning_rate_scheduler = use_learning_rate_scheduler,
            gradient_clip_norm = gradient_clip_norm,
            weighted_loss=weighted_loss,
            cache_manager = cache_manager
        )
        print("Training complete.")
        print("Evaluating model...")
        val_data_tensor = [(seq, label_processor.transform_sequence(labels)) 
                          for seq, labels in val_data]
      
        metrics, _= evaluate_with_teacher_forcing(
            trained_model,
            val_data_tensor,
            device,
            label_processor,
            epoch=None,
            optional_tag=optional_tag,
            cache_manager = cache_manager
        )
        
        results[ablation_name] = {
            'ablation_config': ablation_config,
            'ablated_features': ablated_features,
            'metrics': metrics,
            'model': trained_model
        }
        
        if 'SUMMARY' in metrics:
            summary = metrics['SUMMARY']
            print(f"\nResults:")
            print(f"   Micro F1: {summary['micro_avg']['f1']:.4f}")
            print(f"   Macro F1: {summary['macro_avg']['f1']:.4f}")
            print(f"   Weighted F1: {summary['weighted_avg']['f1']:.4f}")
    
    print("ABLATION STUDY COMPLETE")
    
    return results

def compare_ablation_studies_dataframe(results):
    comparison_data = []
    for config_name, result in results.items():
        metrics = result['metrics']
        if 'SUMMARY' in metrics:
            summary = metrics['SUMMARY']
            ablation_config = result['ablation_config']
            #give me the ablated features as a comma separated string
            ablated_features = [k for k, v in ablation_config.items() if v]
            ablation_str = ', '.join(ablated_features) if ablated_features else 'None'
            comparison_data.append({
                'Configuration': config_name,
                'Ablated Features': ablation_str,
                'Micro F1': summary['micro_avg']['f1'],
                'Macro F1': summary['macro_avg']['f1'],
                'Weighted F1': summary['weighted_avg']['f1'],
                'Micro Precision': summary['micro_avg']['precision'],
                'Micro Recall': summary['micro_avg']['recall'],
            })
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

