# Parameters
parameters_dict = {
    'model_validation_strategy': 'k_fold_cross_validation',  # Options: 'k_fold_cross_validation', 'train_val_split'
    'number_of_cases_in_sample': 1000,
    'number_of_epochs_for_training': 20,
    'batch_size_for_training': 16,
    'top_n_performing_labels_to_display': 10,
    'numbers_of_cases_to_analyze_for_interpretability': 200,
    'minimum_confidence_for_wrong_prediction_inclusion': 0.5,
    'number_of_highest_probability_predictions_per_step': 5,
    'maximum_number_of_confident_mistakes_to_return': 25,
    'maximum_validation_sequences_to_process_for_analysis_of_incorrect': 100,
    'num_best_and_worst_cases_to_display': 5,
    'k_fold_num_folds': 5,        # Number of folds for cross-validation
    'k_fold_shuffle': True,       # Shuffle before splitting
    'new_dataset_creation': True,
    'sequence_data_mode': 'export',  # Options: 'current', 'export', 'import'    
    'random_seed': 42,
    'weighted_loss_function': True,
    'sampling_strategy': 'balanced', #options are 'uniform' and 'oversample_long_sequences' and 'balanced' and None
    'skip_duplicate_labels': True,
    'skip_duplicate_labels_on_import': True,
    'regex_labels_mode': 'start_from_scratch', #options are 'load_in' or 'start_from_scratch'
    'folder_name_directory': 'balanced_skipped_duplicates'
}