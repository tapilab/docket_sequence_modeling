"""
# SECTION Method to generate and cache regex labels
def generate_and_cache_regex_labels(data, dir_for_cache, load_in_or_start_from_scratch, cache_manager = None):
    if dir_for_cache is None:
        dir_for_cache = cache_manager.get_cache_dir("labels")   
    if not dir_for_cache.exists():
        dir_for_cache.mkdir(parents=True, exist_ok=True)
    labels_cache_file = dir_for_cache / "labels_cache.pkl"

    proceeding_labels = {}
    # Load existing cache
    if load_in_or_start_from_scratch == "start_from_scratch":
        proceeding_labels = {}
    elif load_in_or_start_from_scratch == "load_in":
      if labels_cache_file.exists():
          with open(labels_cache_file, "rb") as f:
              print(f"Loading labels cache from {labels_cache_file}")
              proceeding_labels = pickle.load(f)

    # Get all current proceeding IDs
    current_ids = set()

    # Process mag data
    for _, row in tqdm(data['mag'].iterrows(), total=len(data['mag'])):
        proceeding_id = row['proceeding_id']
        current_ids.add(proceeding_id)

        # Only process if not in cache or if you want to always update
        if proceeding_id not in proceeding_labels:
            text = row.get('text', '')
            labels = true_labels_generator(text)
            proceeding_labels[proceeding_id] = labels

    # Process cdc data
    for _, row in tqdm(data['cdc'].iterrows(), total=len(data['cdc'])):
        proceeding_id = row['proceeding_id']
        current_ids.add(proceeding_id)

        if proceeding_id not in proceeding_labels:
            text = row.get('text', '')
            labels = true_labels_generator(text)
            proceeding_labels[proceeding_id] = labels

    # Remove old entries that are no longer in the data
    proceeding_labels = {k: v for k, v in proceeding_labels.items() if k in current_ids}

    # Save updated cache
    with open(labels_cache_file, "wb") as f:
        pickle.dump(proceeding_labels, f)

    return proceeding_labels
"""
