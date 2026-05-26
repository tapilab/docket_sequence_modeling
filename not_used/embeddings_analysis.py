class CaseEmbeddingAnalysis:
    """
    Class for analyzing case embeddings from a trained case transformer model.
    """
    def __init__(
        self,
        model: Any,
        label_processor: Any,
        device: torch.device,
        model_config: Optional[Any] = None,
        plots_dir: Path = Path("plots"),
        analysis_dir: Path = Path("analysis"),
        target_var: float = 0.90,
        k_min: int = 2,
        k_max: int = 10,
    ) -> None:
        self.model = model
        self.label_processor = label_processor
        self.device = device
        self.model_config = model_config
        self.plots_dir = Path(plots_dir)
        self.analysis_dir = Path(analysis_dir)
        self.target_var = float(target_var)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None

    def get_encoder_output(self, sequence, labels_binary):
        if not sequence:
            return None
        # Limit to max sequence length from the end
        start_idx = max(0, len(sequence) - self.model.max_len)
        seq_trimmed = sequence[-self.model.max_len:]
        labels_binary_trimmed = labels_binary[-self.model.max_len:] # Trim labels accordingly
        hidden_features = self.model.fuse[0].in_features # Get input dim for fuse layer
        reps = []
        for i, p in enumerate(seq_trimmed):
            current_proc_dict = dict(p)  # Make a copy
            # --- Add true_prior_labels based on teacher forcing context ---
            # Calculate the index in the *original* untrimmed labels_binary
            original_label_idx_for_prior = start_idx + i - 1
            if 0 <= original_label_idx_for_prior < len(labels_binary):
                prior_labels_list = [float(lbl) for lbl in labels_binary[original_label_idx_for_prior]]
                current_proc_dict['true_prior_labels'] = prior_labels_list
            else:
                # Handle beginning of sequence or initial state if applicable
                current_proc_dict['true_prior_labels'] = [0.0] * len(self.label_processor.label_names)
            #Special handling for explicitly marked initial state (if present)
            #This might occur if your sequence builder includes a dummy first step
            if current_proc_dict.get('text') == 'INITIAL_STATE':
                current_proc_dict['proceeding_id'] = f'initial_{current_proc_dict.get("defendant_id", "unknown")}'
                current_proc_dict['original_bond'] = None
                current_proc_dict['sum_of_bonds'] = None
                # Ensure prior labels are zero for the initial state representation
                current_proc_dict['true_prior_labels'] = [0.0] * len(self.label_processor.label_names)
            # Encode the proceeding dictionary using the model's method
            rep = self.model.encode_proceeding(current_proc_dict, start_idx + i, hidden_features)
            reps.append(rep)
        if not reps:
            return None # Return None if sequence becomes empty 
        # Stack representations
            x = torch.stack(reps, dim=1).to(self.device) # Shape: (1, seq_len, hidden_dim_input_to_fuse)
        # Positional encoding
        x_pos = x + self.model.pos[:, :x.size(1), :].to(self.device) # Shape: (1, seq_len, hidden_dim)
        # Create attention mask (no masking for standard processing)
        mask = torch.zeros(x_pos.size(0), x_pos.size(1), dtype=torch.bool, device=self.device)
        # Pass through encoder
        encoder_output = self.model.encoder(x_pos, src_key_padding_mask=mask) # Shape: (1, seq_len, hidden_dim)
        return encoder_output # Return the full encoder output sequence
    def extract_case_embeddings(
        self,
        sequences_and_labels,
        id_prefix, 
        max_cases,
        show_tqdm,):
        case_embedding_data = []
        n_total = len(sequences_and_labels)
        iterator = enumerate(sequences_and_labels)
        if show_tqdm:
            iterator = tqdm(iterator, total=n_total, desc="Extracting Encoder Outputs")

        with torch.no_grad():
            for idx, (sequence, labels_raw) in iterator:
                if max_cases is not None and idx >= max_cases:
                    break
                if not sequence:
                    continue

                # Get case identifier
                case_id = f"{id_prefix}_{idx}"
                if sequence[0].get('docket_id'):
                    case_id = f"{sequence[0].get('type', 'unk')}_{sequence[0].get('docket_id')}_{sequence[0].get('defendant_id', 'unk')}"

                # Convert raw labels to binary format needed for teacher forcing context
                try:
                    labels_binary = label_processor.transform_sequence(labels_raw)
                except ValueError as e:
                    print(f"Warning: Skipping case {case_id} due to label processing error: {e}")
                    continue

                # Get encoder output for the sequence
                encoder_output = self.get_encoder_output(sequence, labels_binary)
                if encoder_output is None:
                    print(f"Warning: Skipping case {case_id} due to empty sequence after processing.")
                    continue

                # --- Aggregate ---
                # Use mean pooling over the sequence length dimension
                aggregated_embedding = torch.mean(encoder_output.squeeze(0), dim=0).cpu().numpy()

                # Label summary (respect optional exclusions)
                excluded = getattr(self.model_config, "EXCLUDED_LABELS", set())
                all_true_labels_in_case = set()
                for labels in labels_raw:
                    valid_labels = [l for l in labels if l not in excluded]
                    all_true_labels_in_case.update(valid_labels)
                label_summary = ", ".join(sorted(all_true_labels_in_case))

                case_embedding_data.append(
                    {
                        "case_id": case_id,
                        "embedding_vector": aggregated_embedding,
                        "num_proceedings": len(sequence),
                        "label_summary": label_summary,
                    }
                )

                # Memory hygiene
                del encoder_output, aggregated_embedding, labels_binary
                if idx % 50 == 0:
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

        if not case_embedding_data:
            raise ValueError("No case embeddings could be extracted. Check data and model.")

        embedding_df = pd.DataFrame(case_embedding_data)
        embedding_df["embedding_vector"] = embedding_df["embedding_vector"].apply(
            lambda x: np.array(x) if isinstance(x, list) else x
        )
        embeddings_array = np.stack(embedding_df["embedding_vector"].values)

        print(f"   Successfully extracted embeddings for {len(embedding_df)} cases.")
        print(f"   Embedding array shape: {embeddings_array.shape}")

        return embedding_df, embeddings_array

# -------------------------------------------------------------------------
    # 15c. PCA + clustering helpers
    # -------------------------------------------------------------------------
    def run_pca(self, embeddings_array: np.ndarray) -> Tuple[StandardScaler, PCA, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
        """
        Scales embeddings, runs PCA, picks m to reach target variance.
        Returns: scaler, pca, Z (all PCs), Z_m (first m), Z_2 (first 2), m, cumvar, scaled
        """
        print("\n" + "=" * 80)
        print("DIMENSIONALITY REDUCTION OF CASE EMBEDDINGS")
        print("=" * 80)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(embeddings_array)

        pca = PCA()
        Z = pca.fit_transform(scaled)
        cumvar = np.cumsum(pca.explained_variance_ratio_)

        m = int(np.searchsorted(cumvar, self.target_var) + 1)
        m = max(2, min(m, Z.shape[1]))  # at least 2
        Z_m = Z[:, :m]
        Z_2 = Z[:, :2]

        print(f"PCA chose m={m} (captures {cumvar[m-1]*100:.2f}% variance).")
        first_two_var = cumvar[min(1, len(cumvar) - 1)] * 100.0
        print(f"First two PCs alone capture {first_two_var:.2f}%.")

        self.scaler = scaler
        self.pca = pca
        return scaler, pca, Z, Z_m, Z_2, m, cumvar, scaled


    @staticmethod
    def pick_k_by_silhouette(Zm: np.ndarray, k_min: int = 2, k_max: int = 10) -> Tuple[int, np.ndarray]:
        """
        Uses silhouette score to choose k for KMeans on Zm.
        Returns: (best_k, labels_for_best_k)
        """
        k_max = min(k_max, len(Zm) - 1)  # need k < n
        if k_max < k_min:
            print(f"   Warning: Not enough samples ({len(Zm)}) to test k up to {k_min}. Setting k=1.")
            return 1, np.zeros(len(Zm), dtype=int)

        best = None
        print(f"   Finding best k using Silhouette Score (k={k_min} to {k_max})...")
        for k in range(k_min, k_max + 1):
            labels = KMeans(n_clusters=k, n_init=10).fit_predict(Zm)
            if len(np.unique(labels)) < 2:
                print(f"     k={k}: Only 1 cluster found, skipping silhouette calculation.")
                continue
            score = silhouette_score(Zm, labels)
            print(f"     k={k}: Silhouette Score = {score:.4f}")
            if (best is None) or (score > best[0]):
                best = (score, k, labels)

        if best is None:
            print("   Warning: Could not find suitable k > 1. Defaulting to k=1.")
            return 1, np.zeros(len(Zm), dtype=int)

        score_star, k_star, labels_star = best
        print(f"   Best k found: {k_star} (Silhouette Score: {score_star:.4f})")
        return k_star, labels_star

    def attach_pca_and_clusters(
        self,
        embedding_df: pd.DataFrame,
        Z_2: np.ndarray,
        k_min: Optional[int] = None,
        k_max: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, int]:
        """
        Adds pca_x, pca_y and cluster_id columns to embedding_df.
        Chooses best k by silhouette on Z_2.
        """
        if self.pca is None:
            raise RuntimeError("PCA not run. Call run_pca() first.")

        k_min = self.k_min if k_min is None else k_min
        k_max = self.k_max if k_max is None else k_max

        df = embedding_df.copy()
        df["pca_x"] = Z_2[:, 0]
        df["pca_y"] = Z_2[:, 1]

        if len(Z_2) < 2:
            print("   Skipping clustering: Not enough samples.")
            best_k = 1
            cluster_labels = np.zeros(len(Z_2), dtype=int)
        else:
            best_k, cluster_labels = self.pick_k_by_silhouette(Z_2, k_min=k_min, k_max=k_max)

        df["cluster_id"] = cluster_labels
        print(f"   Assigned cases to {best_k} clusters based on Silhouette Score.")
        if best_k > 1:
            print(f"   Cluster distribution:\n{df['cluster_id'].value_counts().sort_index()}")

            # Optional: quick label distribution peek
            print("\n   Analyzing label distributions per cluster:")
            df["label_summary"] = df["label_summary"].astype(str)
            try:
                cluster_labels_analysis = df.groupby("cluster_id")["label_summary"].apply(
                    lambda summaries: Counter(
                        label.strip()
                        for summary in summaries if isinstance(summary, str)
                        for label in summary.split(",")
                        if label.strip()
                    )
                )
                if not cluster_labels_analysis.empty:
                    for cluster_id, label_counts in cluster_labels_analysis.items():
                        print(f"   Cluster {cluster_id} Top 3 Labels: {label_counts.most_common(3)}")
                else:
                    print("   No labels found to analyze distributions per cluster.")
            except Exception as e:
                print(f"   Error analyzing cluster labels: {e}")

        return df, best_k

    def plot_pca_scatter(self, embedding_df: pd.DataFrame, out_path: Path) -> None:
        """
        Simple PCA scatter (top 2 PCs). Saves PNG.
        """
        if self.pca is None:
            raise RuntimeError("PCA not run. Call run_pca() first.")

        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=embedding_df, x="pca_x", y="pca_y", alpha=0.6)
        var1 = self.pca.explained_variance_ratio_[0] * 100.0 if len(self.pca.explained_variance_ratio_) > 0 else 0.0
        var2 = self.pca.explained_variance_ratio_[1] * 100.0 if len(self.pca.explained_variance_ratio_) > 1 else 0.0
        plt.title("Case Embeddings PCA (Top 2 Components) after Scaling")
        plt.xlabel(f"PCA Component 1 ({var1:.1f}% Var)")
        plt.ylabel(f"PCA Component 2 ({var2:.1f}% Var)")
        plt.grid(True, alpha=0.3)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"   Saved Scaled PCA (Top 2) plot to {out_path}")

    def plot_cluster_scatter(self, embedding_df: pd.DataFrame, best_k: int, out_path: Path) -> None:
        """
        Clustered scatter if best_k > 1. Saves PNG.
        """
        if self.pca is None:
            raise RuntimeError("PCA not run. Call run_pca() first.")

        if best_k <= 1:
            print("   Skipped cluster plot: Only one cluster was determined.")
            return

        plt.figure(figsize=(12, 10))
        palette = sns.color_palette("viridis", n_colors=best_k)
        sns.scatterplot(
            data=embedding_df,
            x="pca_x",
            y="pca_y",
            hue="cluster_id",
            palette=palette,
            alpha=0.7,
            legend="full",
        )
        var1 = self.pca.explained_variance_ratio_[0] * 100.0 if len(self.pca.explained_variance_ratio_) > 0 else 0.0
        var2 = self.pca.explained_variance_ratio_[1] * 100.0 if len(self.pca.explained_variance_ratio_) > 1 else 0.0
        plt.title(f"Case Embeddings K-Means Clusters (k={best_k} via Silhouette) on Scaled PCA (Top 2)")
        plt.xlabel(f"PCA Component 1 ({var1:.1f}% Var)")
        plt.ylabel(f"PCA Component 2 ({var2:.1f}% Var)")
        plt.legend(title="Cluster ID", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"   Saved Scaled Cluster (k={best_k}) plot to {out_path}")

    # -------------------------------------------------------------------------
    # Saving helpers
    # -------------------------------------------------------------------------
    def save_arrays_and_tables(
        self,
        embedding_df: pd.DataFrame,
        scaled_embeddings: np.ndarray,
        Z_m: np.ndarray,
        m: int,
        best_k: int,
        csv_basename: str = "case_embeddings_scaled_pca_cluster",
        scaled_npy_name: str = "case_embeddings_scaled.npy",
        pca_m_npy_basename: str = "case_embeddings_scaled_pca",
    ) -> Dict[str, Path]:
        """
        Saves:
          - CSV (without 'embedding_vector')
          - scaled embeddings .npy
          - PCA(m) .npy
        Returns paths.
        """
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        df_to_save = embedding_df.drop(columns=["embedding_vector"])
        csv_path = self.analysis_dir / f"{csv_basename}_k{best_k}.csv"
        df_to_save.to_csv(csv_path, index=False)
        print(f"   Saved DataFrame (metadata, PCA, clusters) to {csv_path}")

        scaled_npy_path = self.analysis_dir / scaled_npy_name
        np.save(scaled_npy_path, scaled_embeddings)
        print(f"   Saved SCALED NumPy embeddings array to {scaled_npy_path}")

        pca_m_npy_path = self.analysis_dir / f"{pca_m_npy_basename}_{m}comp.npy"
        np.save(pca_m_npy_path, Z_m)
        print(f"   Saved PCA-reduced ({m} components) NumPy array to {pca_m_npy_path}")

        return {
            "csv": csv_path,
            "scaled_npy": scaled_npy_path,
            "pca_m_npy": pca_m_npy_path,
        }

    # -------------------------------------------------------------------------
    # Orchestrator
    # -------------------------------------------------------------------------
    def run(
        self,
        sequences_and_labels: Iterable[Tuple[Sequence[Dict[str, Any]], Sequence[Sequence[str]]]],
        id_prefix: str = "val_case",
        max_cases: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline:
          1) Extract mean-pooled encoder embeddings per case
          2) Scale + PCA; pick m to hit target variance
          3) Choose k via silhouette on 2D PCA; attach clusters
          4) Save plots + artifacts
          5) Return artifacts in a dict
        """
        # 1) Embeddings
        embedding_df, embeddings_array = self.extract_case_embeddings(
            sequences_and_labels=sequences_and_labels,
            id_prefix=id_prefix,
            max_cases=max_cases,
        )

        # 2) PCA
        scaler, pca, Z, Z_m, Z_2, m, cumvar, scaled_embeddings = self.run_pca(embeddings_array)

        # 3) Clusters
        embedding_df, best_k = self.attach_pca_and_clusters(embedding_df=embedding_df, Z_2=Z_2)

        # 4) Plots
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        pca_plot_path = self.plots_dir / "case_embeddings_scaled_pca_top2.png"
        self.plot_pca_scatter(embedding_df, pca_plot_path)

        cluster_plot_path = self.plots_dir / f"case_embeddings_scaled_clusters_k{best_k}_silhouette.png"
        self.plot_cluster_scatter(embedding_df, best_k, cluster_plot_path)

        # 5) Save arrays/tables
        saved_paths = self.save_arrays_and_tables(
            embedding_df=embedding_df,
            scaled_embeddings=scaled_embeddings,
            Z_m=Z_m,
            m=m,
            best_k=best_k,
        )

        # Cleanup GPU mem
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        print("\n✅ Analysis complete!")

        return {
            "embedding_df": embedding_df,
            "embeddings_array": embeddings_array,
            "scaled_embeddings": scaled_embeddings,
            "Z": Z,
            "Z_m": Z_m,
            "Z_2": Z_2,
            "pca": pca,
            "scaler": scaler,
            "best_k": best_k,
            "plots": {
                "pca_top2": pca_plot_path,
                "clusters": cluster_plot_path,
            },
            "saved": saved_paths,
        }

    