# Airway Case Study

This optional workflow runs an Airway case study using the bundled Airway dataset. It applies RAG clustering, performs one-versus-rest differential expression, computes RPS marker-uniqueness ranking, and generates an Airway UMAP plus top-marker panel for biological interpretation.

## Input

The workflow uses the bundled demo dataset:

```text
../../demo/data/Airway.h5
```

It also uses two small case-study inputs under `data/`:

- `Airway_umap_obs.csv`: fixed Airway UMAP coordinates used for the case-study layout.
- `airway_main_figure_clusters.csv`: curated list of Airway clusters to display and their plot labels.

The top marker genes shown in the figure are generated from the DE output.

## Run

From this folder:

```powershell
.\run_airway_all.ps1
```

Or run the steps individually:

```powershell
python 01_run_airway_clustering.py
python 02_run_airway_de.py
python 03_run_airway_rps.py
python 04_plot_airway_de_final_style.py
```

## Workflow

1. `01_run_airway_clustering.py` calls the `RAG.py` on `demo/data/Airway.h5` and writes `outputs/clustering/Airway_h5/Airway_clustered.h5ad`.
2. `02_run_airway_de.py` performs one-versus-rest Wilcoxon differential expression using the clustered h5ad.
3. `03_run_airway_rps.py` computes marker-set uniqueness and RPS ranking from `outputs/de/DE_sig_up_all_groups.csv`.
4. `04_plot_airway_de_final_style.py` generates the Airway UMAP and marker panel using the top upregulated genes from the DE output.

Key outputs include:

```text
outputs/de/DE_sig_up_all_groups.csv
outputs/de/DE_top_up_markers_by_group.csv
outputs/rps/RPS_cluster_ranking.csv
outputs/rps/candidate_marker_sets_long.csv
outputs/figures/airway_de_final_style/airway_main_matched_type_refs_top5_genes_DEstyle.png
outputs/figures/airway_de_final_style/airway_main_matched_type_refs_top5_genes_DEstyle.pdf
```

Outputs are written under:

```text
case_studies/airway/outputs/
```
