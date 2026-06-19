# Biomarker Discovery & Multi-Stage Bioinformatics Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/INXS-03/pipeline_biomarqueurs.py/blob/main/pipeline_biomarqueurs_py.ipynb)

An advanced, end-to-end transcriptomic data analysis pipeline designed to process Differential Expression Analysis (DEA) datasets, execute multi-database functional enrichment, integrate cross-cohort matrices for machine learning, rank features, and perform comprehensive gene annotation.

### Pipeline Architecture (The 6 Stages)

This repository implements a modular 6-stage architecture that seamlessly transitions from raw transcriptomic study data to fully annotated, prioritized biomarker panels:

### Étape 1: Functional Enrichment Analysis (ORA)
* **Filtering & Thresholding**: Extracts significantly altered features from multi-cohort datasets (e.g., `D1` and `D2`) using user-defined metrics ($P_{adj} < 0.05$ and $|\log_2(FC)| > 1.0$).
* **Over-Representation Analysis (ORA)**: Leverages `gseapy` to systematically scan 6 target databases: **KEGG**, **Reactome**, **WikiPathways**, and **Gene Ontology (BP, MF, CC)**.
* **Outputs**: Automates creation of volcano plots, up/down-regulated gene isolation lists, and comparative pathway enrichment bar plots.

### Étape 2: Preprocessing & Fusion for Machine Learning
* **Data Integration**: Integrates normalized counts matrices across distinct platforms and synchronization targets.
* **Feature Harmonization**: Filters global feature matrices down to match the validated DEGs from Stage 1.
* **Data Splitting & Scaling**: Handles Z-score normalization (`StandardScaler`) and outputs balanced, stratified train-test subsets (`train_X.csv`, `test_X.csv`, etc.).
* **Dimensionality Controls**: Renders exploratory Principal Component Analysis (PCA) scatter diagrams and top feature cluster heatmaps.

### Étape 3: Signature Extraction & Venn Cohort Intersection
* **Meta-Analysis Intersections**: Computes overlapping genomic profiles across multiple datasets ($D1 \cap D2$).
* **Robust Core Signature Extraction**: Minimizes cohort-specific batch artifacts by isolating strict consensus target spaces shared universally across your validation samples.

### Étape 4: Biomarker Prioritization & Ranking
* **Feature Importance Evaluation**: Leverages integrated ranking mechanisms (such as statistical scores or ensemble model weights) to order candidates.
* **Metric Sorting**: Groups markers by magnitude of expression change alongside statistical reliability metrics to prioritize top-tier candidates.

### Étape 5: Multi-Database Functional Querying
* **Custom Cross-Database Profiling**: Re-evaluates prioritized candidates against deep functional databases to discover niche biochemical mechanisms.
* **Target Pathways Clustering**: Links the prioritized subset specifically to targeted cellular pathways or disease phenotypes.

### Étape 6: Advanced Biomarker Annotation & Final Export
* **Gene Biotype & Chromosome Mapping**: Enriches target gene symbols with comprehensive genomic context (e.g., identifying protein-coding sequences, lncRNAs, and precise chromosomal map locations).
* **Automated Visual Reports**: Generates distribution statistics of biotypes, chromosomal distribution charts, and a publication-ready visual summary table of the top prioritized biomarkers.
* **Final Curated Deliverables**: Automatically exports the definitive biomarker configurations (`biomarqueurs_finaux_annotes.csv` and an isolated text list of symbols).

### Repository File Structure

The workspace isolates steps systematically under a structured `resultats/` tree directory:

```text
├── pipeline_biomarqueurs_py.ipynb     # Main execution Jupyter Notebook
├── D1.csv                             # DEA results for Cohort 1
├── D2.csv                             # DEA results for Cohort 2
├── normalized_counts_D1.csv           # Expression matrix for Cohort 1
├── normalized_counts_D2.csv           # Expression matrix for Cohort 2
│
└── resultats/                         # Automated Output Vault
    ├── etape1/                        # DEA Filtering & ORA Pathways
    │   ├── figures/                   # Volcano plots, Enrichments barplots
    │   └── tables/                    # Comprehensive csv pathway sheets
    ├── etape2/                        # ML Preprocessing matrices
    │   ├── figures/                   # PCA variance plots, Cluster Heatmaps
    │   └── tables/                    # train_X, test_X, train_y, test_y data splits
    ├── etape3/                        # Intersection analysis & Core panels
    ├── etape4/                        # Ranked feature sets & priority scores
    ├── etape5/                        # Target-specific database query profiles
    └── etape6/                        # Advanced Annotations (Final Stage)
        ├── figures/                   # biomarqueurs_biotypes.png, biomarqueurs_chromosomes.png
        └── tables/                    # biomarqueurs_finaux_annotes.csv, liste_symboles_biomarqueurs.txt
