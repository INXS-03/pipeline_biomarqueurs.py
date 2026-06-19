
#Biomarker Discovery and Machine Learning Pipeline
A modular, data-driven bioinformatics pipeline designed to detect transcriptomic biomarkers and prioritize diagnostic candidates for Endometriosis using Differential Expression Analysis (DEA), Functional Enrichment Analysis, and Machine Learning.

The pipeline handles multiple transcriptomic datasets (such as D1: GSE153739 and D2: GSE279835), performs high-throughput filtering, discovers functional targets across public pathway databases, and prepares standardized matrices optimized for training classifier models.

🧬 Pipeline Architecture & Features
🔹 Step 1 — Functional Enrichment Analysis
Automated Data Cleaning & Filtering: Parses differential expression inputs applying dynamic statistical cutoffs (P 
adj
​
 <0.05, ∣log 
2
​
 (FC)∣>1.0).

Gene List Exportation: Automatically isolates and extracts segmented up-regulated, down-regulated, and global Differentially Expressed Gene (DEG) lists.

Multi-Database Over-Representation Analysis (ORA): Seamlessly interrogates 6 benchmark biological databases via gseapy:

Gene Ontology (Biological Process, Molecular Function, Cellular Component)

KEGG Pathways

Reactome Pathway Knowledgebase

WikiPathways

Automated Visualizations: Generates customized volcano plots identifying elite targets along with comparative publication-ready barplots highlighting −log 
10
​
 (P 
adj
​
 ) pathways.

Cross-Dataset Meta-Analysis: Computes intersections (D1∩D2) to isolate high-confidence biological signatures shared across cohorts.

🔹 Step 2 — ML Data Preprocessing & Fusion
Cross-Cohort Matrix Integration: Fuses normalized count matrices from multiple platforms into a unified analytical matrix.

Feature Filtering: Syncs common features matching the validated DEGs discovered during Step 1.

Advanced Normalization: Applies Z-score standardization using robust StandardScaler mechanisms.

Stratified Splitting: Generates partitioned Train/Test subsets maintaining categorical disease/control class distribution balances.

Dimensionality Assessment: Evaluates feature distributions through automated Principal Component Analysis (PCA) and high-density clustering Heatmaps.

📂 Repository File Structure
The project dynamically generates standard-compliant outputs organized under a clean resultats/ tree structure:

Plaintext
├── pipeline_biomarqueurs_py.ipynb    # Main execution Jupyter Notebook
├── D1.csv                            # DEA metrics file for Dataset 1 (GSE153739)
├── D2.csv                            # DEA metrics file for Dataset 2 (GSE279835)
├── normalized_counts_D1.csv          # Matrix for D1 samples
├── normalized_counts_D2.csv          # Matrix for D2 samples
│
└── resultats/
    ├── etape1/                       # Step 1 Analytics
    │   ├── figures/                  # D1_volcano.png, D2_volcano.png, barplots
    │   ├── tables/                   # Filtered enrichments (.csv)
    │   └── *.txt                     # SEGMENTED Gene Lists (All, UP, DOWN)
    │
    └── etape2/                       # Step 2 ML Matrix Analytics
        ├── figures/                  # pca_plot.png, heatmap_top50.png
        ├── tables/                   # train_X.csv, test_X.csv, train_y.csv, test_y.csv
        └── data_ml.pkl               # Pickled data object ready for modeling
🛠️ Installation & Prerequisites
This pipeline is built on Python 3.10+ and relies heavily on scientific computing and bioinformatics packages.

1. Local Environment Setup
Clone this repository and install all required dependencies:

Bash
git clone https://github.com/INXS-03/pipeline_biomarqueurs.py.git
cd pipeline_biomarqueurs.py
pip install pandas numpy matplotlib seaborn scikit-learn scipy gseapy openpyxl
2. Expected Data Format
Your raw differential expression inputs (D1.csv, D2.csv) must be semi-colon (;) separated tables featuring the following header structure:

Extrait de code
GeneID;Base mean;log2(FC);StdErr;Wald-Stats;P-value;P-adj;Chromosome;Start;End;Strand;Feature;Gene symbol
🚀 Usage
Place your target file arrays (D1.csv, D2.csv, normalized_counts_D1.csv, normalized_counts_D2.csv) straight into the root repository directory.

Open the notebook in a terminal or inside standard Jupyter environments:

Bash
jupyter notebook pipeline_biomarqueurs_py.ipynb
Adjust path configurations inside the configuration cells if your file titles vary from defaults:

Python
D1_PATH = "D1.csv"
D2_PATH = "D2.csv"
SEP     = ";"
Click Run All cells. Check the runtime logs inside standard streams to monitor progress and find visual metrics stored automatically in your resultats/ directory.

📊 Sample Visual Analytics Expected
Volcano Plots: Highlighting up-regulated target candidates in red, down-regulated elements in blue, and low-priority ones in grey.

Enrichment Histograms: Horizontal bar diagrams rank-ordering pathways based on strict statistical confidence measures.

PCA Components Mapping: Dimensional clustering analysis checking for platform batch variances or biological sample separations.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
