pip install pandas numpy matplotlib seaborn scikit-learn scipy gseapy openpyxl

"""
╔══════════════════════════════════════════════════════════════════════╗
║  ÉTAPE 1 — ANALYSE FONCTIONNELLE                                     ║
║  GO Enrichment + KEGG + Reactome + WikiPathways                      ║
║  Datasets : D1 (GSE153739) & D2 (GSE279835)                         ║
╚══════════════════════════════════════════════════════════════════════╝

PRÉREQUIS :
    pip install gseapy pandas matplotlib seaborn numpy

STRUCTURE DES CSV ATTENDUE (séparateur = ;) :
    GeneID ; Base mean ; log2(FC) ; StdErr ; Wald-Stats ;
    P-value ; P-adj ; Chromosome ; Start ; End ; Strand ; Feature ; Gene symbol

OUTPUTS GÉNÉRÉS :
    resultats/etape1/
    ├── D1_genes_all.txt              ← liste des gènes D1 (tous)
    ├── D1_genes_UP.txt               ← gènes upregulés D1
    ├── D1_genes_DOWN.txt             ← gènes downregulés D1
    ├── D2_genes_filtered.txt         ← gènes D2 (p-adj<0.05, |lfc|>1)
    ├── D2_genes_UP.txt
    ├── D2_genes_DOWN.txt
    ├── figures/
    │   ├── D1_volcano.png
    │   ├── D2_volcano.png
    │   ├── D1_GO_BP_barplot.png
    │   ├── D1_KEGG_barplot.png  ...
    └── tables/
        ├── D1_GO_BP.csv
        ├── D1_KEGG.csv  ...
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import gseapy as gp

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — MODIFIE CES CHEMINS SI NÉCESSAIRE
# ─────────────────────────────────────────────────────────────────────────────
D1_PATH = "D1.csv"          # chemin vers ton fichier D1
D2_PATH = "D2.csv"          # chemin vers ton fichier D2
SEP     = ";"               # séparateur des CSV

# Seuils de filtrage pour D2 (D1 est déjà filtré)
D2_PADJ_THRESH  = 0.05
D2_LFC_THRESH   = 1.0

# Seuil de significativité pour l'enrichissement
ENRICH_PADJ = 0.05
TOP_N       = 20            # nombre de termes à afficher dans les plots

OUT = "resultats/etape1"
os.makedirs(f"{OUT}/figures", exist_ok=True)
os.makedirs(f"{OUT}/tables",  exist_ok=True)

print("=" * 65)
print("  ÉTAPE 1 : ANALYSE FONCTIONNELLE")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 1.1  CHARGEMENT & FILTRAGE
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 1.1  Chargement des données ──────────────────────────────────")

d1 = pd.read_csv(D1_PATH, sep=SEP)
d2 = pd.read_csv(D2_PATH, sep=SEP)

# Nettoyage : retirer espaces dans les noms de colonnes
d1.columns = d1.columns.str.strip()
d2.columns = d2.columns.str.strip()
d1["Gene symbol"] = d1["Gene symbol"].str.strip()
d2["Gene symbol"] = d2["Gene symbol"].str.strip()

# D1 : déjà filtré sur |log2FC| > 1 → on garde tous les 313 gènes
d1_filtered = d1.copy()
print(f"  D1 : {len(d1_filtered)} gènes (déjà filtrés, |log2FC| > 1)")
print(f"       ↑ Upreg  : {(d1_filtered['log2(FC)'] > 0).sum()}")
print(f"       ↓ Downreg: {(d1_filtered['log2(FC)'] < 0).sum()}")

# D2 : filtrage P-adj < 0.05 ET |log2FC| > 1
d2_filtered = d2[
    (d2["P-adj"]      < D2_PADJ_THRESH) &
    (d2["log2(FC)"].abs() > D2_LFC_THRESH)
].copy()
print(f"\n  D2 : {len(d2_filtered)} gènes après filtrage")
print(f"       (P-adj < {D2_PADJ_THRESH}, |log2FC| > {D2_LFC_THRESH})")
print(f"       ↑ Upreg  : {(d2_filtered['log2(FC)'] > 0).sum()}")
print(f"       ↓ Downreg: {(d2_filtered['log2(FC)'] < 0).sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.2  EXTRACTION DES LISTES DE GÈNES
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 1.2  Extraction des listes de gènes ──────────────────────────")

def extract_genes(df, name, out_dir):
    """Extrait et sauvegarde les listes all / UP / DOWN."""
    genes_all  = df["Gene symbol"].dropna().unique().tolist()
    genes_up   = df[df["log2(FC)"] >  0]["Gene symbol"].dropna().unique().tolist()
    genes_down = df[df["log2(FC)"] <  0]["Gene symbol"].dropna().unique().tolist()

    for label, lst in [("all", genes_all), ("UP", genes_up), ("DOWN", genes_down)]:
        path = f"{out_dir}/{name}_genes_{label}.txt"
        with open(path, "w") as f:
            f.write("\n".join(lst))
        print(f"  {name} {label:5s}: {len(lst)} gènes → {path}")

    return genes_all, genes_up, genes_down

d1_all, d1_up, d1_down = extract_genes(d1_filtered, "D1", OUT)
d2_all, d2_up, d2_down = extract_genes(d2_filtered, "D2", OUT)


# ─────────────────────────────────────────────────────────────────────────────
# 1.3  VOLCANO PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 1.3  Volcano plots ───────────────────────────────────────────")

def volcano_plot(df, title, out_path,
                 lfc_thresh=1.0, padj_thresh=0.05, top_n=15):
    """
    Volcano plot coloré :
      - Rouge  = significatif UP   (padj < seuil et lfc > +seuil)
      - Bleu   = significatif DOWN (padj < seuil et lfc < -seuil)
      - Gris   = non significatif
    Annote les top_n gènes les plus significatifs.
    """
    df = df.copy()
    df["-log10(P-adj)"] = -np.log10(df["P-adj"].clip(lower=1e-300))

    # Catégories
    conditions = [
        (df["P-adj"] < padj_thresh) & (df["log2(FC)"] >  lfc_thresh),
        (df["P-adj"] < padj_thresh) & (df["log2(FC)"] < -lfc_thresh),
    ]
    colors_map = ["#E8312A", "#2E75B6"]
    labels_map = [
        f"UP   (n={conditions[0].sum()})",
        f"DOWN (n={conditions[1].sum()})",
    ]
    df["color"] = "#BBBBBB"
    for cond, col in zip(conditions, colors_map):
        df.loc[cond, "color"] = col

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        df["log2(FC)"], df["-log10(P-adj)"],
        c=df["color"], alpha=0.6, s=18, linewidths=0
    )

    # Lignes de seuil
    ax.axvline( lfc_thresh,  color="black", linestyle="--", lw=0.8, alpha=0.5)
    ax.axvline(-lfc_thresh,  color="black", linestyle="--", lw=0.8, alpha=0.5)
    ax.axhline(-np.log10(padj_thresh), color="black", linestyle="--", lw=0.8, alpha=0.5)

    # Top gènes annotés
    sig = df[df["color"] != "#BBBBBB"].nsmallest(top_n, "P-adj")
    for _, row in sig.iterrows():
        ax.annotate(
            row["Gene symbol"],
            xy=(row["log2(FC)"], row["-log10(P-adj)"]),
            fontsize=6.5, color="black",
            xytext=(4, 2), textcoords="offset points",
        )

    # Légende
    patches = [
        mpatches.Patch(color=colors_map[0], label=labels_map[0]),
        mpatches.Patch(color=colors_map[1], label=labels_map[1]),
        mpatches.Patch(color="#BBBBBB",     label="Non significatif"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper left")
    ax.set_xlabel("log2(Fold Change)", fontsize=11)
    ax.set_ylabel("-log10(P-adj)",     fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Volcano sauvegardé → {out_path}")

# D1 : P-adj souvent > 0.05 → on utilise P-value brute pour le volcano
d1_vol = d1_filtered.copy()
if (d1_vol["P-adj"] < 0.05).sum() < 5:
    print("  [INFO] D1 : peu de gènes avec P-adj < 0.05 — utilisation de P-value brute pour le volcano")
    d1_vol["P-adj"] = d1_vol["P-value"]   # remplacement temporaire pour le plot

volcano_plot(
    d1_vol,
    title    = "Volcano Plot — D1 (GSE153739)\nEndométriose vs Contrôle",
    out_path = f"{OUT}/figures/D1_volcano.png",
    lfc_thresh  = 1.0,
    padj_thresh = 0.05,
)

volcano_plot(
    d2_filtered,
    title    = "Volcano Plot — D2 (GSE279835)\nEndométriose vs Contrôle",
    out_path = f"{OUT}/figures/D2_volcano.png",
    lfc_thresh  = 1.0,
    padj_thresh = 0.05,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.4  ANALYSE D'ENRICHISSEMENT (GO + KEGG + Reactome + WikiPathways)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 1.4  Analyses d'enrichissement ───────────────────────────────")

DATABASES = {
    "GO_Biological_Process_2023" : "GO_BP",
    "GO_Molecular_Function_2023" : "GO_MF",
    "GO_Cellular_Component_2023" : "GO_CC",
    "KEGG_2021_Human"            : "KEGG",
    "Reactome_2022"              : "Reactome",
    "WikiPathway_2023_Human"     : "WikiPathways",
}

def run_enrichr(gene_list, label, top_n=TOP_N):
    """
    Lance gseapy.enrichr sur toutes les bases de données.
    Sauvegarde CSV + barplot pour chaque base.
    Retourne un dict {db_short: DataFrame_significatif}.
    """
    if not gene_list:
        print(f"  [{label}] Liste vide — ignoré.")
        return {}

    print(f"\n  ▶  {label} ({len(gene_list)} gènes)")
    all_results = {}

    for db_full, db_short in DATABASES.items():
        try:
            enr = gp.enrichr(
                gene_list = gene_list,
                gene_sets = db_full,
                organism  = "human",
                outdir    = None,
                verbose   = False,
            )
            res = enr.results.copy()

            if res.empty:
                print(f"    {db_short:15s}: aucun résultat.")
                continue

            # Filtrage significatif
            sig = res[res["Adjusted P-value"] < ENRICH_PADJ].copy()
            sig = sig.sort_values("Adjusted P-value").head(top_n)

            # Sauvegarde CSV
            csv_path = f"{OUT}/tables/{label}_{db_short}.csv"
            sig.to_csv(csv_path, index=False)
            print(f"    {db_short:15s}: {len(sig):4d} termes significatifs → {csv_path}")

            # Barplot
            if not sig.empty:
                _plot_enrichment_bar(sig, label, db_short, top_n)

            all_results[db_short] = sig

        except Exception as e:
            print(f"    {db_short:15s}: ERREUR — {e}")

    return all_results


def _plot_enrichment_bar(sig_df, label, db_short, top_n):
    """Barplot horizontal des termes enrichis (–log10 p-adj)."""
    df = sig_df.head(top_n).copy()
    df["-log10(padj)"] = -np.log10(df["Adjusted P-value"].clip(lower=1e-300))
    df["Term_short"]   = df["Term"].str[:55]   # tronquer les longs noms

    # Palette : du vert clair au rouge foncé selon significance
    palette = sns.color_palette("RdYlGn_r", len(df))

    fig, ax = plt.subplots(figsize=(11, max(4, len(df) * 0.4)))
    bars = ax.barh(
        df["Term_short"][::-1],
        df["-log10(padj)"][::-1],
        color=palette[::-1], edgecolor="white", linewidth=0.5
    )
    ax.axvline(-np.log10(ENRICH_PADJ), color="black",
               linestyle="--", lw=1, label=f"p-adj = {ENRICH_PADJ}")
    ax.set_xlabel("-log10(Adjusted P-value)", fontsize=11)
    ax.set_title(
        f"{label} — {db_short}  |  Top {len(df)} termes enrichis\n"
        f"(adj. p < {ENRICH_PADJ}, {len(sig_df)} gènes input)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path = f"{OUT}/figures/{label}_{db_short}_barplot.png"
    plt.savefig(out_path, dpi=150)
    plt.close()


# ── Lancement pour D1 (tous les gènes) ──────────────────────────────────────
print("\n  ━━━  D1 : tous les DEGs (n=313)  ━━━")
d1_results = run_enrichr(d1_all, label="D1_all")

print("\n  ━━━  D1 : UP-regulated  ━━━")
d1_up_res  = run_enrichr(d1_up,  label="D1_UP")

print("\n  ━━━  D1 : DOWN-regulated  ━━━")
d1_dn_res  = run_enrichr(d1_down, label="D1_DOWN")

# ── Lancement pour D2 (filtrés) ─────────────────────────────────────────────
print("\n  ━━━  D2 : tous les DEGs filtrés (n=6232)  ━━━")
d2_results = run_enrichr(d2_all, label="D2_all")

print("\n  ━━━  D2 : UP-regulated  ━━━")
d2_up_res  = run_enrichr(d2_up,  label="D2_UP")

print("\n  ━━━  D2 : DOWN-regulated  ━━━")
d2_dn_res  = run_enrichr(d2_down, label="D2_DOWN")


# ─────────────────────────────────────────────────────────────────────────────
# 1.5  COMPARAISON D1 vs D2 — TERMES COMMUNS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 1.5  Termes enrichis communs D1 ∩ D2 ────────────────────────")

summary_rows = []

for db_short in DATABASES.values():
    t1 = set(d1_results.get(db_short, pd.DataFrame()).get("Term", []))
    t2 = set(d2_results.get(db_short, pd.DataFrame()).get("Term", []))
    common = t1 & t2
    summary_rows.append({
        "Database" : db_short,
        "D1_terms" : len(t1),
        "D2_terms" : len(t2),
        "Common"   : len(common),
        "Common_terms" : "; ".join(sorted(common)[:10])   # top 10 pour lisibilité
    })
    if common:
        print(f"  {db_short:15s}: {len(common)} termes communs")
        for t in sorted(common)[:5]:
            print(f"      • {t}")

summary_df = pd.DataFrame(summary_rows)
summary_csv = f"{OUT}/tables/D1_D2_common_terms_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"\n  Résumé comparatif sauvegardé → {summary_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.6  RÉSUMÉ FINAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RÉSUMÉ — ÉTAPE 1 TERMINÉE")
print("=" * 65)
print(f"\n  Fichiers générés dans : {OUT}/")
print(f"  ├── figures/    → volcano plots + barplots d'enrichissement")
print(f"  ├── tables/     → CSV de tous les termes enrichis")
print(f"  └── *.txt       → listes de gènes (all/UP/DOWN)")
print(f"\n  Prochaine étape → Étape 2 : Fusion & préparation données ML")
print("=" * 65)

"""
╔══════════════════════════════════════════════════════════════════════╗
║  ÉTAPE 2 — PRÉPARATION DES DONNÉES POUR MACHINE LEARNING            ║
║  Fusion D1 + D2 · Filtrage DEGs · Standardisation · Train/Test      ║
╚══════════════════════════════════════════════════════════════════════╝

FICHIERS NÉCESSAIRES (même dossier que ce script) :
    normalized_counts_D1.csv   → 313 gènes × 7  échantillons (sep=;)
    normalized_counts_D2.csv   → 7374 gènes × 20 échantillons (sep=;)
    D1.csv                     → DEGs D1 avec GeneID, log2(FC), P-adj
    D2.csv                     → DEGs D2 avec GeneID, log2(FC), P-adj

OUTPUTS GÉNÉRÉS :
    resultats/etape2/
    ├── matrice_fusionnee.csv          ← 27 échantillons × gènes communs
    ├── matrice_standardisee.csv       ← après StandardScaler
    ├── labels.csv                     ← condition par échantillon
    ├── features_retenus.txt           ← liste des gènes utilisés
    ├── train_X.csv / test_X.csv
    ├── train_y.csv / test_y.csv
    ├── data_ml.pkl                    ← données prêtes pour Étape 3
    └── figures/
        ├── heatmap_top50.png
        ├── pca_plot.png
        └── distribution_avant_apres_norm.png
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
COUNTS_D1  = "normalized_counts_D1.csv"
COUNTS_D2  = "normalized_counts_D2.csv"
DEGS_D1    = "D1.csv"
DEGS_D2    = "D2.csv"
SEP        = ";"

D2_PADJ_THRESH = 0.05
D2_LFC_THRESH  = 1.0
TEST_SIZE      = 0.30
RANDOM_STATE   = 42

OUT = "resultats/etape2"
os.makedirs(f"{OUT}/figures", exist_ok=True)

print("=" * 65)
print("  ÉTAPE 2 : PRÉPARATION DES DONNÉES ML")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 2.1  CHARGEMENT DES COUNTS NORMALISÉS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 2.1  Chargement des counts normalisés ────────────────────────")

c1 = pd.read_csv(COUNTS_D1, sep=SEP, index_col=0)
c2 = pd.read_csv(COUNTS_D2, sep=SEP, index_col=0)

print(f"  D1 counts : {c1.shape[0]} gènes × {c1.shape[1]} échantillons")
print(f"  D2 counts : {c2.shape[0]} gènes × {c2.shape[1]} échantillons")
print(f"  D1 échantillons : {list(c1.columns)}")
print(f"  D2 échantillons : {list(c2.columns)}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.2  CHARGEMENT DES DEGs & FILTRAGE D2
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 2.2  Chargement et filtrage des DEGs ─────────────────────────")

deg1 = pd.read_csv(DEGS_D1, sep=SEP)
deg2 = pd.read_csv(DEGS_D2, sep=SEP)
deg1.columns = deg1.columns.str.strip()
deg2.columns = deg2.columns.str.strip()

# Normaliser les IDs : enlever la version (.14, .10 etc.)
deg1["GeneID_clean"] = deg1["GeneID"].str.split(".").str[0]
deg2["GeneID_clean"] = deg2["GeneID"].str.split(".").str[0]
c1.index = c1.index.str.split(".").str[0]
c2.index = c2.index.str.split(".").str[0]

degs_d1 = set(deg1["GeneID_clean"].tolist())
print(f"  D1 DEGs retenus : {len(degs_d1)}")

deg2_filt = deg2[
    (deg2["P-adj"] < D2_PADJ_THRESH) &
    (deg2["log2(FC)"].abs() > D2_LFC_THRESH)
]
degs_d2 = set(deg2_filt["GeneID_clean"].tolist())
print(f"  D2 DEGs après filtrage (p<{D2_PADJ_THRESH}, |lfc|>{D2_LFC_THRESH}) : {len(degs_d2)}")

common_genes = degs_d1 & degs_d2
print(f"\n  Gènes communs D1 ∩ D2 : {len(common_genes)}")

if len(common_genes) == 0:
    print("  ⚠️  Aucun gène commun → on utilise tous les DEGs de D1 (union)")
    use_common = False
else:
    print(f"  ✅  {len(common_genes)} gènes communs utilisés pour le ML")
    use_common = True


# ─────────────────────────────────────────────────────────────────────────────
# 2.3  CONSTRUCTION DE LA MATRICE FUSIONNÉE
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 2.3  Fusion des datasets ──────────────────────────────────────")

if use_common:
    genes_list = list(common_genes)
    c1_sub = c1.loc[c1.index.isin(genes_list)]
    c2_sub = c2.loc[c2.index.isin(genes_list)]
    X1 = c1_sub.T
    X2 = c2_sub.T
    X2 = X2.reindex(columns=X1.columns, fill_value=0)
else:
    c1_sub = c1.loc[c1.index.isin(degs_d1)]
    c2_sub = c2.loc[c2.index.isin(degs_d2)]
    X1 = c1_sub.T
    X2 = c2_sub.T
    X2 = X2.reindex(columns=X1.columns, fill_value=0)

X_merged = pd.concat([X1, X2], axis=0)
print(f"  Matrice fusionnée : {X_merged.shape[0]} échantillons × {X_merged.shape[1]} gènes")

nan_count = X_merged.isna().sum().sum()
if nan_count > 0:
    print(f"  ⚠️  {nan_count} NaN → remplacement par 0")
    X_merged = X_merged.fillna(0)
else:
    print(f"  ✅  Aucun NaN")


# ─────────────────────────────────────────────────────────────────────────────
# 2.4  CONSTRUCTION DES LABELS (y)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 2.4  Construction des labels ──────────────────────────────────")

def assign_label(sample_name):
    name_lower = sample_name.lower()
    if any(k in name_lower for k in ["treated", "endo", "case"]):
        return "endometriosis"
    elif any(k in name_lower for k in ["control", "ctrl"]):
        return "control"
    else:
        raise ValueError(f"Label impossible pour : {sample_name}")

labels = pd.Series(
    {s: assign_label(s) for s in X_merged.index},
    name="condition"
)

print(f"  endometriosis : {(labels == 'endometriosis').sum()} échantillons")
print(f"  control       : {(labels == 'control').sum()} échantillons")

labels.to_csv(f"{OUT}/labels.csv", header=True)
print(f"  Labels → {OUT}/labels.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 2.5  DISTRIBUTION AVANT / APRÈS STANDARDISATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 2.5  Visualisation distribution ─────────────────────────────")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

vals_log = np.log1p(X_merged.values.flatten())
axes[0].hist(vals_log[vals_log > 0], bins=80,
             color="#4C72B0", alpha=0.85, edgecolor="none")
axes[0].set_title("Avant standardisation\n(log1p des counts normalisés)")
axes[0].set_xlabel("log1p(expression)")
axes[0].set_ylabel("Fréquence")

scaler_prev = StandardScaler()
X_prev = scaler_prev.fit_transform(X_merged.values)
axes[1].hist(X_prev.flatten(), bins=80,
             color="#DD8452", alpha=0.85, edgecolor="none")
axes[1].set_title("Après StandardScaler\n(moyenne=0, std=1)")
axes[1].set_xlabel("Valeur standardisée")
axes[1].set_ylabel("Fréquence")

plt.suptitle("Distribution des données d'expression", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/figures/distribution_avant_apres_norm.png", dpi=150)
plt.close()
print(f"  Distribution → {OUT}/figures/distribution_avant_apres_norm.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2.6  STANDARDISATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 2.6  Standardisation ──────────────────────────────────────────")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_merged.values)

X_scaled_df = pd.DataFrame(
    X_scaled, index=X_merged.index, columns=X_merged.columns
)

X_merged.to_csv(f"{OUT}/matrice_fusionnee.csv")
X_scaled_df.to_csv(f"{OUT}/matrice_standardisee.csv")

with open(f"{OUT}/features_retenus.txt", "w") as f:
    f.write("\n".join(X_merged.columns.tolist()))

print(f"  matrice_fusionnee.csv    → {OUT}/")
print(f"  matrice_standardisee.csv → {OUT}/")
print(f"  features_retenus.txt ({X_merged.shape[1]} gènes)")


# ─────────────────────────────────────────────────────────────────────────────
# 2.7  HEATMAP TOP 50 GÈNES
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 2.7  Heatmap top 50 gènes ────────────────────────────────────")

variances   = X_scaled_df.var(axis=0)
top50_genes = variances.nlargest(50).index
X_top50     = X_scaled_df[top50_genes]

palette_colors = {"endometriosis": "#E8312A", "control": "#2E75B6"}

fig, ax = plt.subplots(figsize=(14, 9))
sns.heatmap(
    X_top50.T,
    cmap="RdBu_r", center=0, ax=ax,
    xticklabels=True, yticklabels=True,
    linewidths=0,
    cbar_kws={"label": "Expression standardisée"},
)
ax.set_title(
    "Heatmap — Top 50 gènes (variance maximale)\nD1 + D2 fusionnés",
    fontsize=12, fontweight="bold"
)
ax.set_xlabel("Échantillons", fontsize=10)
ax.set_ylabel("Gènes", fontsize=10)
ax.tick_params(axis='y', labelsize=6)
ax.tick_params(axis='x', labelsize=8, rotation=45)

legend_elems = [
    mpatches.Patch(facecolor="#E8312A", label="Endométriose (Treated)"),
    mpatches.Patch(facecolor="#2E75B6", label="Contrôle (Control)"),
]
ax.legend(handles=legend_elems, loc="upper right",
          bbox_to_anchor=(1.15, 1.12), fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/figures/heatmap_top50.png", dpi=150)
plt.close()
print(f"  Heatmap → {OUT}/figures/heatmap_top50.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2.8  ACP (PCA) — Visualisation 2D
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 2.8  ACP (PCA) ───────────────────────────────────────────────")

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca    = pca.fit_transform(X_scaled)
var_exp  = pca.explained_variance_ratio_ * 100

pca_df = pd.DataFrame(X_pca, columns=["PC1","PC2"], index=X_merged.index)
pca_df["condition"] = labels.values

fig, ax = plt.subplots(figsize=(8, 6))
for cond, color in palette_colors.items():
    sub = pca_df[pca_df["condition"] == cond]
    ax.scatter(sub["PC1"], sub["PC2"],
               label=cond, color=color, s=90,
               alpha=0.85, edgecolors="white", linewidths=0.6)
    for idx, row in sub.iterrows():
        ax.annotate(idx, (row["PC1"], row["PC2"]),
                    fontsize=6.5, alpha=0.75,
                    xytext=(3, 3), textcoords="offset points")

ax.axhline(0, color="gray", lw=0.5, linestyle="--")
ax.axvline(0, color="gray", lw=0.5, linestyle="--")
ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)", fontsize=11)
ax.set_title(
    "ACP — D1 + D2 fusionnés\nExpression des DEGs",
    fontsize=12, fontweight="bold"
)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/figures/pca_plot.png", dpi=150)
plt.close()
print(f"  PCA → {OUT}/figures/pca_plot.png")
print(f"  Variance expliquée : PC1={var_exp[0]:.1f}%, PC2={var_exp[1]:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 2.9  SPLIT TRAIN / TEST (70/30 stratifié)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 2.9  Split Train / Test ───────────────────────────────────────")

le = LabelEncoder()
y_enc = le.fit_transform(labels.values)
print(f"  Encodage : {dict(zip(le.classes_, le.transform(le.classes_)))}")

all_idx = np.arange(len(labels))
idx_train, idx_test, y_train, y_test = train_test_split(
    all_idx, y_enc,
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
)

X_train = X_scaled[idx_train]
X_test  = X_scaled[idx_test]
sample_names = X_merged.index.tolist()

print(f"  Train : {len(idx_train)} échantillons")
print(f"  Test  : {len(idx_test)}  échantillons")
print(f"  Features : {X_train.shape[1]} gènes")

pd.DataFrame(X_train, columns=X_merged.columns,
             index=[sample_names[i] for i in idx_train]
             ).to_csv(f"{OUT}/train_X.csv")
pd.DataFrame(X_test, columns=X_merged.columns,
             index=[sample_names[i] for i in idx_test]
             ).to_csv(f"{OUT}/test_X.csv")
pd.Series(y_train, name="label",
          index=[sample_names[i] for i in idx_train]
          ).to_csv(f"{OUT}/train_y.csv")
pd.Series(y_test, name="label",
          index=[sample_names[i] for i in idx_test]
          ).to_csv(f"{OUT}/test_y.csv")

print(f"  Fichiers train/test sauvegardés dans {OUT}/")


# ─────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE PICKLE POUR ÉTAPE 3
# ─────────────────────────────────────────────────────────────────────────────
data_ml = {
    "X_train"      : X_train,
    "X_test"       : X_test,
    "y_train"      : y_train,
    "y_test"       : y_test,
    "feature_names": X_merged.columns.tolist(),
    "label_encoder": le,
    "sample_names" : sample_names,
}
with open(f"{OUT}/data_ml.pkl", "wb") as f:
    pickle.dump(data_ml, f)


# ─────────────────────────────────────────────────────────────────────────────
# RÉSUMÉ FINAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RÉSUMÉ — ÉTAPE 2 TERMINÉE ✅")
print("=" * 65)
print(f"""
  ┌──────────────────────────────────────────────────────┐
  │  Échantillons totaux    : {len(labels):>3}                         │
  │    ↳ Endométriose       : {(labels=='endometriosis').sum():>3}                         │
  │    ↳ Contrôle           : {(labels=='control').sum():>3}                         │
  │  Gènes (features)       : {X_merged.shape[1]:>3}                         │
  │  Train set              : {len(idx_train):>3} échantillons              │
  │  Test set               : {len(idx_test):>3} échantillons              │
  └──────────────────────────────────────────────────────┘

  Fichiers générés dans : {OUT}/
    matrice_fusionnee.csv     · matrice_standardisee.csv
    train_X.csv · test_X.csv  · train_y.csv · test_y.csv
    labels.csv  · features_retenus.txt · data_ml.pkl
    figures/ → distribution · heatmap_top50 · pca_plot
""")
print("  ➡  Prochaine étape → Étape 3 : Modèles ML (RF, SVM, LASSO)")
print("=" * 65)


"""
=================================================================
  ÉTAPES 3, 4 & 5 — MACHINE LEARNING & SÉLECTION DE BIOMARQUEURS
  Mémoire : Transcriptome + ML → Biomarqueurs Endométriose
=================================================================
  Prérequis : résultats de l'étape 2 dans resultats/etape2/
    - train_X.csv, test_X.csv, train_y.csv, test_y.csv
    - features_retenus.txt
=================================================================
"""

import os, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LassoCV, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, cross_val_predict, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── Couleurs globales ────────────────────────────────────────────
PALETTE = {
    "UP":      "#E05C5C",
    "DOWN":    "#5C8BE0",
    "bg":      "#F7F9FC",
    "accent":  "#8A4FFF",
    "text":    "#1A1A2E",
    "grid":    "#DDE3EE",
}

# ─── Dossiers de sortie ───────────────────────────────────────────
for d in ["resultats/etape3/figures",
          "resultats/etape3/tables",
          "resultats/etape4/figures",
          "resultats/etape4/tables",
          "resultats/etape5/figures",
          "resultats/etape5/tables"]:
    os.makedirs(d, exist_ok=True)


# =================================================================
#  CHARGEMENT DES DONNÉES (issues de l'étape 2)
# =================================================================
print("=================================================================")
print("  CHARGEMENT DES DONNÉES — ÉTAPE 2 → ÉTAPE 3")
print("=================================================================\n")

train_X = pd.read_csv("resultats/etape2/train_X.csv", index_col=0)
test_X  = pd.read_csv("resultats/etape2/test_X.csv",  index_col=0)
train_y = pd.read_csv("resultats/etape2/train_y.csv", index_col=0).squeeze()
test_y  = pd.read_csv("resultats/etape2/test_y.csv",  index_col=0).squeeze()

with open("resultats/etape2/features_retenus.txt") as f:
    gene_names = [l.strip() for l in f if l.strip()]

print(f"  Train : {train_X.shape[0]} échantillons × {train_X.shape[1]} gènes")
print(f"  Test  : {test_X.shape[0]} échantillons × {test_X.shape[1]} gènes")
print(f"  Features : {len(gene_names)} gènes\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# =================================================================
#  ÉTAPE 3 : MODÈLES DE MACHINE LEARNING
# =================================================================
print("=================================================================")
print("  ÉTAPE 3 : ENTRAÎNEMENT DES MODÈLES ML")
print("=================================================================\n")

# ── 3.1  Random Forest ───────────────────────────────────────────
print("── 3.1  Random Forest ──────────────────────────────────────────")

param_grid_rf = {
    "n_estimators": [100, 300, 500],
    "max_depth":    [None, 5, 10],
    "min_samples_split": [2, 5],
}
rf_gs = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced"),
    param_grid_rf, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0
)
rf_gs.fit(train_X, train_y)
rf_best = rf_gs.best_estimator_

print(f"  Meilleurs hyperparamètres RF : {rf_gs.best_params_}")
print(f"  Meilleur AUC CV (train)      : {rf_gs.best_score_:.4f}\n")

# Importance des gènes
rf_importances = pd.Series(rf_best.feature_importances_, index=gene_names).sort_values(ascending=False)
rf_importances.to_csv("resultats/etape3/tables/RF_feature_importances.csv", header=["importance"])

# ── 3.2  SVM ─────────────────────────────────────────────────────
print("── 3.2  SVM ────────────────────────────────────────────────────")

param_grid_svm = {
    "svm__C":      [0.1, 1, 10, 100],
    "svm__kernel": ["rbf", "linear"],
    "svm__gamma":  ["scale", "auto"],
}
svm_pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC(probability=True, class_weight="balanced", random_state=42))])
svm_gs = GridSearchCV(svm_pipe, param_grid_svm, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
svm_gs.fit(train_X, train_y)
svm_best = svm_gs.best_estimator_

print(f"  Meilleurs hyperparamètres SVM : {svm_gs.best_params_}")
print(f"  Meilleur AUC CV (train)       : {svm_gs.best_score_:.4f}\n")

# ── 3.3  LASSO Logistic Regression ───────────────────────────────
print("── 3.3  LASSO (Logistic Regression L1) ─────────────────────────")

lasso_model = LogisticRegression(
    penalty="l1", solver="liblinear", class_weight="balanced",
    max_iter=10000, random_state=42
)
Cs = np.logspace(-3, 2, 30)
lasso_gs = GridSearchCV(lasso_model, {"C": Cs}, cv=cv, scoring="roc_auc", n_jobs=-1)
lasso_gs.fit(train_X, train_y)
lasso_best = lasso_gs.best_estimator_

lasso_coefs = pd.Series(lasso_best.coef_[0], index=gene_names)
n_nonzero_lasso = (lasso_coefs != 0).sum()
print(f"  Meilleur C LASSO : {lasso_gs.best_params_['C']:.4f}")
print(f"  AUC CV (train)   : {lasso_gs.best_score_:.4f}")
print(f"  Gènes sélectionnés (coef ≠ 0) : {n_nonzero_lasso}\n")

lasso_coefs.to_csv("resultats/etape3/tables/LASSO_coefficients.csv", header=["coef"])

# ── 3.4  Elastic Net ─────────────────────────────────────────────
print("── 3.4  Elastic Net (Logistic Regression L1+L2) ─────────────────")

enet_model = LogisticRegression(
    penalty="elasticnet", solver="saga", class_weight="balanced",
    max_iter=10000, random_state=42
)
param_grid_enet = {"C": np.logspace(-3, 2, 15), "l1_ratio": [0.1, 0.5, 0.7, 0.9]}
enet_gs = GridSearchCV(enet_model, param_grid_enet, cv=cv, scoring="roc_auc", n_jobs=-1)
enet_gs.fit(train_X, train_y)
enet_best = enet_gs.best_estimator_

enet_coefs = pd.Series(enet_best.coef_[0], index=gene_names)
n_nonzero_enet = (enet_coefs != 0).sum()
print(f"  Meilleurs params Elastic Net : {enet_gs.best_params_}")
print(f"  AUC CV (train)               : {enet_gs.best_score_:.4f}")
print(f"  Gènes sélectionnés (coef ≠ 0): {n_nonzero_enet}\n")

enet_coefs.to_csv("resultats/etape3/tables/ElasticNet_coefficients.csv", header=["coef"])

# Sauvegarde des modèles
models = {"RandomForest": rf_best, "SVM": svm_best, "LASSO": lasso_best, "ElasticNet": enet_best}
with open("resultats/etape3/models.pkl", "wb") as f:
    pickle.dump(models, f)
print("  Modèles sauvegardés → resultats/etape3/models.pkl\n")

# ── Figure 3.1 : Importance RF (top 20) ──────────────────────────
fig, ax = plt.subplots(figsize=(9, 6), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])
top20 = rf_importances.head(20)
bars = ax.barh(top20.index[::-1], top20.values[::-1],
               color=PALETTE["accent"], edgecolor="white", linewidth=0.5)
ax.set_xlabel("Importance (Mean Decrease Impurity)", fontsize=11)
ax.set_title("Random Forest — Top 20 gènes importants", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color=PALETTE["grid"], linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("resultats/etape3/figures/RF_feature_importance_top20.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Figure → resultats/etape3/figures/RF_feature_importance_top20.png")

# ── Figure 3.2 : Coefficients LASSO ──────────────────────────────
nonzero = lasso_coefs[lasso_coefs != 0].sort_values()
if len(nonzero) > 0:
    fig, ax = plt.subplots(figsize=(9, max(4, len(nonzero) * 0.35)), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    colors = [PALETTE["UP"] if v > 0 else PALETTE["DOWN"] for v in nonzero.values]
    ax.barh(nonzero.index, nonzero.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=PALETTE["text"], linewidth=1)
    ax.set_xlabel("Coefficient LASSO", fontsize=11)
    ax.set_title("LASSO — Coefficients non nuls (biomarqueurs candidats)", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=PALETTE["grid"], linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("resultats/etape3/figures/LASSO_coefficients.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Figure → resultats/etape3/figures/LASSO_coefficients.png\n")


# =================================================================
#  ÉTAPE 4 : ÉVALUATION DES MODÈLES
# =================================================================
print("\n=================================================================")
print("  ÉTAPE 4 : ÉVALUATION DES MODÈLES")
print("=================================================================\n")


def eval_model(name, model, X_train, y_train, X_test, y_test, cv):
    """Évalue un modèle : CV + test set + métriques complètes."""
    # Cross-validation sur train
    cv_res = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring=["accuracy", "roc_auc", "precision", "recall", "f1"],
        return_train_score=False, n_jobs=-1
    )
    # Prédictions sur test
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Modèle":              name,
        "CV Accuracy (mean)":  cv_res["test_accuracy"].mean(),
        "CV Accuracy (std)":   cv_res["test_accuracy"].std(),
        "CV AUC (mean)":       cv_res["test_roc_auc"].mean(),
        "CV AUC (std)":        cv_res["test_roc_auc"].std(),
        "CV Precision (mean)": cv_res["test_precision"].mean(),
        "CV Recall (mean)":    cv_res["test_recall"].mean(),
        "CV F1 (mean)":        cv_res["test_f1"].mean(),
        "Test Accuracy":       accuracy_score(y_test, y_pred),
        "Test AUC":            roc_auc_score(y_test, y_proba),
        "Test Precision":      precision_score(y_test, y_pred, zero_division=0),
        "Test Recall":         recall_score(y_test, y_pred, zero_division=0),
        "Test F1":             f1_score(y_test, y_pred, zero_division=0),
    }
    return metrics, y_pred, y_proba


all_metrics = []
predictions = {}

for name, model in models.items():
    print(f"── Évaluation : {name} ─────────────────────────────────────────")
    m, y_pred, y_proba = eval_model(name, model, train_X, train_y, test_X, test_y, cv)
    all_metrics.append(m)
    predictions[name] = {"y_pred": y_pred, "y_proba": y_proba}

    print(f"  CV  AUC  : {m['CV AUC (mean)']:.4f} ± {m['CV AUC (std)']:.4f}")
    print(f"  CV  Acc  : {m['CV Accuracy (mean)']:.4f} ± {m['CV Accuracy (std)']:.4f}")
    print(f"  Test AUC : {m['Test AUC']:.4f}")
    print(f"  Test Acc : {m['Test Accuracy']:.4f}\n")

metrics_df = pd.DataFrame(all_metrics).set_index("Modèle")
metrics_df.to_csv("resultats/etape4/tables/evaluation_metrics.csv")
print("  Métriques → resultats/etape4/tables/evaluation_metrics.csv\n")

# ── Figure 4.1 : Courbes ROC ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])
colors_roc = [PALETTE["accent"], PALETTE["UP"], PALETTE["DOWN"], "#2ECC71"]

for i, (name, preds) in enumerate(predictions.items()):
    fpr, tpr, _ = roc_curve(test_y, preds["y_proba"])
    auc_val = roc_auc_score(test_y, preds["y_proba"])
    ax.plot(fpr, tpr, color=colors_roc[i], lw=2,
            label=f"{name} (AUC = {auc_val:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
ax.set_xlabel("Taux de faux positifs (FPR)", fontsize=11)
ax.set_ylabel("Taux de vrais positifs (TPR)", fontsize=11)
ax.set_title("Courbes ROC — Comparaison des modèles", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color=PALETTE["grid"], linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("resultats/etape4/figures/ROC_curves_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Figure → resultats/etape4/figures/ROC_curves_comparison.png")

# ── Figure 4.2 : Matrices de confusion ───────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor=PALETTE["bg"])
axes = axes.flatten()
class_labels = ["Control", "Endométriose"]

for i, (name, preds) in enumerate(predictions.items()):
    cm = confusion_matrix(test_y, preds["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
    axes[i].set_title(f"{name}", fontsize=12, fontweight="bold")
    axes[i].set_facecolor(PALETTE["bg"])

fig.suptitle("Matrices de confusion — Test set", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("resultats/etape4/figures/confusion_matrices.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Figure → resultats/etape4/figures/confusion_matrices.png")

# ── Figure 4.3 : Comparaison des métriques ────────────────────────
metric_cols = ["Test Accuracy", "Test AUC", "Test Precision", "Test Recall", "Test F1"]
plot_data = metrics_df[metric_cols].copy()

fig, ax = plt.subplots(figsize=(10, 5), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])
x = np.arange(len(metric_cols))
width = 0.18
bar_colors = [PALETTE["accent"], PALETTE["UP"], PALETTE["DOWN"], "#2ECC71"]

for i, (model_name, row) in enumerate(plot_data.iterrows()):
    ax.bar(x + i * width, row.values, width, label=model_name,
           color=bar_colors[i], edgecolor="white", linewidth=0.5, alpha=0.9)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([m.replace("Test ", "") for m in metric_cols], fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Comparaison des métriques sur le test set", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=PALETTE["grid"], linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("resultats/etape4/figures/metrics_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Figure → resultats/etape4/figures/metrics_comparison.png\n")

# ── Figure 4.4 : CV scores boxplot ───────────────────────────────
cv_scores_data = {}
for name, model in models.items():
    cv_auc = cross_validate(model, train_X, train_y, cv=cv,
                            scoring="roc_auc", n_jobs=-1)["test_score"]
    cv_scores_data[name] = cv_auc

fig, ax = plt.subplots(figsize=(8, 5), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])
bp = ax.boxplot(
    [cv_scores_data[n] for n in models],
    labels=list(models.keys()),
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2)
)
for patch, color in zip(bp["boxes"], bar_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
ax.set_ylabel("AUC (5-fold CV)", fontsize=11)
ax.set_title("Distribution des AUC en validation croisée (5-fold)", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=PALETTE["grid"], linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("resultats/etape4/figures/CV_AUC_boxplot.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Figure → resultats/etape4/figures/CV_AUC_boxplot.png\n")

# ── Sélection du meilleur modèle ──────────────────────────────────
best_model_name = metrics_df["Test AUC"].idxmax()
print(f"  ✅ Meilleur modèle : {best_model_name} (AUC test = {metrics_df.loc[best_model_name, 'Test AUC']:.4f})\n")


# =================================================================
#  ÉTAPE 5 : SÉLECTION FINALE DES BIOMARQUEURS
# =================================================================
print("\n=================================================================")
print("  ÉTAPE 5 : SÉLECTION FINALE DES BIOMARQUEURS")
print("=================================================================\n")

# ── 5.1  Score composite multi-méthodes ──────────────────────────
print("── 5.1  Calcul du score composite ──────────────────────────────")

# Normalisation rang RF (top gene = rank 1)
rf_ranks = rf_importances.rank(ascending=False)
rf_score = (rf_importances - rf_importances.min()) / (rf_importances.max() - rf_importances.min())

# LASSO : valeur absolue normalisée
if lasso_coefs.abs().max() > 0:
    lasso_score = lasso_coefs.abs() / lasso_coefs.abs().max()
else:
    lasso_score = lasso_coefs.abs()

# ElasticNet : valeur absolue normalisée
if enet_coefs.abs().max() > 0:
    enet_score = enet_coefs.abs() / enet_coefs.abs().max()
else:
    enet_score = enet_coefs.abs()

composite = (rf_score + lasso_score + enet_score) / 3
composite = composite.sort_values(ascending=False)

biomarkers_df = pd.DataFrame({
    "Gene":              composite.index,
    "Composite_Score":   composite.values,
    "RF_Importance":     rf_importances[composite.index].values,
    "LASSO_Coef":        lasso_coefs[composite.index].values,
    "ElasticNet_Coef":   enet_coefs[composite.index].values,
    "LASSO_selected":    (lasso_coefs[composite.index] != 0).values,
    "ElasticNet_selected": (enet_coefs[composite.index] != 0).values,
})
biomarkers_df = biomarkers_df.sort_values("Composite_Score", ascending=False).reset_index(drop=True)
biomarkers_df.to_csv("resultats/etape5/tables/biomarqueurs_scores_complets.csv", index=False)
print(f"  Tableau complet → resultats/etape5/tables/biomarqueurs_scores_complets.csv")

# ── 5.2  Sélection finale (top candidats) ────────────────────────
print("\n── 5.2  Sélection finale des biomarqueurs ───────────────────────")

# Critère : sélectionné par LASSO ET/OU ElasticNet + top RF
selected_lasso = set(lasso_coefs[lasso_coefs != 0].index)
selected_enet  = set(enet_coefs[enet_coefs != 0].index)
selected_both  = selected_lasso & selected_enet
selected_union = selected_lasso | selected_enet

# Top 20 RF
top20_rf = set(rf_importances.head(20).index)

# Biomarqueurs finaux : union des méthodes ∩ top RF + score composite
final_candidates = biomarkers_df[
    (biomarkers_df["LASSO_selected"] | biomarkers_df["ElasticNet_selected"]) |
    (biomarkers_df["Gene"].isin(top20_rf))
].head(30).copy()

final_candidates["Méthodes"] = final_candidates.apply(
    lambda r: " | ".join(filter(None, [
        "LASSO" if r["LASSO_selected"] else "",
        "ElasticNet" if r["ElasticNet_selected"] else "",
        "RF_Top20" if r["Gene"] in top20_rf else ""
    ])), axis=1
)

final_candidates.to_csv("resultats/etape5/tables/biomarqueurs_finaux.csv", index=False)
print(f"  Biomarqueurs finaux (n={len(final_candidates)}) → resultats/etape5/tables/biomarqueurs_finaux.csv")

# Top 10 affichés
print(f"\n  ┌─ TOP 10 BIOMARQUEURS CANDIDATS ─────────────────────────────")
for i, row in final_candidates.head(10).iterrows():
    print(f"  │  {i+1:2d}. {row['Gene']:<15}  Score={row['Composite_Score']:.4f}  [{row['Méthodes']}]")
print("  └──────────────────────────────────────────────────────────────\n")

# Liste simple
with open("resultats/etape5/tables/liste_biomarqueurs_finaux.txt", "w") as f:
    f.write("\n".join(final_candidates["Gene"].tolist()))
print("  Liste txt → resultats/etape5/tables/liste_biomarqueurs_finaux.txt\n")

# ── Figure 5.1 : Score composite top 20 ──────────────────────────
top_plot = biomarkers_df.head(20)
fig, ax = plt.subplots(figsize=(9, 7), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])

bar_colors_bio = [
    PALETTE["accent"] if (row["LASSO_selected"] and row["ElasticNet_selected"])
    else PALETTE["UP"] if row["LASSO_selected"] or row["ElasticNet_selected"]
    else PALETTE["DOWN"]
    for _, row in top_plot.iterrows()
]

ax.barh(top_plot["Gene"][::-1], top_plot["Composite_Score"][::-1],
        color=bar_colors_bio[::-1], edgecolor="white", linewidth=0.5)
ax.set_xlabel("Score composite normalisé", fontsize=11)
ax.set_title("Top 20 biomarqueurs candidats — Score composite\n(RF + LASSO + ElasticNet)",
             fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color=PALETTE["grid"], linestyle="--", alpha=0.7)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=PALETTE["accent"], label="LASSO + ElasticNet"),
    Patch(facecolor=PALETTE["UP"],     label="LASSO ou ElasticNet"),
    Patch(facecolor=PALETTE["DOWN"],   label="RF seulement"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("resultats/etape5/figures/biomarqueurs_composite_score.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Figure → resultats/etape5/figures/biomarqueurs_composite_score.png")

# ── Figure 5.2 : Heatmap des biomarqueurs finaux ─────────────────
try:
    full_data = pd.read_csv("resultats/etape2/matrice_standardisee.csv", index_col=0)
    labels    = pd.read_csv("resultats/etape2/labels.csv", index_col=0).squeeze()

    top_genes_avail = [g for g in final_candidates["Gene"].head(20) if g in full_data.columns]
    if top_genes_avail:
        heat_data = full_data[top_genes_avail].T

        group_colors = labels.map({"control": PALETTE["DOWN"], "endometriosis": PALETTE["UP"]})
        col_colors_map = group_colors if group_colors.index.isin(heat_data.columns).any() \
                         else heat_data.columns.map(lambda c: group_colors.get(c, "#AAAAAA"))

        g = sns.clustermap(
            heat_data,
            col_colors=group_colors.reindex(heat_data.columns),
            cmap="RdBu_r", center=0,
            figsize=(12, 8),
            dendrogram_ratio=(0.15, 0.2),
            cbar_pos=(0.02, 0.8, 0.03, 0.18),
            yticklabels=True, xticklabels=True
        )
        g.fig.suptitle("Heatmap — Top biomarqueurs candidats", fontsize=13, fontweight="bold", y=1.01)
        plt.savefig("resultats/etape5/figures/heatmap_biomarqueurs_finaux.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("  Figure → resultats/etape5/figures/heatmap_biomarqueurs_finaux.png")
except Exception as e:
    print(f"  [WARN] Heatmap biomarqueurs : {e}")

# ── Figure 5.3 : Venn diagram (gènes communs méthodes) ───────────
try:
    from matplotlib_venn import venn3
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    venn3(
        [top20_rf, selected_lasso, selected_enet],
        set_labels=("RF Top20", "LASSO", "ElasticNet"),
        set_colors=(PALETTE["DOWN"], PALETTE["UP"], PALETTE["accent"]),
        alpha=0.6,
        ax=ax
    )
    ax.set_title("Overlap des gènes sélectionnés par chaque méthode", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("resultats/etape5/figures/venn_methodes.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Figure → resultats/etape5/figures/venn_methodes.png")
except ImportError:
    print("  [INFO] matplotlib_venn non installé — diagramme de Venn ignoré")
    print("         Installez avec : pip install matplotlib-venn")


# =================================================================
#  RÉSUMÉ FINAL
# =================================================================
print("\n=================================================================")
print("  RÉSUMÉ FINAL — PIPELINE COMPLET ✅")
print("=================================================================\n")

print("  ┌──────────────────────────────────────────────────────────┐")
print("  │  ÉTAPE 3 : Modèles entraînés                             │")
for name in models:
    auc_cv  = [cv_scores_data[name].mean()]
    auc_t   = metrics_df.loc[name, "Test AUC"]
    print(f"  │    {name:<15} AUC CV={auc_cv[0]:.3f}  AUC test={auc_t:.3f}     │")
print("  │                                                          │")
print(f"  │  ÉTAPE 4 : Meilleur modèle → {best_model_name:<12}              │")
print(f"  │    AUC test = {metrics_df.loc[best_model_name,'Test AUC']:.4f}                              │")
print("  │                                                          │")
print(f"  │  ÉTAPE 5 : {len(final_candidates)} biomarqueurs candidats identifiés         │")
print(f"  │    Top biomarqueur : {final_candidates['Gene'].iloc[0]:<20}              │")
print("  └──────────────────────────────────────────────────────────┘\n")

print("  Fichiers générés :")
print("    resultats/etape3/ → modèles + importances RF + coefficients LASSO/ElasticNet")
print("    resultats/etape4/ → métriques CV + test + courbes ROC + matrices de confusion")
print("    resultats/etape5/ → scores composites + biomarqueurs finaux + heatmap + Venn\n")


"""
=================================================================
  ÉTAPE 6 : ANNOTATION DES BIOMARQUEURS
  Conversion Ensembl IDs → Gene Symbols + Description + Biotype
  Mémoire : Transcriptome + ML → Biomarqueurs Endométriose
=================================================================
  Dépendances : pip install mygene requests pandas
=================================================================
"""

import os, time, json
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ─── Optionnel : pybiomart ────────────────────────────────────────
try:
    from pybiomart import Server
    BIOMART_AVAILABLE = True
except ImportError:
    BIOMART_AVAILABLE = False

# ─── Optionnel : mygene ───────────────────────────────────────────
try:
    import mygene
    MYGENE_AVAILABLE = True
except ImportError:
    MYGENE_AVAILABLE = False

os.makedirs("resultats/etape6/figures", exist_ok=True)
os.makedirs("resultats/etape6/tables",  exist_ok=True)

PALETTE = {
    "UP":     "#E05C5C",
    "DOWN":   "#5C8BE0",
    "bg":     "#F7F9FC",
    "accent": "#8A4FFF",
    "text":   "#1A1A2E",
    "grid":   "#DDE3EE",
    "green":  "#2ECC71",
}

# =================================================================
#  CHARGEMENT DES BIOMARQUEURS (étape 5)
# =================================================================
print("=================================================================")
print("  ÉTAPE 6 : ANNOTATION DES BIOMARQUEURS")
print("=================================================================\n")

biomarkers_df = pd.read_csv("resultats/etape5/tables/biomarqueurs_finaux.csv")
all_scores_df = pd.read_csv("resultats/etape5/tables/biomarqueurs_scores_complets.csv")

ensembl_ids = biomarkers_df["Gene"].tolist()
print(f"  {len(ensembl_ids)} biomarqueurs à annoter\n")

# Nettoyage : retirer la version si présente (ex: ENSG00000168386.5 → ENSG00000168386)
ensembl_ids_clean = [g.split(".")[0] for g in ensembl_ids]
id_map_clean = dict(zip(ensembl_ids_clean, ensembl_ids))  # propre → original


# =================================================================
#  MÉTHODE 1 : Ensembl REST API (sans installation)
# =================================================================
def annotate_via_ensembl_rest(ids):
    """
    Utilise l'API REST Ensembl pour récupérer :
    gene symbol, description, biotype, chromosome, start, end
    Fonctionne sans aucune bibliothèque tierce.
    """
    print("── Annotation via Ensembl REST API ─────────────────────────────")
    url = "https://rest.ensembl.org/lookup/id"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    results = {}
    batch_size = 50  # max recommandé par l'API Ensembl

    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        payload = json.dumps({"ids": batch})
        try:
            resp = requests.post(url, headers=headers, data=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                for eid, info in data.items():
                    if info:
                        results[eid] = {
                            "Gene_Symbol":   info.get("display_name", ""),
                            "Description":   info.get("description", "").split("[")[0].strip(),
                            "Biotype":       info.get("biotype", ""),
                            "Chromosome":    info.get("seq_region_name", ""),
                            "Start":         info.get("start", ""),
                            "End":           info.get("end", ""),
                            "Strand":        info.get("strand", ""),
                            "Species":       info.get("species", ""),
                        }
                    else:
                        results[eid] = {"Gene_Symbol": "", "Description": "", "Biotype": "",
                                        "Chromosome": "", "Start": "", "End": "", "Strand": "", "Species": ""}
                print(f"  Batch {i//batch_size + 1} : {len(batch)} IDs annotés ✓")
                time.sleep(0.3)  # respecter le rate limit
            else:
                print(f"  [WARN] Batch {i//batch_size + 1} : statut {resp.status_code}")
                for eid in batch:
                    results[eid] = {"Gene_Symbol": "", "Description": "", "Biotype": "",
                                    "Chromosome": "", "Start": "", "End": "", "Strand": "", "Species": ""}
        except Exception as e:
            print(f"  [WARN] Batch {i//batch_size + 1} erreur : {e}")
            for eid in batch:
                results[eid] = {"Gene_Symbol": "", "Description": "", "Biotype": "",
                                "Chromosome": "", "Start": "", "End": "", "Strand": "", "Species": ""}

    return results


# =================================================================
#  MÉTHODE 2 : mygene (si disponible)
# =================================================================
def annotate_via_mygene(ids):
    """Utilise mygene pour enrichir avec NCBI/Uniprot info."""
    print("── Enrichissement via MyGene.info ──────────────────────────────")
    mg = mygene.MyGeneInfo()
    try:
        results = mg.querymany(
            ids,
            scopes="ensembl.gene",
            fields="symbol,name,summary,entrezgene,uniprot,go,pathway",
            species="human",
            returnall=True,
            verbose=False,
        )
        out = {}
        for hit in results.get("out", []):
            qid = hit.get("query", "")
            out[qid] = {
                "Symbol_MG":     hit.get("symbol", ""),
                "Name_MG":       hit.get("name", ""),
                "Summary_MG":    hit.get("summary", "")[:200] if hit.get("summary") else "",
                "Entrez_ID":     hit.get("entrezgene", ""),
                "UniProt_ID":    list(hit.get("uniprot", {}).get("Swiss-Prot", [""]))[0]
                                  if hit.get("uniprot") else "",
            }
        print(f"  {len(out)} gènes enrichis via MyGene ✓")
        return out
    except Exception as e:
        print(f"  [WARN] MyGene erreur : {e}")
        return {}


# =================================================================
#  MÉTHODE 3 : BioMart (si pybiomart disponible)
# =================================================================
def annotate_via_biomart(ids):
    """Utilise Ensembl BioMart pour annotation complète."""
    print("── Annotation via BioMart ──────────────────────────────────────")
    try:
        server   = Server(host="http://www.ensembl.org")
        dataset  = server.marts["ENSEMBL_MART_ENSEMBL"].datasets["hsapiens_gene_ensembl"]
        result   = dataset.query(
            attributes=[
                "ensembl_gene_id", "hgnc_symbol", "description",
                "gene_biotype", "chromosome_name",
                "start_position", "end_position",
                "hgnc_id", "entrezgene_id",
            ],
            filters={"ensembl_gene_id": ids},
        )
        result = result.rename(columns={
            "Gene stable ID":    "Ensembl_ID",
            "HGNC symbol":       "Gene_Symbol",
            "Gene description":  "Description",
            "Gene type":         "Biotype",
            "Chromosome/scaffold name": "Chromosome",
            "Gene start (bp)":   "Start",
            "Gene end (bp)":     "End",
            "HGNC ID":           "HGNC_ID",
            "NCBI gene (formerly Entrezgene) ID": "Entrez_ID",
        })
        result = result.drop_duplicates("Ensembl_ID").set_index("Ensembl_ID")
        out = result.to_dict(orient="index")
        print(f"  {len(out)} gènes annotés via BioMart ✓")
        return out
    except Exception as e:
        print(f"  [WARN] BioMart erreur : {e}")
        return {}


# =================================================================
#  ANNOTATION PRINCIPALE
# =================================================================

# Méthode primaire : Ensembl REST (toujours disponible)
annot_ensembl = annotate_via_ensembl_rest(ensembl_ids_clean)

# Méthode complémentaire : mygene si disponible
annot_mygene = {}
if MYGENE_AVAILABLE:
    annot_mygene = annotate_via_mygene(ensembl_ids_clean)
else:
    print("  [INFO] mygene non installé → pip install mygene")
    print("         Annotation Ensembl REST utilisée seule.\n")

# Méthode complémentaire : BioMart si disponible
annot_biomart = {}
if BIOMART_AVAILABLE:
    annot_biomart = annotate_via_biomart(ensembl_ids_clean)
else:
    print("  [INFO] pybiomart non installé → pip install pybiomart")
    print("         BioMart ignoré.\n")


# =================================================================
#  FUSION DES ANNOTATIONS
# =================================================================
print("\n── Fusion des annotations ───────────────────────────────────────")

rows = []
for orig_id, clean_id in zip(ensembl_ids, ensembl_ids_clean):
    e  = annot_ensembl.get(clean_id, {})
    mg = annot_mygene.get(clean_id, {})
    bm = annot_biomart.get(clean_id, {})

    # Priorité pour le symbole : BioMart > Ensembl REST > MyGene
    symbol = (bm.get("Gene_Symbol") or e.get("Gene_Symbol") or mg.get("Symbol_MG") or "")
    desc   = (bm.get("Description") or e.get("Description") or mg.get("Name_MG") or "")
    biotype = (bm.get("Biotype") or e.get("Biotype") or "")
    chrom   = (bm.get("Chromosome") or e.get("Chromosome") or "")

    rows.append({
        "Ensembl_ID":   orig_id,
        "Gene_Symbol":  symbol,
        "Description":  desc,
        "Biotype":      biotype,
        "Chromosome":   str(chrom),
        "Start":        bm.get("Start") or e.get("Start") or "",
        "End":          bm.get("End")   or e.get("End")   or "",
        "Entrez_ID":    bm.get("Entrez_ID") or mg.get("Entrez_ID") or "",
        "UniProt_ID":   mg.get("UniProt_ID") or "",
        "Summary":      mg.get("Summary_MG") or "",
    })

annot_df = pd.DataFrame(rows)

# =================================================================
#  FUSION AVEC SCORES BIOMARQUEURS
# =================================================================
merged = biomarkers_df.merge(annot_df, left_on="Gene", right_on="Ensembl_ID", how="left")

# Label d'affichage : symbole si disponible, sinon ID court
merged["Label"] = merged.apply(
    lambda r: r["Gene_Symbol"] if r["Gene_Symbol"] else r["Gene"], axis=1
)

# Direction réglementation (depuis coefficients LASSO)
merged["Direction"] = merged["LASSO_Coef"].apply(
    lambda x: "UP" if x > 0 else ("DOWN" if x < 0 else "ND")
)

# Colonnes finales ordonnées
cols_out = [
    "Ensembl_ID", "Gene_Symbol", "Label", "Description", "Biotype",
    "Chromosome", "Start", "End", "Entrez_ID", "UniProt_ID",
    "Composite_Score", "RF_Importance", "LASSO_Coef", "ElasticNet_Coef",
    "LASSO_selected", "ElasticNet_selected", "Méthodes", "Direction", "Summary"
]
cols_out = [c for c in cols_out if c in merged.columns]
merged_clean = merged[cols_out].sort_values("Composite_Score", ascending=False)

merged_clean.to_csv("resultats/etape6/tables/biomarqueurs_annotes_complets.csv", index=False)
print(f"  ✅ Tableau annoté → resultats/etape6/tables/biomarqueurs_annotes_complets.csv\n")

# Résumé annotation
n_annotated = (merged_clean["Gene_Symbol"] != "").sum()
print(f"  Gènes annotés avec symbole : {n_annotated}/{len(merged_clean)}")
print(f"  Gènes sans symbole         : {len(merged_clean) - n_annotated}\n")


# =================================================================
#  TOP 10 BIOMARQUEURS AFFICHÉS
# =================================================================
print("  ┌─ TOP 10 BIOMARQUEURS ANNOTÉS ───────────────────────────────────")
for i, row in merged_clean.head(10).iterrows():
    sym  = row["Gene_Symbol"] if row["Gene_Symbol"] else "N/A"
    desc = row["Description"][:45] + "…" if len(str(row["Description"])) > 45 else row["Description"]
    direction = "▲" if row.get("Direction") == "UP" else ("▼" if row.get("Direction") == "DOWN" else "—")
    print(f"  │ {i+1:2d}. {sym:<12} {direction}  Score={row['Composite_Score']:.4f}  {desc}")
print("  └──────────────────────────────────────────────────────────────────\n")


# =================================================================
#  FIGURES
# =================================================================

# ── Figure 6.1 : Score composite avec symboles ───────────────────
top20 = merged_clean.head(20).copy()
top20["display"] = top20["Label"]

fig, ax = plt.subplots(figsize=(10, 8), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])

bar_colors = [
    PALETTE["UP"]     if d == "UP"
    else PALETTE["DOWN"]  if d == "DOWN"
    else PALETTE["accent"]
    for d in top20["Direction"]
]

bars = ax.barh(
    top20["display"][::-1],
    top20["Composite_Score"][::-1],
    color=bar_colors[::-1],
    edgecolor="white", linewidth=0.6, height=0.7
)

# Annotation des scores
for bar, score in zip(bars, top20["Composite_Score"][::-1]):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f"{score:.3f}", va="center", ha="left", fontsize=8, color=PALETTE["text"])

ax.set_xlabel("Score composite normalisé (RF + LASSO + ElasticNet)", fontsize=11)
ax.set_title("Top 20 biomarqueurs candidats — Endométriose\n(Score composite multi-méthodes ML)",
             fontsize=13, fontweight="bold", color=PALETTE["text"])
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color=PALETTE["grid"], linestyle="--", alpha=0.7)
ax.set_xlim(0, top20["Composite_Score"].max() * 1.18)

legend_elements = [
    mpatches.Patch(facecolor=PALETTE["UP"],     label="UP-regulated (LASSO coef > 0)"),
    mpatches.Patch(facecolor=PALETTE["DOWN"],   label="DOWN-regulated (LASSO coef < 0)"),
    mpatches.Patch(facecolor=PALETTE["accent"], label="Non déterminé"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.8)
plt.tight_layout()
plt.savefig("resultats/etape6/figures/biomarqueurs_annotes_top20.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Figure → resultats/etape6/figures/biomarqueurs_annotes_top20.png")

# ── Figure 6.2 : Distribution par biotype ────────────────────────
biotype_counts = merged_clean["Biotype"].value_counts()
if len(biotype_counts) > 0:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    biotype_colors = sns.color_palette("Set2", len(biotype_counts))
    bars = ax.bar(biotype_counts.index, biotype_counts.values,
                  color=biotype_colors, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Biotype", fontsize=11)
    ax.set_ylabel("Nombre de gènes", fontsize=11)
    ax.set_title("Distribution des biotypes — Biomarqueurs candidats", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=PALETTE["grid"], linestyle="--", alpha=0.7)
    plt.xticks(rotation=30, ha="right")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig("resultats/etape6/figures/biomarqueurs_biotypes.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Figure → resultats/etape6/figures/biomarqueurs_biotypes.png")

# ── Figure 6.3 : Distribution chromosomique ──────────────────────
chrom_counts = merged_clean["Chromosome"].replace("", np.nan).dropna()
chrom_order  = sorted(chrom_counts.unique(),
                      key=lambda x: int(x) if x.isdigit() else (23 if x == "X" else 24 if x == "Y" else 25))
chrom_val_counts = chrom_counts.value_counts().reindex(chrom_order, fill_value=0)

if len(chrom_val_counts) > 0:
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.bar(chrom_val_counts.index, chrom_val_counts.values,
           color=PALETTE["accent"], edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_xlabel("Chromosome", fontsize=11)
    ax.set_ylabel("Nombre de biomarqueurs", fontsize=11)
    ax.set_title("Localisation chromosomique des biomarqueurs candidats", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=PALETTE["grid"], linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("resultats/etape6/figures/biomarqueurs_chromosomes.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Figure → resultats/etape6/figures/biomarqueurs_chromosomes.png")

# ── Figure 6.4 : Tableau visuel top 15 ───────────────────────────
top15 = merged_clean.head(15)[["Gene_Symbol", "Ensembl_ID", "Composite_Score",
                                "Direction", "Biotype", "Chromosome", "Méthodes"]].copy()
top15["Gene_Symbol"] = top15["Gene_Symbol"].replace("", "—")
top15["Composite_Score"] = top15["Composite_Score"].round(4)
top15.insert(0, "Rang", range(1, len(top15) + 1))

fig, ax = plt.subplots(figsize=(14, 6), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])
ax.axis("off")

col_labels = ["Rang", "Symbole", "Ensembl ID", "Score", "Direction", "Biotype", "Chr.", "Méthodes ML"]
table_data  = top15.values.tolist()

cell_colors = []
dir_col_idx = 4  # colonne Direction
for row in table_data:
    row_c = ["#FFFFFF"] * len(col_labels)
    direction = row[dir_col_idx]
    if direction == "UP":
        row_c[dir_col_idx] = "#FDDEDE"
    elif direction == "DOWN":
        row_c[dir_col_idx] = "#DEE9FD"
    cell_colors.append(row_c)

tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    cellColours=cell_colors,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.6)

for j in range(len(col_labels)):
    tbl[0, j].set_facecolor(PALETTE["accent"])
    tbl[0, j].set_text_props(color="white", fontweight="bold")

ax.set_title("Top 15 biomarqueurs candidats — Endométriose",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("resultats/etape6/figures/table_biomarqueurs_top15.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Figure → resultats/etape6/figures/table_biomarqueurs_top15.png\n")


# =================================================================
#  EXPORT FINAL SIMPLIFIÉ
# =================================================================
final_simple = merged_clean[["Gene_Symbol", "Ensembl_ID", "Composite_Score",
                              "Direction", "Biotype", "Chromosome",
                              "Description", "Méthodes"]].copy()
final_simple = final_simple.rename(columns={"Gene_Symbol": "Symbol"})
final_simple.to_csv("resultats/etape6/tables/biomarqueurs_finaux_annotes.csv", index=False)
print("  Export simplifié → resultats/etape6/tables/biomarqueurs_finaux_annotes.csv")

# Export liste symboles
symbols = merged_clean["Gene_Symbol"].replace("", np.nan).dropna().tolist()
with open("resultats/etape6/tables/liste_symboles_biomarqueurs.txt", "w") as f:
    f.write("\n".join(symbols))
print(f"  Liste symboles  → resultats/etape6/tables/liste_symboles_biomarqueurs.txt")
print(f"  ({len(symbols)} symboles exportés)\n")


# =================================================================
#  RÉSUMÉ
# =================================================================
print("=================================================================")
print("  RÉSUMÉ — ÉTAPE 6 TERMINÉE ✅")
print("=================================================================\n")
print(f"  Biomarqueurs annotés : {len(merged_clean)}")
print(f"  Avec symbole HGNC    : {n_annotated}")
print(f"  UP-regulated         : {(merged_clean['Direction'] == 'UP').sum()}")
print(f"  DOWN-regulated       : {(merged_clean['Direction'] == 'DOWN').sum()}\n")
print("  Fichiers générés dans resultats/etape6/")
print("    tables/ → biomarqueurs_annotes_complets.csv")
print("              biomarqueurs_finaux_annotes.csv")
print("              liste_symboles_biomarqueurs.txt")
print("    figures/ → top20 barplot · biotypes · chromosomes · table visuelle\n")
print("  💡 Pour enrichir davantage :")
print("     pip install mygene    → ajoute Entrez ID, UniProt, résumé")
print("     pip install pybiomart → ajoute HGNC ID via BioMart")

