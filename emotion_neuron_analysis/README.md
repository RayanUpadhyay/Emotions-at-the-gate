# Emotion Neuron Activation Analysis

**Investigating emotion-specific neuron clusters in Qwen2.5-1.5B-Instruct**

This pipeline probes the internal representations of a large language model to answer four research questions about how LLMs encode emotions at the neuron level.

## Research Questions

| RQ | Question | Module |
|----|----------|--------|
| RQ1 | When an LLM tries to understand an emotion, is there a specific set of neurons that get activated? | `rq1_neuron_frequency.py` |
| RQ2 | Is that specific set the same for specific types of primary emotions? | `rq2_emotion_specificity.py` |
| RQ3 | Can we find out all such clusters for all types of primary emotions? | `rq3_cluster_enumeration.py` |
| RQ4 | Is there any overlap in such clusters? | `rq4_overlap_analysis.py` |

## Setup

```bash
pip install -r requirements.txt
```

**Hardware**: Tested on Windows with NVIDIA RTX 2080 Ti (11 GB VRAM). Also supports CUDA GPUs, Apple MPS, and CPU.

**Model**: Qwen/Qwen2.5-1.5B-Instruct (~3 GB in bfloat16). Downloaded automatically on first run.

## Usage

### Full pipeline (recommended first run)
```bash
python main.py --csv /path/to/emotion_dataset_translated.csv
```

### Skip extraction (use cached activations)
```bash
python main.py --csv /path/to/data.csv --skip-extraction
```

### Run specific RQs only
```bash
python main.py --csv /path/to/data.csv --skip-extraction --rq 1 3
```

### Extraction only (no analysis)
```bash
python main.py --csv /path/to/data.csv --skip-analysis
```

### Custom settings
```bash
python main.py --csv /path/to/data.csv --batch-size 16 --top-k-pct 3.0
```

## Project Structure

```
emotion_neuron_analysis/
├── main.py                      # Orchestrator with CLI
├── config.py                    # All constants and hyperparameters
├── utils.py                     # Logging, device selection, plot helpers
├── preprocessing.py             # CSV loading, deduplication, validation
├── extraction.py                # Model loading, hooks, activation caching
├── rq1_neuron_frequency.py      # Neuron frequency, chi-square, permutation
├── rq2_emotion_specificity.py   # Signature neurons, split-half, classifier
├── rq3_cluster_enumeration.py   # UMAP/t-SNE/PCA, silhouette, enumeration
├── rq4_overlap_analysis.py      # Jaccard, UpSet, dendrogram, shared/exclusive
├── sensitivity.py               # Threshold sensitivity (1%, 3%, 5%, 10%)
├── requirements.txt
└── README.md
```

## Output Structure

```
<output_dir>/
├── analysis.log                          # Full log (also printed to terminal)
├── cache/
│   ├── ffn_layer_{14..22}.npy            # Cached FFN activations (float16)
│   ├── residual_layer_{14..22}.npy       # Cached residual activations (float16)
│   ├── emotions.npy                      # Emotion labels array
│   └── metadata.json                     # Cache bookkeeping
└── results/
    ├── rq1_neuron_frequency/
    │   ├── freq_heatmap_top50_layer*.png
    │   ├── consistent_neurons_per_emotion.png
    │   ├── volcano_*_layer*.png
    │   ├── permutation_*_layer*.png
    │   ├── chi2_significant_neurons.png
    │   └── rq1_summary.txt
    ├── rq2_emotion_specificity/
    │   ├── signature_neuron_counts.png
    │   ├── split_half_jaccard.png
    │   ├── classifier_confusion_matrix.png
    │   ├── classifier_accuracy_per_emotion.png
    │   └── rq2_summary.txt
    ├── rq3_cluster_enumeration/
    │   ├── cluster_pca.png
    │   ├── cluster_tsne.png
    │   ├── cluster_umap.png
    │   ├── silhouette_scores.png
    │   ├── purity_per_emotion.png
    │   ├── signature_enumeration.csv
    │   └── rq3_summary.txt
    ├── rq4_overlap_analysis/
    │   ├── jaccard_heatmap.png
    │   ├── jaccard_heatmap_per_layer.png
    │   ├── shared_exclusive_bar.png
    │   ├── emotion_dendrogram.png
    │   ├── upset_intersections.png
    │   ├── overlap_matrix.csv
    │   └── rq4_summary.txt
    └── sensitivity_analysis/
        ├── consistent_vs_threshold.png
        ├── accuracy_vs_threshold.png
        ├── jaccard_vs_threshold.png
        └── sensitivity_summary.txt
```

## Disk Space Estimate

- **Cache** (raw activations): ~3 GB for ~16K samples (9 layers × FFN + residual)
- **Results** (plots + summaries): ~100 MB
- **Model** (downloaded once): ~3 GB

## Method Summary

1. **Preprocessing**: Load bilingual emotion dataset, deduplicate, validate against Ekman's 6 primary emotions.
2. **Extraction**: Run each text through Qwen2.5-1.5B-Instruct. Forward hooks capture post-SwiGLU FFN gate activations (8960-dim) and residual stream hidden states (1536-dim) from layers 14–22. Mean-pool across token positions. Cache as float16 `.npy` files.
3. **RQ1**: Compute per-neuron activation frequency per emotion. Chi-square tests for neuron-emotion association. Permutation tests for statistical significance of consistent neuron counts.
4. **RQ2**: Identify signature neurons via one-sided z-test (emotion vs all-others). Split-half reliability via Jaccard similarity. Logistic regression classifier on binary masks for discriminability.
5. **RQ3**: UMAP/t-SNE/PCA on residual stream vectors. Silhouette analysis. k-NN purity. Complete signature neuron enumeration table.
6. **RQ4**: Pairwise Jaccard similarity of aggregated signature sets. Exclusive vs shared neuron counts. Hierarchical clustering dendrogram. UpSet intersection plot.
7. **Sensitivity**: All key metrics re-computed at 1%, 3%, 5%, 10% thresholds.

## Citation

Based on Paul Ekman's six primary emotions: Anger, Disgust, Fear, Happiness, Sadness, Surprise.
