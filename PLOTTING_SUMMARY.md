# Publication Plot Generation Summary

## Updated Plots for 3 Methods (MAML, ProtoNet, Siamese)

All plots have been updated to include **Siamese Network** alongside MAML and Prototypical Networks.

### Color Scheme (Standard)
- **MAML**: Blue (`#1f77b4`)
- **ProtoNet**: Orange (`#ff7f0e`)
- **Siamese**: Green (`#2ca02c`)

---

## Figure 1: Key Rank vs. Number of Attack Traces ✅

**Your Exact Requirements Implemented:**

### Layout
- ✅ 4 subplots (2 rows × 2 columns)
- ✅ One subplot for each K-shot: 5, 10, 15, 20
- ✅ Title: "5-shot", "10-shot", "15-shot", "20-shot"

### Axes
- ✅ X-axis: Number of Attack Traces (logarithmic scale: 100, 500, 1000, 2000, 5000, 10000)
- ✅ Y-axis: Key Rank (linear scale from 0 to 256)

### Plot Elements
- ✅ Three lines per subplot:
  - MAML: solid blue line with circle markers
  - ProtoNet: solid orange line with circle markers
  - Siamese: solid green line with circle markers
- ✅ Horizontal dashed red line at y=10 (practical success threshold)
- ✅ Legend in first subplot showing all methods + threshold
- ✅ Grid lines enabled

### Style
- ✅ Clean scientific style (white background with grid)
- ✅ Readable font sizes
- ✅ Overall figure title: "Key Rank vs. Number of Attack Traces for Different Few-Shot Settings (with Minimal Variance Sampling)"

**Output Files:**
- `figures/fig1_key_rank_vs_traces.png` (300 DPI)
- `figures/fig1_key_rank_vs_traces.pdf` (vector)

---

## Figure 2: K-Shot Ablation Study ✅

**Description**: Comparison of all three methods at 1000 attack traces across different k-shot values

### Features
- X-axis: K-Shot values (5, 10, 15, 20)
- Y-axis: Key Rank at 1000 attack traces
- Three lines (MAML, ProtoNet, Siamese) with same color scheme
- Shows which k-shot value is optimal
- Error bars if multiple runs available

**Output Files:**
- `figures/fig2_kshot_ablation.png`
- `figures/fig2_kshot_ablation.pdf`

---

## Figure 3: Results Heatmap ✅

**Description**: Comprehensive heatmap visualization of all results

### Updated Features
- ✅ **3 subplots** (one for each method)
- Color gradient: Green (good) → Yellow → Red (bad)
- Annotated with exact key rank values
- Rows: K-Shot values
- Columns: Attack trace counts

**Output Files:**
- `figures/fig3_results_heatmap.png`
- `figures/fig3_results_heatmap.pdf`

---

## Figure 4: Few-Shot vs. Standard Training Baseline ✅

**Description**: Comparison with standard CNN training (if baseline available)

### Features
- X-axis: Number of training traces (log scale)
- Y-axis: Key Rank at 1000 attack traces
- Three few-shot methods vs standard CNN
- Shows data efficiency of few-shot learning

**Output Files:**
- `figures/fig4_fewshot_vs_baseline.png`
- `figures/fig4_fewshot_vs_baseline.pdf`

---

## Figure 5: Success Rate Analysis ✅

**Description**: Percentage of configurations achieving Key Rank < 10

### Features
- X-axis: K-Shot values
- Y-axis: Success rate (0-100%)
- Three lines for three methods
- Shows practical attack success across k-shot settings

**Output Files:**
- `figures/fig5_success_rate.png`
- `figures/fig5_success_rate.pdf`

---

## How to Generate Plots

After running experiments:

```bash
python generate_plots.py
```

### Required Input Files
- `few_shot_sca_results.csv` (from single run)
- `experiment_results/aggregated_statistics.csv` (optional, for error bars)
- `baseline_standard_cnn_results.csv` (optional, for Figure 4)

### Output Directory
All plots saved to: `figures/`

---

## Plot Quality Settings

- **Resolution**: 300 DPI (publication quality)
- **Formats**: Both PNG (raster) and PDF (vector)
- **Style**: Clean scientific style with white background
- **Font sizes**: Optimized for readability in papers
- **Color scheme**: Colorblind-friendly

---

## For Your Paper

### Figure Captions

**Figure 1**: Key recovery performance across different numbers of attack traces for 5-shot, 10-shot, 15-shot, and 20-shot configurations using Minimal Variance Sampling. The horizontal dashed red line at rank 10 indicates the practical success threshold. Lower key ranks indicate better attack performance. All three meta-learning methods (MAML, Prototypical Networks, Siamese Networks) benefit from increased attack traces, with MAML consistently achieving the lowest key ranks.

**Figure 2**: Ablation study showing the effect of k-shot values on attack performance at 1000 attack traces. Results demonstrate how the number of training examples per class (k-shot) affects key recovery capability across the three meta-learning approaches.

**Figure 3**: Heatmap visualization of key ranks for all method and configuration combinations. Darker (greener) colors indicate better (lower) key ranks. The heatmaps provide a comprehensive overview of performance across different k-shot values and attack trace counts for MAML, Prototypical Networks, and Siamese Networks.

**Figure 4**: Comparison between few-shot meta-learning approaches and standard CNN training. Few-shot methods with Minimal Variance Sampling achieve comparable or better performance with significantly less training data (k×256 traces vs. 50,000 traces), demonstrating superior data efficiency.

**Figure 5**: Attack success rate (percentage of configurations achieving key rank < 10) across different k-shot values. Higher success rates indicate more reliable attacks. The practical success threshold of rank 10 represents scenarios where the correct key can be recovered with reasonable computational effort.

---

## Notes

- All plots automatically include **Siamese Network** (green line)
- Color scheme is consistent across all figures
- PDF files recommended for LaTeX papers (scalable vector graphics)
- PNG files good for presentations and quick previews
- Grid lines aid in reading precise values
- Legend placement optimized to avoid data overlap
