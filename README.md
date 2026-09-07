# Crash2FunctionalScenario

Code for extracting autonomous-vehicle test scenarios from in-depth accident data, as used in
**"Accident-Based Freight-Vehicle Scenarios Extraction for Autonomous Driving: A Clustering and Association Rule Approach"**.

The pipeline has two stages:

1. **Clustering** — accident process features (encoded as binary/categorical attributes) are clustered
   with k-medoids (PAM) under a Hamming distance to obtain a set of functional scenarios.
2. **Association rule mining** — Apriori is applied within the accident data to identify the
   environmental and roadway conditions associated with each scenario, turning the functional
   scenarios into environment-conditioned test scenarios.

## Contents

| File | Description |
| --- | --- |
| `k_medoids_clustering.py` | k-medoids (PAM) clustering of accident process features using a Hamming distance matrix, with silhouette and SSE diagnostics and a silhouette plot. |
| `apriori_association_rules.py` | Apriori association rule mining over the accident attributes, filtered by minimum support, confidence and lift. |

## Requirements

- Python 3.8+
- `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `mlxtend`, `openpyxl`

```bash
pip install pandas numpy scipy scikit-learn matplotlib mlxtend openpyxl
```

## Usage

### k-medoids clustering

Input: a **CSV file without a header row**, where the first column is the sample ID and the
remaining columns are the encoded accident process features (one row per accident).

Edit the parameters at the top of the script and run it:

```python
file_path = "path/to/your/features.csv"   # input CSV
n_clusters = 9                            # number of clusters
np.random.seed(22)                        # random seed for the initial medoids
```

```bash
python k_medoids_clustering.py
```

The script prints the cluster label of every sample, the medoid indices, the average silhouette
width (ASW), the sum of squared errors (SSE), and the centre features and sample IDs of every
cluster. It then shows a silhouette plot with the ASW marked as a dashed red line.

To choose `n_clusters`, sweep the value and compare the reported ASW and SSE.

### Apriori association rules

Input: an **XLSX file** with a header row containing an `id` column plus one column per accident
attribute. Every attribute value is treated as an item, so values must be distinct across columns
(e.g. `weather_rain`, `light_night`) to avoid collisions during one-hot encoding.

Set the parameters at the bottom of the script and run it:

```python
input_file = 'path/to/your/accidents.xlsx'
output_file = 'path/to/your/rules.xlsx'
min_support = 0.05
min_confidence = 0.8
min_lift = 1
```

```bash
python apriori_association_rules.py
```

The output XLSX contains the columns `antecedents`, `consequents`, `support`, `confidence` and
`lift`, restricted to the rules that pass all three thresholds.

## Notes

- The clustering script implements PAM directly rather than calling a library, so the medoid
  update is exact but scales as O(n²) in memory for the distance matrix — fine for the corpus
  sizes used here (hundreds to low thousands of accidents).
- Hamming distance is used because the accident process features are categorical/binary; change
  the `metric` argument to `pdist` if your features are continuous.
- Results depend on the random seed for the initial medoids. The seed is fixed (`22`) for
  reproducibility; re-run with several seeds to confirm the solution is stable.

## Related data

The freight corpus consists of 122 in-depth investigations of Chinese crashes involving freight
vehicles. The reference corpus consists of 1,162 U.S. CRSS records. Accident data are not
included in this repository.
