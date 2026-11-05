# Label-overlap (achievable-accuracy) analysis — seed-42 split, standardised feature space

For every **malware** test dump, the nearest training dump was found (Euclidean distance after the train-only signed-log standardisation, 52 indicators).

| NN distance threshold | malware test dumps with a near-twin in train | …of which category differs | …of which family differs |
|---|---|---|---|
| < 0.05 | 46.9 % | 32.6 % | 60.8 % |
| < 0.10 | 61.3 % | 33.2 % | 61.1 % |
| < 0.25 | 75.7 % | 35.6 % | 63.1 % |
| < 0.50 | 86.2 % | 36.0 % | 63.2 % |

The median nearest-neighbour distance is 0.056 for malware test dumps and 0.008 for benign ones.

**Interpretation.** Almost half of the malware test dumps have a practically identical memory summary in the training set (distance < 0.05 over 52 standardised indicators). For one third of those twins the twin belongs to a **different category**, and for 61 % to a **different family**. No classifier operating on these 52 Volatility indicators can separate such pairs. This label overlap — not model capacity — caps leak-free accuracy at ≈ 88 % (4-class) and ≈ 76–77 % (16-class). Results far above these values on CIC-MalMem-2022 should be checked for duplicate leakage or for oversampling before the split.

Reproduce: `python code/ceiling_analysis.py` (writes `results/ceiling_analysis.json`).
