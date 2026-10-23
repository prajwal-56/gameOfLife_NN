# Dataset Synthesis, Training & Evaluation

This document details the multi-source pattern synthesis strategy, deduplication guarantees, training methodology, and empirical evaluation metrics.

---

## 1. Multi-Source Pattern Synthesis

A model trained solely on uniform random noise fails when users draw sparse, structured sketches. To bridge this distribution gap, `generate_data.py` combines four distinct generative distributions:

| Generator Source | Sampling Ratio | Description | Purpose |
|---|---|---|---|
| **`uniform_soup`** | 35% | The full $N \times N$ board filled with cell density sampled uniformly from $[0.05, 0.60]$. | Teaches chaotic dynamics, overpopulation collapse, and dense board behavior. |
| **`patch_soup`** | 30% | Random cells confined to a compact bounding box (dimensions $3 \times 3$ to $16 \times 16$) at arbitrary offsets. | Mimics hand-drawn scribbles and localized clusters that decay or stabilize quickly. |
| **`library_perturb`** | 20% | Classic archetypes (gliders, spaceships, pulsars, pentominoes) stamped with $2\%\text{--}15\%$ random cell noise. | Exposes the network to structured automata surrounded by realistic user perturbations. |
| **`combo`** | 15% | Two to three known library patterns stamped near each other at random rotations. | Trains the model on collision dynamics, glider annihilations, and emergent debris. |

---

## 2. Canonical Deduplication

Because the dihedral group $D_4$ preserves physical outcomes, two boards that are rotated or reflected versions of one another produce identical trajectories. Allowing rotated duplicates into the dataset would risk severe data leakage across the train/test split.

### The Canonical Key Algorithm:
For each generated grid $X$, we compute all 8 elements of its dihedral symmetry orbit:

$$\mathcal{O}(X) = \{\text{rot}_{90}^k(X), \text{rot}_{90}^k(\text{flip}(X)) \mid k \in \{0, 1, 2, 3\}\}$$

We select the lexicographically minimal serialized byte sequence as the **canonical representative**:

$$\text{key}(X) = \min_{G \in \mathcal{O}(X)} \text{bytes}(G)$$

A global set of observed canonical keys guarantees that no duplicate pattern or rotated variant enters the dataset.

---

## 3. Training Dynamics & Class Imbalance

### Handling Class Imbalance
In random cellular automata, class distributions are inherently skewed (e.g., `OSCILLATING` and `STABLE` typically dominate over rare `UNSETTLED` patterns).

To prevent the classifier from collapsing to the majority class:
1. **Weighted Sampling**: During training, a `WeightedRandomSampler` selects batches using inverse frequency probabilities:
   $$w_i = \frac{1}{\text{count}(y_i)}$$
2. **Class-Weighted Cross-Entropy**:
   $$\mathcal{L} = - \sum_{c=1}^C w_c \cdot y_c \log(\hat{p}_c)$$

### Learning Rate & Scheduler
- **Optimizer**: Adam with learning rate $\eta = 10^{-3}$ and weight decay $10^{-4}$.
- **Scheduler**: `ReduceLROnPlateau` monitoring validation loss (halving $\eta$ when validation loss stagnates for 5 epochs).
- **Early Stopping**: Halts training if no validation loss improvement occurs over 10 consecutive epochs.

---

## 4. Empirical Evaluation & Baselines

To rigorously assess whether `FateNet` learns meaningful spatial representations rather than exploiting spurious statistical correlations, `evaluate.py` benchmarks against four distinct models:

```text
Baseline Comparison Hierarchy:
1. Majority Class               -> Predicts the most frequent class in train set
2. Hand-Engineered Classifier   -> Multinomial Logistic Regression on:
                                   • Total Population
                                   • Spatial Density
                                   • Bounding Box Dimensions (Width, Height, Area)
                                   • Connected Component Cluster Count
3. Fully-Connected Dense Net    -> MLP on flattened raw grid (ignoring 2D topology)
4. FateNet (CNN)                -> Deep circular-convolutional neural network
```

### Calibration & Expected Calibration Error (ECE)
Because the application presents live confidence probability bars to the user, probabilities must be well-calibrated (i.e., a predicted confidence of $80\%$ should be correct approximately $80\%$ of the time).

ECE is calculated by binning predictions into $M = 10$ confidence intervals:

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

Low ECE indicates that the model's confidence scores accurately reflect true empirical risk.
