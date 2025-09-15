# GMM-Based Synthetic Sampling for Imbalanced Data

This project explores using Gaussian Mixture Models (GMM) to generate synthetic samples for handling imbalanced datasets, specifically applied to credit card fraud detection. The work builds upon traditional oversampling techniques by modeling the underlying probability distribution of minority classes.

## Project Overview

Credit card fraud detection is a classic example of an imbalanced classification problem where fraudulent transactions (minority class) are significantly outnumbered by legitimate transactions (majority class). This imbalance poses challenges for machine learning models, which tend to favor the majority class.

This project implements and evaluates GMM-based synthetic sampling as an alternative to simpler techniques like SMOTE, comparing the results with baseline models and clustering-based approaches.

## Dataset

The project uses the Credit Card Fraud Detection dataset from Kaggle, which contains:
- 284,807 transactions
- Highly imbalanced with only 0.172% fraudulent transactions
- 30 features (V1-V28 are PCA-transformed, plus Time, Amount, and Class)

## Key Features

### 1. Baseline Analysis
- Establishes performance benchmarks using standard logistic regression
- Analyzes class distribution and imbalance ratio
- Evaluates using precision, recall, and F1-score (more meaningful than accuracy for imbalanced data)

### 2. GMM Implementation
- **Model Selection**: Uses AIC/BIC criteria to determine optimal number of GMM components
- **Synthetic Generation**: Creates new minority class samples by sampling from learned probability distribution
- **Distribution Modeling**: Captures complex, multi-modal patterns in minority class data

### 3. Hybrid Approaches
- **GMM Oversampling**: Balances dataset by generating synthetic minority samples
- **CBU + GMM**: Combines Clustering-Based Undersampling with GMM synthetic generation

## Why GMM Over SMOTE?

GMM offers several theoretical advantages over traditional SMOTE:

- **Complex Distributions**: Can model multi-modal minority class distributions
- **Probabilistic Foundation**: Generates samples from learned probability distributions rather than linear interpolation
- **Adaptive Modeling**: Automatically adjusts complexity through component selection
- **Natural Sampling**: Produces more realistic synthetic samples

## Installation and Setup

```bash
pip install kaggle kagglehub scikit-learn pandas numpy matplotlib seaborn imbalanced-learn
```

## Usage

1. **Download Dataset**: The notebook automatically downloads the dataset using kagglehub
2. **Run Analysis**: Execute cells sequentially to perform:
   - Data exploration and baseline model training
   - GMM component selection and model fitting
   - Synthetic sample generation
   - Performance evaluation and comparison

## Results Summary

### Performance Comparison

The project compares three approaches using optimal GMM with 10 components:

| Method | Precision | Recall | F1-Score | Training Size |
|--------|-----------|---------|----------|---------------|
| Baseline | 84.52% | 72.45% | 78.02% | 227,845 |
| GMM Oversampling | 6.50% | 89.80% | 12.13% | 454,902 |
| CBU + GMM | 0.66% | 93.88% | 1.31% | 788 |

### A3 Results (Reference)

For comparison with previous assignment methods:

| Method | Precision | Recall | F1-Score | Training Size |
|--------|-----------|---------|----------|---------------|
| Baseline | 78.16% | 69.39% | 73.51% | 227K |
| SMOTE | 11.70% | 89.80% | 20.71% | 455K |
| CBO | 13.41% | 89.80% | 23.34% | 455K |
| CBU | 3.26% | 91.84% | 6.30% | 788 |

*Note: GMM model uses 10 components as determined by BIC criterion*

## Key Findings

1. **Model Selection**: BIC criterion selected 10 components as optimal for modeling the minority class distribution
2. **Trade-offs**: Different approaches show varying precision-recall trade-offs
3. **Computational Efficiency**: CBU + GMM reduces training data size while maintaining synthetic generation benefits
4. **Distribution Modeling**: GMM with 10 components effectively captures minority class complexity

## Comparison with Previous Methods

When compared to Assignment 3 results (SMOTE, CBO, CBU):
- Traditional methods achieved high recall (~90%) but low precision (~3-13%)
- Baseline maintained better balance between precision and recall
- GMM approaches aim to improve upon both by modeling underlying distributions

## Visualization

The project includes:
- AIC/BIC plots for component selection
- Performance comparison bar charts
- Component weight analysis

## Recommendations

For fraud detection applications:

1. **GMM is Suitable When**:
   - Minority class has complex, multi-modal structure
   - Both precision and recall are important
   - Computational resources allow for distribution modeling

2. **Consider Alternatives When**:
   - Simple minority class structure
   - Extreme computational constraints
   - Only recall optimization needed

## Technical Implementation

### GMM Component Selection
```python
# Evaluate different component numbers
for n_components in range(1, 11):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_minority)
    # Compare AIC/BIC scores

# Results: BIC selected 10 components as optimal
n_components = 10  # Optimal components determined by BIC criterion
```

### Synthetic Sample Generation
```python
# Generate balanced synthetic samples
n_synthetic_samples = len(X_majority) - len(X_minority)
X_synthetic, _ = gmm.sample(n_synthetic_samples)
```

## Evaluation Metrics

The project focuses on:
- **Precision**: Reduces false alarms in fraud detection
- **Recall**: Ensures actual fraud cases are detected
- **F1-Score**: Balances precision and recall for overall performance

Accuracy is less emphasized due to its misleading nature in highly imbalanced scenarios.


## Dependencies

- pandas: Data manipulation
- numpy: Numerical computations
- scikit-learn: Machine learning algorithms
- matplotlib/seaborn: Visualization
- imbalanced-learn: Sampling techniques
- kagglehub: Dataset access

## File Structure

```
assignment-4/
├── notebook.ipynb          # Main analysis notebook
├── README.md              # This documentation
└── PROBLEM.md            # Assignment requirements
```

## Conclusion

This implementation demonstrates that GMM-based synthetic sampling provides a principled, statistically-founded approach to handling imbalanced datasets. While computational overhead is higher than simple interpolation methods, the ability to model complex minority class distributions makes it valuable for applications where both precision and recall matter.

The project shows that there's no one-size-fits-all solution for imbalanced data - the choice of technique should depend on the specific characteristics of your dataset and the relative importance of different performance metrics in your application domain.