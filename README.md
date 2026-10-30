# Colon cancer classification from gene expression

Course project in Machine Learning (T-504-ITML) at Reykjavik University, fall 2026.

The task is to classify colon tissue samples as **tumour** or **normal** from the expression levels of 2000 genes. The dataset (Alon et al., 1999) has only 62 samples (40 tumour, 22 normal), so the main focus is getting an **honest performance estimate**: many published results on this data are too optimistic because the genes were selected using all samples, including the test samples.

📄 Full write-up: [report.pdf](report.pdf)

## Method

- **Nested cross-validation.** The outer loop (stratified 5-fold, repeated 3 times, so 15 test folds) estimates performance. The inner loop (stratified 5-fold `GridSearchCV`) picks the hyperparameters and the number of genes.
- **Gene selection inside the pipeline.** `SelectKBest` (ANOVA F-test) with k ∈ {10, 50, 100, 500, all} is refitted on the training data in every fold, so the test samples never influence which genes are chosen.
- **Six models:** logistic regression (L1/L2), linear SVM, RBF SVM, k-nearest neighbours, categorical Naive Bayes, and random forest. They're compared with two baselines: majority class and untuned logistic regression on all genes.
- **Metric:** balanced accuracy, because always predicting "tumour" already gives 64.6% plain accuracy.
- A runtime estimate was made before running the full search.

## Results

Nested CV results, mean ± std over the 15 outer test folds:

| Model | Balanced accuracy | Accuracy | ROC-AUC |
|---|---|---|---|
| Logistic regression | **0.848 ± 0.085** | 0.835 | 0.877 |
| Linear SVM | 0.842 ± 0.105 | 0.855 | 0.884 |
| RBF SVM | 0.828 ± 0.107 | 0.840 | 0.875 |
| Naive Bayes | 0.818 ± 0.109 | 0.822 | 0.901 |
| kNN | 0.805 ± 0.128 | 0.812 | 0.870 |
| Random forest | 0.788 ± 0.087 | 0.808 | 0.910 |
| *Untuned logistic regression* | 0.833 ± 0.114 | 0.851 | 0.882 |
| *Majority class* | 0.500 ± 0.000 | 0.646 | 0.500 |

- Every model clearly beats the majority baseline. The top four are within about 0.03 of each other, which is well inside the fold-to-fold spread.
- Linear models did at least as well as non-linear ones. Naive Bayes and kNN dropped sharply when all 2000 genes were used.
- Logistic regression was chosen as the final model because it scored highest and is the simplest to interpret.
- The ~85% balanced accuracy is close to the ~82.5% that Ambroise & McLachlan (2002) report once the selection bias is removed. That supports their view that the 98% figures in the literature are optimistic.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook colon_cancer_classification.ipynb
```

`colon.mat` is the discretised version of the dataset from the [scikit-feature repository](https://jundongl.github.io/scikit-feature/datasets.html). It must be in the same folder as the notebook. Choose **Run All**. It takes a few minutes, and seeds are fixed (`random_state=42`), so results are reproducible.

The notebook writes `results_summary.csv`, `fold_results.csv`, and the figures used in the report (`fig_models.png`, `fig_genes.png`, `fig_logreg_C.png`).
