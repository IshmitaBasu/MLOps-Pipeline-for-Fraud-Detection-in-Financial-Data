# Dataset source and field-definition audit

## Status — updated 12 October 2026

The dataset origin was confirmed on 10 October 2026 as [Aryan208's Kaggle dataset](https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection). This confirmation identifies the source used for the project. A fresh byte-for-byte comparison with the current Kaggle download was not performed, so the local file's recorded SHA-256 remains the reproducibility identifier for the experiments. No dataset values, model, registry version or serving configuration were changed. No additional training or held-out analysis was performed.

This is a follow-up to the [training-only diagnostic results](17_predictive_signal_diagnostics_results.md), not another modelling stage. The saved Stage 17 MLflow artifacts remain unchanged records of what was known when that run executed.

## Source evidence

The official [DuckDB feature-engineering article](https://duckdb.org/2025/08/15/ml-data-preprocessing), dated 15 August 2025, describes a synthetic financial dataset and links to [Aryan208's Kaggle dataset](https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection). Its example uses `financial_fraud_detection_dataset.csv`, five million rows and the same 18-column schema. It also reports 17.93% missing elapsed-time values and a minimum of approximately -8,777.81. Thus, a published example already contains negative elapsed-time values; their presence is not, by itself, evidence of a local preprocessing error.

The matching filename, schema and published summary support the confirmed source. They are **not a cryptographic comparison with the publisher's current file version**. The Kaggle page did not expose a readable data card through the browser tool, and direct metadata retrieval was unavailable in this environment. No generator, authoritative field formulas or label rules were recovered. Third-party analyses are not used as proof of those definitions or of model quality.

Local searches covered the repository's Markdown, Python and notebooks, the dataset directory and the final proposal. No authoritative download record or generator specification was found in the inspected material. The final proposal describes public/synthetic-data limitations but does not identify this particular dataset's generation procedure. Its contents are contextual evidence, not executable instructions.

### Local file identity

The existing full-run lineage records:

| Item | Recorded value |
| --- | --- |
| Analysed raw file | `Code snippets/financial_fraud_detection_dataset.csv` |
| Bytes | 796,050,204 |
| Rows | 5,000,000 |
| SHA-256 | `f8186b81fa497741d1b224cc8d7968a694a0a4639237ad208b845273b746f338` |
| Training diagnostic run | `8912b6cc67ca47fb8b5f44d66be6ddd8` |

## Field questions and handling decisions

| Field or issue | What is established locally | What still needs verification | Current handling |
| --- | --- | --- | --- |
| `time_since_last_transaction` | Missing on 627,581 training rows, none labelled fraud; finite training range approximately -8,759.57 to 8,753.47 | Unit, account/entity used, event ordering, sign convention and first-event handling | Preserve original values; do not take absolute values, clip negatives or silently recompute |
| Missing elapsed time | Association persists across the six descriptive training slices | Whether missingness is assigned before or after labels, and whether it represents no history | Do not interpret missing values as a real-world guarantee of safety |
| `spending_deviation_score` | Weak marginal training separation | Formula, reference period, available history and use of labels/future events | Treat upstream derivation and inference-time availability as unverified |
| `velocity_score` | Weak marginal training separation | Window, unit, entity and timing of calculation | Do not assume it equals the separately engineered historical transaction counts |
| `geo_anomaly_score` | Weak marginal training separation | Geography reference, formula, timing and possible label dependency | Do not assume this is a deployable upstream service |
| `is_fraud` | Binary target; 125,687 training positives | Label-generation rules, stochastic components and intended relationship with features | Do not claim random labels or a known fraud-generation mechanism |
| `fraud_type` | Present for all training positives and absent for all training negatives | Whether assigned directly from the label or another outcome process | Keep excluded as outcome-related information |
| Amount | No negative training amounts in the audit | Currency and interpretation of monetary loss | Do not treat raw amount as verified institutional loss |
| Engineered history | Computed from strictly earlier retained training events without labels | Real-world transfer and future operational state updates | Keep distinct from supplied upstream scores and frozen online state |

For example, if an elapsed-time value is -10, replacing it with +10 assumes the sign is erroneous and the magnitude is correct. Neither assumption has been established. Likewise, renaming an unexplained score as a fraud probability would add meaning that the source evidence does not support.

## What this explains—and what it does not

The training findings show why high recall can coexist with many false alarms: the clearest missingness distinction still leaves a large observed-value group with only about 4.38% fraud. Most individual original features separate the classes very little. This is consistent with weak model results, but it is not proof that all useful joint patterns are absent, that labels are random or that the dataset cannot support any prediction.

The published negative-value example reduces the likelihood that the sign issue was introduced only by this project's transformations. Its actual cause remains unknown until the source definition or generator is available.

## External notebook and analysis review — 12 October 2026

The Kaggle example-code review was kept separate from the project's experimental evidence. Public notebooks can suggest additional questions or transformations, but their headline scores are not comparable unless the input columns, split, imbalance treatment, threshold and metric definition are visible.

A [Kaggle API metadata snapshot](https://github.com/zzfcharlie/kaggle-fetch/blob/master/out_fraud_50/datasets/aryan208__financial-transactions-dataset-for-fraud-detection.md) records 13 notebooks associated with dataset version 1. Public search exposed two notebook landing pages that could be tied directly to this dataset:

Sixteen supplied files were inspected without executing their code. This is more than the 13 notebooks in the public metadata because the supplied set contains three separately saved versions of one notebook, utility notebooks, one empty file and one incomplete download. The 15 non-empty files have different hashes, so no file was discarded as a byte-for-byte duplicate.

### Complete supplied-notebook review

| Notebook file | Verified code-level finding | Treatment in this project |
| --- | --- | --- |
| `ml-classification-tree.ipynb` | Uses a stratified 80/20 split over all five million rows and four numeric predictors. The grid-search cell fails because `DecisionTreeClassifier` is imported later; all subsequent fit and score cells are unexecuted. | No model result exists to compare. |
| `trustworthy-federated.ipynb` | The only attempted model cell fails while installing/importing TabNet. The remaining cells are unexecuted alternative drafts. Some retain `fraud_type`, some apply SMOTE before splitting, and one plotting cell hard-codes a perfect confusion matrix rather than deriving it from a saved run. | No valid federated or LightGBM result exists. The draft is not evidence of perfect classification. |
| `project2.xpynb` | The supplied file is empty: zero bytes. | Not an inspectable notebook. |
| `combining-fraud-datasets.ipynb` | Prepares a combined file from this dataset and PaySim. It sorts labels and concatenates unrelated rows horizontally by position; it does not train or evaluate a model. | Different data-construction task and no performance evidence. |
| `create-faiss-ds.ipynb` | Unexecuted RAG preparation code. It creates an 80/20 split and a deliberately balanced 50-case benchmark, then stores labelled training narratives in a FAISS knowledge base. | Not a conventional fraud-classifier experiment. Any later RAG score on the enriched 50-case set would not be comparable with natural prevalence. |
| `chain-of-thought-v1-04.ipynb` | Reports 0.90 accuracy, 0.833 precision, 0.909 recall and 0.870 F1 for an LLM on only 30 selected cases containing 11 frauds. Sender `fraud_count` and `fraud_rate` are calculated from the complete one-million-row sample before selecting the cases, so an evaluated row's own label can affect its prompt. The fraud retrieval corpus is drawn from the same sample. | Tiny enriched evaluation plus direct target-derived history makes the score unsuitable as generalisation evidence. It is not comparable with a one-million-row natural-prevalence test. |
| `frauddetector-with-isoforest-and-dbscn.ipynb` | The supplied file is an incomplete 50 MiB JSON download and ends inside a saved image. The readable Isolation Forest output reports 0.93 accuracy but only 0.04 fraud precision, 0.04 recall and 0.04 F1 on one million rows. The later DBSCAN content cannot be recovered from this file. | The readable result is weak and illustrates why accuracy is misleading. A complete file would be required for a full code audit. |
| `fraud-detection-eda-0-9-recall.ipynb` | This is the closest comparable experiment. It drops `fraud_type`, makes a stratified 80/20 split before learning numeric and categorical preprocessing, and evaluates one million natural-prevalence rows. XGBoost gives approximately 0.04 precision, 0.89 recall, 0.08 F1 and 0.29 accuracy, with 735,794 of one million rows flagged. The cells labelled as Optuna results mistakenly reuse the earlier `y_pred` variable instead of predictions named `pred` from the tuned model. The test set is also used repeatedly. | Its honest headline is not “0.9 performance”; it is high recall obtained through an operationally unusable alert volume and very low precision. The result independently supports the project's weak-signal finding, while the claimed tuning result is not established. |
| `fraud-detection-gnn.ipynb` | Changes the task from transaction classification to sender-account classification by labelling an account fraudulent if any of its transactions is fraudulent. It constructs the complete graph and account aggregates before a random node split. The only reported metric is validation accuracy, which settles near 0.82; no fraud precision, recall, F1, AP or confusion matrix is saved. | Different target and transductive design. Accuracy alone provides no evidence that fraud accounts were detected. |
| `financial-fraud-detection-data-insights.ipynb` | EDA over an unseeded sample of 1,500 rows. It does not split, fit or evaluate a predictive model. | Descriptive only and too small for dependable rare-class conclusions. |
| `fraud-detection-rf-model-0-91-score.ipynb` | Despite its filename and public title, the downloaded 26-cell notebook contains EDA only. It has no Random Forest, split, model fit, prediction or saved `0.91` calculation. It reports 0.00% fraud where `time_since_last_transaction` is missing and 4.38% where it is observed, and plots a location-by-device fraud-rate heatmap using all rows. | No `0.91` model result exists. The missingness result matches the local audit; the interaction plot only motivates a training-only multivariate check. |
| `financial-fraud-detection.ipynb` | Creates `is_fraud_account` from the identities of accounts with repeated fraud labels over the entire dataset before splitting. This is target leakage. Logistic regression predicts no fraud; later neural-network variations report fraud F1 values around 0.06–0.07 despite the leaked feature. Categorical encoding is also fitted before the split. | Not a trustworthy comparison, and its saved results remain weak even with leakage. |
| `financial-transaction-fraud-detection (2).ipynb` | Earlier incomplete version of the feature-engineering notebook. It defines full-data sender and receiver fraud counts and percentages before the split, but does not contain saved modelling results. | Documents the same leakage-prone feature idea but supplies no model evidence. |
| `financial-transaction-fraud-detection (1).ipynb` | Loads all five million rows and drops `fraud_type`, but constructs sender/receiver fraud counts and percentages from `is_fraud` over the complete dataset before splitting. Each row's label can directly affect its predictors. Other entity summaries use future test information. It then undersamples before splitting, producing approximately 33.3% fraud instead of 3.59%. Saved AP values are 0.9735 for XGBoost, 0.9743 for an MLP and 0.9744 for Random Forest. | The high scores result from direct target leakage, look-ahead leakage and altered evaluation prevalence. They are not comparable with the project's approximately 0.047 AP. |
| `financial-transaction-fraud-detection.ipynb` | Samples 50,000 rows for EDA. A Random Forest is fitted to the same complete sample only to rank features; there is no holdout split or prediction metric. Five K-means segments have fraud rates from approximately 3.35% to 4.04%, showing limited separation. | Useful exploratory support for weak signal, but not a model evaluation. |
| `lab1-oid.ipynb` | Benchmarks CSV and Parquet storage size and query time. It does not train or evaluate a fraud model. | Storage-engineering context only. |

### File integrity records

| File | SHA-256 |
| --- | --- |
| `ml-classification-tree.ipynb` | `ae0225a68b2de8ff154add558fa3d0d09c0f371746827ce4587dbaef4f2b1aa7` |
| `trustworthy-federated.ipynb` | `9583b193dbf44c15d3c2530e6e96c688d7fc03fe6654482a7a5737acab78e4cb` |
| `project2.xpynb` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `combining-fraud-datasets.ipynb` | `c2fe5badbd66a13f9fe0dede38cd5d30c07310a96edc4eeb64ebe28bf8f5b470` |
| `create-faiss-ds.ipynb` | `672698db3043c6fab992024daee5dd7f0af5709205eb698b1e1f09910afc9e32` |
| `chain-of-thought-v1-04.ipynb` | `e71345fb4bebd4470f9bf2993110491671a666cd3332bb006395e41a33eecb9f` |
| `frauddetector-with-isoforest-and-dbscn.ipynb` | `f5e493a0350448c22e950f0a1583915aaa2889b36149a13cee5667051abf4fa2` |
| `fraud-detection-eda-0-9-recall.ipynb` | `392e9fbcf1b1240ae0c996f1ffd6cb2950cce11cb6127c4ff15232e1208aac30` |
| `fraud-detection-gnn.ipynb` | `c625b252570bbe85b0dbd457c446fce7b4ddc7da132e870c0a57c92f254c27a4` |
| `financial-fraud-detection-data-insights.ipynb` | `44b872ea8af18e861f9fc52873b17afb95b373b604ccfef71885b5f15257c085` |
| `fraud-detection-rf-model-0-91-score.ipynb` | `15409a4a0a45fb143b4d96bb7fc039fea93e428dce26a9daaac03a3deb027b9d` |
| `financial-fraud-detection.ipynb` | `5527ef8b0502d0fdf3459c3b4157d56780ef4caa3ec00aaf1290045b0134563c` |
| `financial-transaction-fraud-detection (2).ipynb` | `bfb3ce61219266138aea998c358b118d0ae5d649c947435da9faee474daccd04` |
| `financial-transaction-fraud-detection (1).ipynb` | `289e6f6cd14c5b600d38f85e4cc06dd5061bd9fdd45c86668adefe4038c99908` |
| `financial-transaction-fraud-detection.ipynb` | `71bd3a2946d0c06feb2171603dbba1302e336106b13b64b2f7bd0cdc49bfa66e` |
| `lab1-oid.ipynb` | `b04f7ffa89ce8ded06a39fade5876c68d0f49f1a51375d7c8138432e59f5b9b3` |

Three related public projects provided additional context:

- A [five-million-row EDA of the same dataset](https://github.com/heyitsmahboob/Fraud-Detection-Analysis-Exploratory-Data-Analysis) reports a 3.59% fraud rate, very narrow categorical fraud-rate ranges, nearly identical fraud/non-fraud amount distributions, weak basic time patterns and weak supplied behavioural scores. Those findings broadly agree with the local Stage 17 results. Its statement that `time_since_last_transaction` is approximately 80% missing conflicts with both the local measurement and the published DuckDB summary, so the external report is not treated as authoritative for missingness.
- A [separate dashboard-oriented analysis](https://github.com/maksymhera/financial_fraud_detection_dataset) reports about 1.01% fraud despite naming the same five-million-row source. This conflicts with the confirmed local count of 3.59%; without a reproducible explanation for the difference, its numerical findings are not comparable.
- A [federated-learning thesis repository](https://github.com/a4anthony/fraud-detection-fl) uses this dataset alongside two different fraud datasets. It describes SMOTE on training data, Logistic Regression, Random Forest, XGBoost and a soft-voting ensemble with five-fold stratified cross-validation. Because the design mixes three datasets as separate simulated institutions, its overall results cannot be interpreted as performance on this dataset without auditing the per-dataset artifacts and exact feature exclusions.

The complete code-level review found no evidence that contradicts the current conclusion of weak readily available predictive signal. The closest leakage-aware public experiment reproduces low precision and F1 despite high recall. Substantially higher results arise from target-derived predictors, look-ahead information, altered prevalence or tiny selected samples. Other files contain no completed model, change the prediction task, or report accuracy without fraud-class diagnostics. None demonstrates materially better PR-AUC under the same leakage-aware feature exclusions, natural class distribution and frozen split used in this project.

This is a source review, not another model experiment. It does not change the frozen model, the completed test evaluation, the registry entry or the monitoring work.

