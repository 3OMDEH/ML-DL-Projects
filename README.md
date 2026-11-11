# Machine Learning, Deep Learning & NLP Projects

A collection of practical Jupyter notebooks exploring structured data, image classification, and natural language processing. Each project documents the path from data exploration and preprocessing to model training and evaluation, with comments and learning notes explaining the implementation.

The repository combines library-based workflows with selected implementations from scratch, including mini-batch gradient descent for logistic regression.

## Project Overview

| Notebook | Purpose | Models | Main Techniques |
| --- | --- | --- | --- |
| [Airline Customer Satisfaction](AirlinesProject.ipynb) | Classify passenger satisfaction from travel details and service ratings. | Logistic Regression, KNN, Decision Tree, SVM, Random Forest, majority baseline; manual Logistic Regression learning implementation. | Mixed-type preprocessing, imputation, one-hot encoding, scaling, stratified splits, development-set model comparison. |
| [Air Quality Estimation](AirQuality.ipynb) | Estimate same-hour carbon monoxide concentration from sensor and weather measurements. | Linear Regression, Ridge Regression, mean baseline. | Missing-value sentinel handling, chronological splits, imputation, scaling, residual analysis. |
| [Chest X-ray Classification](CNN_Chest_Xray.ipynb) | Classify chest X-ray images as NORMAL or PNEUMONIA. | Custom PyTorch Convolutional Neural Network. | Grayscale preprocessing, mild augmentation, adaptive pooling, validation checkpoint selection, duplicate-image checks. |
| [Diabetes Prediction](Diabetes_Prediction.ipynb) | Classify the diabetes outcome recorded in the Pima Indians Diabetes dataset. | KNN, Logistic Regression, majority baseline. | Zero-coded missing-value handling, stratified cross-validation, grid search, class weighting, optional SMOTE. |
| [Credit Card Fraud Detection](FraudDetection.ipynb) | Identify fraudulent transactions under severe class imbalance. | Logistic Regression, weighted Logistic Regression, prior baseline; optional SMOTE variant. | Robust scaling, chronological splits, development-set threshold selection, precision-recall analysis, alert-volume reporting. |
| [Twitter Sentiment Analysis](TwitterSentimentAnalaysis.ipynb) | Classify tweets as negative or positive. | Logistic Regression, majority baseline. | Sparse TF-IDF, word unigrams/bigrams, duplicate-text handling, stratified splits, complete pipeline persistence. |
| [Titanic Survival Classification](TitanicProblem.ipynb) | Explore passenger survival classification through an earlier modeling exercise. | KNN, Decision Tree, Gaussian Naive Bayes, SVM, Random Forest. | Exploratory visualization, imputation, IQR filtering, one-hot encoding, SMOTE, standardization. |
| [Pandas Fundamentals](Pandas.ipynb) | Build familiarity with data manipulation before model development. | No predictive model. | Series, DataFrames, indexing, CSV loading, missing data, grouping, function application. |

## Project Details

### Airline Customer Satisfaction

**Objective:** Predict whether a passenger is satisfied using demographic information, travel characteristics, service ratings, and delays.

- Uses `ColumnTransformer` and `Pipeline` to combine numerical and categorical preprocessing.
- Applies median imputation to numeric features and most-frequent imputation to categorical features.
- Uses one-hot encoding with unknown-category handling and standard scaling for distance-based and linear models; tree pipelines skip scaling.
- Creates stratified 80/10/10 training, development, and test splits.
- Compares library models using development macro F1, then refits the selected pipeline on training and development data for final testing.
- Includes an educational NumPy implementation of Logistic Regression with mini-batch gradient descent, L2 regularization, and loss curves.

**Evaluation:** Accuracy, class-specific precision/recall/F1, and a confusion matrix.

**Scope:** The feature set includes information collected during or after a flight. It supports post-flight satisfaction estimation; pre-flight prediction would require a different feature set.

### Air Quality Estimation

**Objective:** Estimate `CO(GT)` concentration in mg/m? from five sensor responses, temperature, relative humidity, and absolute humidity.

- Reads the UCI CSV using its semicolon delimiter and comma decimal format.
- Converts the `-200` missing-value marker to missing values, removes unlabeled rows, and retains timestamps for alignment.
- Excludes other reference-analyzer pollutant measurements from the feature set.
- Uses chronological 60/20/20 training, development, and test splits.
- Combines median imputation, missingness indicators, standardization, and regression in complete pipelines.
- Selects between the mean baseline, Linear Regression, and Ridge Regression using development RMSE.

**Evaluation:** MAE, RMSE, R?, actual-versus-predicted plots, and residuals over time. Negative predictions are reported for inspection.

**Scope:** This is same-hour concentration estimation, not future pollution forecasting.

### Chest X-ray Classification

**Objective:** Learn a binary NORMAL/PNEUMONIA image classifier with a compact custom CNN.

- Loads the provided training, validation, and test folders through `ImageFolder` and `DataLoader`.
- Resizes images to 256 ? 256, converts them to grayscale, and normalizes pixel values.
- Applies mild random rotation only to training images; validation/test transforms are deterministic.
- Uses three convolution blocks with Batch Normalization, ReLU, and max pooling, followed by adaptive average pooling and a compact dense classifier.
- Trains with cross-entropy loss, Adam, weight decay, and dropout.
- Restores the checkpoint with the lowest validation loss before the separate final test evaluation.
- Checks class mappings and hashes decoded images to detect identical images across splits.

**Evaluation:** Accuracy, precision/recall/F1, sensitivity, specificity, ROC AUC, average precision, and a confusion matrix.

**Limitation:** The provided validation folder contains only 16 images in the reviewed dataset. Duplicate-image checks do not establish patient independence. A larger validation set with verified patient separation is needed for stronger conclusions.

### Diabetes Prediction

**Objective:** Predict the binary `Outcome` label from eight structured features.

- Treats zeros in Glucose, BloodPressure, SkinThickness, Insulin, and BMI as missing while preserving valid zero pregnancy counts.
- Places zero replacement, median imputation, scaling, and classification inside a pipeline.
- Holds out a stratified 20% test set and uses five-fold stratified cross-validation on the remaining training data.
- Uses `GridSearchCV` to compare KNN neighbor counts and weighting schemes, Logistic Regression regularization and class weights, and a majority baseline.
- Selects the pipeline using mean cross-validation macro F1.
- Adds a KNN-with-SMOTE comparison when `imbalanced-learn` is installed; resampling stays inside training folds.

**Evaluation:** Accuracy, precision/recall/F1, sensitivity, specificity, ROC AUC, average precision, and a confusion matrix.

**Scope:** A small benchmark classification exercise; performance does not establish diagnostic usefulness or generalization to other populations.

### Credit Card Fraud Detection

**Objective:** Rank transactions by fraud likelihood and translate scores into an explicit alert decision.

- Uses chronological training, development, and test periods, keeping equal transaction timestamps within the same split.
- Applies training-fitted robust scaling to Time and Amount while retaining the supplied V1?V28 PCA features.
- Compares ordinary and class-weighted Logistic Regression against a prior baseline using development average precision.
- Offers a separate, optional SMOTE pipeline through `USE_SMOTE = True`; it is disabled by default.
- Chooses an illustrative F1-maximizing threshold on development data and keeps the fitted model fixed for final testing.
- Retains exact-record deduplication as an explicit benchmark choice; real transaction data requires an audit using transaction identifiers.

**Evaluation:** Average precision, ROC AUC, log loss, fraud precision/recall/F1, false-positive rate, frauds found/missed, false alerts, and alerts per 1,000 transactions.

**Scope:** The threshold demonstrates the precision-recall tradeoff. An operational fraud system should set its policy using review capacity or error costs and evaluate score calibration separately.

### Twitter Sentiment Analysis

**Objective:** Classify Sentiment140 tweets as negative (`0`) or positive (`1`, mapped from the dataset label `4`).

- Checks missing text, repeated tweet IDs, repeated normalized text, and conflicting labels before splitting.
- Uses stratified 80/10/10 training, development, and test splits with disjoint normalized-text keys.
- Keeps negation and contractions instead of applying blanket stopword removal or stemming.
- Extracts sparse TF-IDF word unigrams and bigrams, with sublinear term frequency, `min_df=2`, and a 100,000-feature cap.
- Fits TF-IDF and Logistic Regression in one raw-text pipeline and compares development performance against a majority baseline.
- Saves the complete pipeline, label mapping, and version metadata with `joblib`, then demonstrates reloading and predicting from raw tweets.

**Evaluation:** Accuracy, macro F1, class-specific precision/recall/F1, log loss, ROC AUC, and a confusion matrix.

**Scope:** The random split measures performance on similar tweets. It does not guarantee performance on unseen authors, future language, sarcasm, or neutral sentiment. Full-corpus training can require substantial memory and time.

### Additional Learning Notebooks

**[Pandas Fundamentals](Pandas.ipynb)** covers Series and DataFrame creation, label-based and positional indexing, boolean filtering, column manipulation, descriptive statistics, missing values, grouping, and function application. Most examples use inline data; the CSV example references Colab's California housing sample.

**[Titanic Survival Classification](TitanicProblem.ipynb)** records an earlier exploration of several classification algorithms. It includes passenger-data visualization, mean imputation, IQR-based age filtering, categorical encoding, oversampling, and model comparison. This notebook still requires revision: preprocessing and SMOTE occur before an explicit train/test split, training cells reference split variables without defining them, and the provenance of `tested.csv` needs verification. Its existing scores should not be presented as validated benchmark results.

## Getting Started

### 1. Set up a Python environment

From the repository directory:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the common notebook dependencies:

```bash
python -m pip install notebook numpy pandas matplotlib seaborn scikit-learn joblib kagglehub
```

Additional dependencies depend on the project:

```bash
# Optional SMOTE comparisons; also imported by the older Titanic notebook
python -m pip install imbalanced-learn

# Chest X-ray notebook
python -m pip install torch torchvision pillow
```

Use compatible PyTorch/torchvision builds for your hardware. The CNN selects CUDA when available and otherwise runs on CPU. These commands are a starting environment setup, not a pinned reproducibility specification.

### 2. Prepare the dataset

| Notebook | Dataset / Source | Expected Input |
| --- | --- | --- |
| Airlines | [Airlines Customer Satisfaction](https://www.kaggle.com/datasets/sjleshrac/airlines-customer-satisfaction) | Downloaded through `kagglehub`; the loader expects one CSV. |
| Air Quality | [UCI Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality) | Place `AirQualityUCI.csv` beside the notebook or update `DATA_PATH`. |
| Chest X-ray | [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | Downloaded through `kagglehub`; expects `train`, `val`, and `test` image folders. |
| Diabetes | [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | Local `diabetes.csv`, or download through `kagglehub`. |
| Fraud Detection | [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | Local `creditcard.csv`, or download through `kagglehub`. |
| Twitter Sentiment | [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) | Local `training.1600000.processed.noemoticon.csv`, or download through `kagglehub`. |
| Titanic | User-supplied passenger data; source needs verification. | `tested.csv`, including a trustworthy `Survived` label. |
| Pandas | Inline examples and Colab sample data. | Update `/content/sample_data/california_housing_train.csv` when running outside Colab. |

Dataset downloads require network access and may require Kaggle authentication. Consult each provider's terms before using or redistributing its data. Changing a filename alone does not make an unrelated dataset compatible with a notebook's expected schema.

### 3. Run a notebook

```bash
jupyter notebook
```

Open a notebook, select the environment's Python kernel, and run its executable cells from top to bottom. Most notebooks also work in Google Colab after installing dependencies and adjusting local paths. Review the Titanic limitations before attempting a complete run.

Several reviewed notebooks preserve original implementations in a reference appendix. Those examples document the learning process; the executable workflow above the appendix contains the revised implementation.

## Evaluation and Reproducibility

- Fit learned preprocessing on training data only. Use development data or training-only cross-validation to choose models and settings.
- Reserve final test evaluation until modeling decisions are complete. Previously inspected benchmark test sets are not fresh external evidence.
- Interpret metrics in context: regression uses MAE/RMSE, sentiment and general classification include macro F1, and rare-event detection emphasizes precision-recall behavior and false alerts.
- Random seeds support repeatability, but library versions, hardware, and GPU operations can affect results. The Twitter artifact records library versions; the repository does not currently provide a complete environment lockfile.
- Model settings and notebook code are documented here without claiming a new full-dataset benchmark run for every project. Earlier reviews used synthetic workflow checks for several notebooks and a real-data run for Air Quality; readers should rerun their chosen project before reporting results.

## Generated Outputs

Running notebooks can create local files, including:

- `data_splits/` from the airline notebook.
- `twitter_sentiment_pipeline.joblib` from the sentiment notebook, containing its fitted text pipeline and metadata.

Large datasets, local environments, and generated artifacts should be excluded from source-control commits unless intentionally distributed. Only load serialized model files from trusted sources and use a compatible environment.

## Repository Scope

The primary portfolio consists of the eight notebooks above. The repository also contains an exploratory [RAG script](RAG%20System.py), which is outside this notebook overview.

These projects document practical learning in machine learning, deep learning, and NLP. They are educational experiments, with project-specific limitations made explicit rather than presented as production-ready systems.
