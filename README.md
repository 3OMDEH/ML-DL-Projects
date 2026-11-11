# Machine Learning, Deep Learning & NLP Projects

A collection of practical Jupyter notebooks and a retrieval-augmented generation (RAG) script exploring structured data, image classification, natural language processing, and document question answering. Each project documents the path from data exploration and preprocessing to model training and evaluation, with comments and learning notes explaining the implementation.

The repository combines library-based workflows with selected implementations from scratch, including mini-batch gradient descent for logistic regression.

## Project Overview

| Project | Purpose | Models | Main Techniques |
| --- | --- | --- | --- |
| [Airline Customer Satisfaction](AirlinesProject.ipynb) | Classify passenger satisfaction from travel details and service ratings. | Logistic Regression, KNN, Decision Tree, SVM, Random Forest, majority baseline; manual Logistic Regression learning implementation. | Mixed-type preprocessing, imputation, one-hot encoding, scaling, stratified splits, development-set model comparison. |
| [Air Quality Estimation](AirQuality.ipynb) | Estimate same-hour carbon monoxide concentration from sensor and weather measurements. | Linear Regression, Ridge Regression, mean baseline. | Missing-value sentinel handling, chronological splits, imputation, scaling, residual analysis. |
| [Chest X-ray Classification](CNN_Chest_Xray.ipynb) | Classify chest X-ray images as NORMAL or PNEUMONIA. | Custom PyTorch Convolutional Neural Network. | Grayscale preprocessing, mild augmentation, adaptive pooling, validation checkpoint selection, duplicate-image checks. |
| [Diabetes Prediction](Diabetes_Prediction.ipynb) | Classify the diabetes outcome recorded in the Pima Indians Diabetes dataset. | KNN, Logistic Regression, majority baseline. | Zero-coded missing-value handling, stratified cross-validation, grid search, class weighting, optional SMOTE. |
| [Credit Card Fraud Detection](FraudDetection.ipynb) | Identify fraudulent transactions under severe class imbalance. | Logistic Regression, weighted Logistic Regression, prior baseline; optional SMOTE variant. | Robust scaling, chronological splits, development-set threshold selection, precision-recall analysis, alert-volume reporting. |
| [Twitter Sentiment Analysis](TwitterSentimentAnalaysis.ipynb) | Classify tweets as negative or positive. | Logistic Regression, majority baseline. | Sparse TF-IDF, word unigrams/bigrams, duplicate-text handling, stratified splits, complete pipeline persistence. |
| [Titanic Survival Classification](TitanicProblem.ipynb) | A "Hello World!" project in machine learning domain, explore passenger survival classification through an earlier modeling exercise. | KNN, Decision Tree, Gaussian Naive Bayes, SVM, Random Forest. | Exploratory visualization, imputation, IQR filtering, one-hot encoding, SMOTE, standardization. |
| [Pandas Fundamentals](Pandas.ipynb) | Build familiarity with data manipulation before model development. | No predictive model. | Series, DataFrames, indexing, CSV loading, missing data, grouping, function application. |

| [PDF Question Answering with RAG](RAG%20System.py) | Answer a question using information retrieved from a PDF document. | Llama 3.2 for query generation and answering; `nomic-embed-text-v2-moe` for embeddings, both through Ollama. | PDF loading, overlapping text chunks, persistent Chroma vector storage, multi-query retrieval, context-based prompting, LangChain composition. |

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

**Objective:** Estimate `CO(GT)` concentration in mg/m^3 from five sensor responses, temperature, relative humidity, and absolute humidity.

- Reads the UCI CSV using its semicolon delimiter and comma decimal format.
- Converts the `-200` missing-value marker to missing values, removes unlabeled rows, and retains timestamps for alignment.
- Excludes other reference-analyzer pollutant measurements from the feature set.
- Uses chronological 60/20/20 training, development, and test splits.
- Combines median imputation, missingness indicators, standardization, and regression in complete pipelines.
- Selects between the mean baseline, Linear Regression, and Ridge Regression using development RMSE.

**Evaluation:** MAE, RMSE, R-squared, actual-versus-predicted plots, and residuals over time. Negative predictions are reported for inspection.

**Scope:** This is same-hour concentration estimation, not future pollution forecasting.

### Chest X-ray Classification

**Objective:** Learn a binary NORMAL/PNEUMONIA image classifier with a compact custom CNN.

- Loads the provided training, validation, and test folders through `ImageFolder` and `DataLoader`.
- Resizes images to 256 x 256, converts them to grayscale, and normalizes pixel values.
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
- Applies training-fitted robust scaling to Time and Amount while retaining the supplied V1-V28 PCA features.
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

### PDF Question Answering with RAG

**Objective:** Build a local document question-answering prototype that retrieves relevant PDF passages and supplies them as context to a language model.

**Implementation:** [RAG System.py](RAG%20System.py)

| Component | Implementation | Role |
| --- | --- | --- |
| Document loading | LangChain `PyPDFLoader` | Reads the configured PDF into page-level document objects. |
| Text splitting | `RecursiveCharacterTextSplitter` | Creates chunks with a 1,200-character target size and 300-character overlap. |
| Embedding model | Ollama `nomic-embed-text-v2-moe` | Converts document chunks and retrieval queries into numerical vectors. |
| Vector store | Chroma | Stores vectors and document content in the `pdf_rag` collection under `./chroma_db`. |
| Query expansion | LangChain `MultiQueryRetriever` with Llama 3.2 | Requests five alternative versions of the question to retrieve passages from multiple perspectives. |
| Answer generation | Ollama `llama3.2` | Produces an answer using the retrieved context and a context-only instruction. |
| Chain composition | LangChain runnables and `StrOutputParser` | Connects retrieval, prompt construction, model invocation, and plain-text output. |

**Workflow:** PDF loading -> text splitting -> embedding -> Chroma indexing -> query expansion and retrieval -> context-based answer generation.

The script uses the same chat model for query expansion and answering, configured with `temperature=0.8` and `top_p=0.95`. It runs one example question about security issues discussed in the configured paper and prints the response. This is retrieval and inference with existing models; it does not train or fine-tune a language model.

**Running the script:**

1. Install and start Ollama on the machine running the script.
2. Install the Python packages used by the implementation:

   ```bash
   python -m pip install langchain-community langchain-text-splitters langchain-core langchain-classic langchain-ollama chromadb ollama pypdf
   ```

3. Make the two models available in Ollama:

   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text-v2-moe
   ```

4. Set `doc_path` to an existing, text-readable PDF and replace the question in `chain.invoke(...)` as needed. The current path is specific to the author's machine. The Python dependencies above are not pinned; compatibility with the script's imports must be checked in the selected environment.
5. Run from the repository directory:

   ```bash
   python "RAG System.py"
   ```

The script also calls `ollama.pull` for the embedding model during startup. Initial model downloads require network access. PDF content is processed through the local Ollama service and the Chroma directory; no PDF is bundled with this example.

**Current limitations:** The script is a single-PDF prototype with a hard-coded path and question. The `if doc_path` check tests whether the path string is nonempty, not whether the file exists. It constructs the index on every run rather than explicitly reopening an existing collection, so repeated ingestion needs duplicate-management logic. It does not implement OCR, explicit source citations in the final answer, a retrieval-quality benchmark, or an answer-faithfulness evaluation. A context-only prompt is an instruction, not a guarantee of factual accuracy. Retrieved documents should also be treated as untrusted input when developing the prototype further.

### Additional Learning Notebooks

**[Pandas Fundamentals](Pandas.ipynb)** covers Series and DataFrame creation, label-based and positional indexing, boolean filtering, column manipulation, descriptive statistics, missing values, grouping, and function application. Most examples use inline data; the CSV example references Colab's California housing sample.