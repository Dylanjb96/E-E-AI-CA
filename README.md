# E&E AI CA

# Continuous Assessment 1 - Multi-Label Email Classification

## Project Overview
This project implements the architecture for **Continuous Assessment 1** in **Emerging & Evolving AI**.

The goal is to design a modular email classification system using software architecture concepts and an **Extreme Programming (XP)** style of development.

The architecture is built to support:

- separation of preprocessing and modeling logic
- consistent input data handling across models
- abstraction through a shared model interface
- extension from single-target classification to **multi-label dependent classification**

---

## Current Status
The current implementation provides a working example for:

### Design Choice 1: Chained Multi-Outputs
The system currently classifies the following chained targets:

- **Type 2**
- **Type 2 + Type 3**
- **Type 2 + Type 3 + Type 4**

This is implemented using:

- CSV data loading
- preprocessing and target generation
- TF-IDF vectorization
- a shared `DatasetBundle` object
- an abstract model interface
- evaluation using separate metrics logic
- a smoke test for the pipeline
- a `ChainedMultiOutputModel` strategy
- model selection via `ModelFactory` (`Random Forest`, `Logistic Regression`, `MultinomialNB`)
- evaluation using `metrics.py` and `reporting.py`

---

## Datasets
The project currently uses the datasets provided in the lecturer skeleton/lab materials:

- `Purchasing.csv`
- `AppGallery.csv`

At the moment, the main implementation is run using:

- `Purchasing.csv`
  
`AppGallery.csv` can also be used to demonstrate that the architecture is reusable across datasets with the same structure.

### Dataset Structure
The dataset includes:

- `Interaction content` → email text input
- `Type 1`
- `Type 2`
- `Type 3`
- `Type 4`

### Important Note
`Type 1` is ignored because it contains only one class in each file, so no classification is needed for that column.

---

## Implemented Architecture

### Main Components
- `src/main_controller.py`  
  Controls the full pipeline.

- `src/config.py`  
  Stores shared configuration such as dataset path, column names, test split, and feature settings.

- `src/preprocessing/data_loader.py`  
  Loads the CSV file, cleans the data, handles missing labels, and creates chained targets.

- `src/preprocessing/text_cleaner.py`  
  Contains the `clean_text()` function used to normalize the email text.

- `src/preprocessing/splitter.py`  
  Splits the dataset into train/test sets and returns the chained targets.

- `src/features/vectorizer.py`  
  Converts text into TF-IDF feature vectors.

- `src/data_models/dataset_bundle.py`  
  Encapsulates all train/test inputs and targets in one object.

- `src/models/base_model.py`  
  Provides the abstract model interface.

- `src/models/chained_multioutput_model.py`  
  Implements `BaseModel` by training three chained estimators (t2, t23, t234).

- `src/models/model_factory.py`  
  Selects the base algorithm (RF/LR/NB) and returns the configured chained model.

- `src/models/random_forest_model.py`  
  Random Forest base model option.

- `src/models/logistic_regression_model.py`  
  Logistic Regression base model option.

- `src/models/multinomial_nb_model.py`  
  MultinomialNB base model option.

- `src/evaluation/metrics.py`  
  Computes accuracy for t2, t23, and t234.

- `src/evaluation/reporting.py`  
  Formats and prints the evaluation results.

- `tests/test_pipeline.py`  
  Smoke test for verifying that the pipeline loads and splits correctly.

- `tests/test_model_predictions.py`  
  Tests that model prediction outputs are in the expected format.

- `tests/test_targets.py`  
  Tests that chained targets are created correctly.

---

## Current Folder Structure

```text
E-E-AI-CA/
├─ data/
│  ├─ AppGallery.csv
│  └─ Purchasing.csv
├─ docs/
│  ├─ Design Choice 1.png
│  ├─ Design Choice 2.png
│  └─ EEAI CA1.docx
├─ src/
│  ├─ config.py
│  ├─ main_controller.py
│  ├─ data_models/
│  │  └─ dataset_bundle.py
│  ├─ evaluation/
│  │  ├─ metrics.py
│  │  └─ reporting.py
│  ├─ features/
│  │  └─ vectorizer.py
│  ├─ models/
│  │  ├─ base_model.py
│  │  ├─ chained_multioutput_model.py
│  │  ├─ logistic_regression_model.py
│  │  ├─ model_factory.py
│  │  ├─ multinomial_nb_model.py
│  │  └─ random_forest_model.py
│  └─ preprocessing/
│     ├─ data_loader.py
│     ├─ splitter.py
│     └─ text_cleaner.py
├─ tests/
│  ├─ test_model_predictions.py
│  ├─ test_pipeline.py
│  └─ test_targets.py
├─ .gitignore
├─ README.md
└─ requirements.txt
```
## Command Line Usage / How to Run

Open a terminal in the project root folder:

### Install Dependencies
```bash
pip install -r requirements.txt

### Run the Main Project
python -m src.main_controller

### Run Tests
python -m pytest

### Optional switch dataset
to switch dataset, update DATASET_NAME or DATA_PATH in src/config.py then run again:

python -m src.main_controller
```