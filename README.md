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
- a `RandomForestModel` implementation
- evaluation using separate metrics logic
- a smoke test for the pipeline

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

- `src/features/vectorizer.py`  
  Converts text into TF-IDF feature vectors.

- `src/data_models/dataset_bundle.py`  
  Encapsulates all train/test inputs and targets in one object.

- `src/models/base_model.py`  
  Provides the abstract model interface.

- `src/models/random_forest_model.py`  
  Implements the model-specific training and prediction logic.

- `src/evaluation/metrics.py`  
  Evaluates predictions and prints the results.

- `tests/test_pipeline.py`  
  Smoke test for verifying that the pipeline loads and splits correctly.

---

## Current Folder Structure

```text
CA/
├─ data/
│  ├─ Purchasing.csv
│  └─ AppGallery.csv
├─ docs/
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ main_controller.py
│  ├─ preprocessing/
│  │  ├─ __init__.py
│  │  └─ data_loader.py
│  ├─ features/
│  │  ├─ __init__.py
│  │  └─ vectorizer.py
│  ├─ data_models/
│  │  ├─ __init__.py
│  │  └─ dataset_bundle.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ base_model.py
│  │  └─ random_forest_model.py
│  └─ evaluation/
│     ├─ __init__.py
│     └─ metrics.py
├─ tests/
│  └─ test_pipeline.py
├─ requirements.txt
└─ README.md

## Command Line Usage / How to Run

Open a terminal in the project root folder, for example:

```bash
cd C:\Users\Dylan B\Desktop\E&EAI\CA

### Install Dependencies
```text
pip install -r requriements.txt

### Run the Main Project
```text
python -m src.main_controller

### Run Tests
```text
python -m pytest

### Optional switch dataset
to switch dataset, update DATASET_NAME or DATA_PATH in src/config.py then run again:
```text
python -m src.main_controller