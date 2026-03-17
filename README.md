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