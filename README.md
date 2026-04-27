# E-Commerce Purchase Prediction

A scalable pipeline for predicting user purchase behavior from large-scale e-commerce interaction logs.

## 🚀 Overview

This project builds a multi-stage data processing and modeling pipeline to predict whether a user will make a purchase based on historical behavior. It transforms raw event logs (~42M rows) into structured feature representations and trains machine learning models for classification.

## 🧠 Key Idea

We bridge the gap between sequential behavioral data and tabular models:

Raw Event Logs → Event-Level Dataset → Feature Engineering → User-Level Dataset → Modeling

- Event-level data preserves full interaction sequences  
- Feature engineering extracts rich behavioral signals  
- User-level data provides model-ready representations  

## 📊 Dataset

We use the REES46 E-commerce Behavioral Dataset (October 2019), which contains ~42 million events and ~3 million users. Each event corresponds to a user interaction, including view, cart, remove_from_cart, and purchase.

For convenience and reproducibility, the dataset is hosted on Hugging Face:  
https://huggingface.co/datasets/skpy/E_Commerce_Behavioral_Analysis

## 🏗️ Pipeline

### 1. Event-Level Processing

We construct a clean event-level dataset by parsing timestamps, handling missing values, extracting hierarchical categories, consolidating rare categories, and sorting events by user and time. This step preserves the full behavioral sequence without aggregation.

### 2. Event Feature Engineering

We build an intermediate feature table (`event_feature_table_v3`) by aggregating and enriching event-level data. This step captures rich behavioral patterns, including:

- Behavioral statistics (e.g., total events, number of products, number of sessions)  
- Temporal dynamics (e.g., active duration, inter-event time gaps)  
- Funnel behavior (view → cart → purchase transitions)  
- Session structure (session length statistics)  
- Category preference and diversity (top category, entropy)  
- Repeated interest signals (repeated views on the same product)  
- Price-related features (price distribution)  
- Category-aware conversion signals (global category-level purchase likelihood)  
- Additional behavioral patterns (e.g., zero-price ratio, unknown category ratio)  

This step converts variable-length behavioral sequences into fixed-length feature representations suitable for tabular models.

### 3. User-Level Aggregation

We further aggregate user behavior into a structured feature table, including engagement metrics, conversion rates, temporal patterns, diversity features, price statistics, and binary flags.

## ⚠️ Data Leakage Handling

Some features derived from purchase events (e.g., cart-to-purchase rate, time to first purchase) are excluded during model training to avoid target leakage.

## 🤖 Modeling

We train standard tabular models, including Logistic Regression, Random Forest, XGBoost, and MLP. The final model uses a combination of user-level features and event-derived features.

## 🧪 Reproducibility

The full pipeline is implemented as standalone Python scripts and is available on GitHub:  
https://github.com/caiyf03/Ecommerce-purchase-prediction

The notebook version provides a structured and explanatory walkthrough, while the script version enables faster execution and reproducibility.

## 📁 Project Structure

The repository is organized into data, scripts, and notebooks. The data folder contains processed datasets, the scripts folder includes feature engineering and modeling code, and the notebooks folder contains analysis and documentation.

## 💡 Key Contributions

- Multi-stage pipeline for large-scale behavioral data  
- Rich feature engineering beyond simple aggregation  
- Integration of user behavior and category-level signals  
- Scalable design suitable for real-world e-commerce systems  

## 📌 Future Work

Future improvements include sequence-based models (e.g., Transformer), session-level prediction, and real-time inference pipelines.

## 📜 License

This project is developed as part of CIS 5450 at the University of Pennsylvania.

© 2026 YiFan Cai
