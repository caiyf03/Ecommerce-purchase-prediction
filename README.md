# E-Commerce Purchase Prediction Using User Behavioral Data

## 1. Introduction

### 1.1 Problem Definition

In this project, we aim to predict whether a user will make a purchase based on their historical interaction behavior on an e-commerce platform. Formally, this is a binary classification problem at the user level, where the goal is to estimate whether a user will convert (i.e., make at least one purchase) within a given time window.

The prediction is constructed using large-scale behavioral logs that record user interactions such as product views, cart additions, and purchases.

---

### 1.2 Motivation

Predicting user purchase behavior is a fundamental task in modern e-commerce systems. It directly supports:

- Personalized recommendation systems  
- Targeted advertising  
- Customer retention strategies  

However, the task is challenging due to:

- Strong class imbalance (most users do not purchase)  
- Sequential and heterogeneous user behavior  
- Large-scale noisy real-world data  

---

### 1.3 Approach Overview

We design a **multi-stage data processing and modeling pipeline**:

Specifically:

1. Construct a clean **event-level dataset**  
2. Build an intermediate **event-derived feature table**  
3. Aggregate into a **user-level dataset**  
4. Train classification models for prediction  

---

## 2. Dataset Description

The dataset used in this project is the **REES46 Multi-Category Store E-commerce Behavioral Dataset (October 2019)**.

### Key characteristics:

- ~42 million interaction records  
- ~3 million unique users  
- Event types:
  - view
  - cart
  - remove_from_cart
  - purchase  

### Key columns:

- `user_id`
- `event_time`
- `event_type`
- `product_id`
- `category_id`
- `category_code`
- `brand`
- `price`
- `user_session`

---

### Why this dataset is suitable

- Directly contains **purchase labels**
- Covers full **user interaction funnel**
- Includes **temporal information**
- Large-scale → realistic modeling scenario

---

## 3. Event-Level Data Processing

### 3.1 Overview

The event-level dataset is the **foundation of the pipeline**.  
Each row represents a single user action, with no aggregation applied.

---

### 3.2 Processing Workflow

The pipeline includes:

- Timestamp parsing (UTC)
- Missing value handling:
  - brand → "unknown"
  - category_code → "unknown"
  - category_id → 0
- Category hierarchy splitting
- Rare category consolidation (<0.01%)
- Sorting by `(user_id, event_time)`

---

### 3.3 Design Considerations

- No aggregation → avoids leakage  
- Preserves full behavioral sequence  
- Maintains session integrity  

---

## 4. Event-Level Feature Engineering

### 4.1 Motivation

Raw event-level data is not directly suitable for modeling, because:

- Models require **fixed-length inputs**
- Event data is **variable-length sequences**

Therefore, we construct an intermediate feature table:

---

### 4.2 Core Idea

Transform:

---

### 4.3 Feature Groups

#### (1) Behavioral Statistics

- total_events  
- num_products  
- num_categories  
- num_sessions  

---

#### (2) Temporal Features

- active_duration  
- mean / std / max delta_time  

→ captures browsing rhythm and activity intensity

---

#### (3) Funnel Features

- view_to_cart_rate  
- cart_to_purchase_rate  
- purchase_per_event  

→ models conversion behavior

---

#### (4) Session Features

- avg_session_len  
- max_session_len  

→ captures session structure

---

#### (5) Purchase Timing Features

- time_to_first_purchase  
- fast_purchase  

→ measures conversion speed

---

#### (6) Category Features

- top_category  
- entropy  

→ measures user interest diversity

---

#### (7) Repeated Interest Features

- avg_repeat_view  
- max_repeat_view  

→ captures strong intent signals

---

#### (8) Price Features

- avg_price  
- max_price  
- min_price  
- std_price  

→ captures price sensitivity

---

#### (9) Category-Aware Conversion Features

- category-level conversion rate  
- smoothed conversion  
- high-conversion category ratio  

→ integrates **global product popularity**

---

#### (10) Additional Features

- zero_price_ratio  
- unknown_category_ratio  

→ captures noise and anomalies

---

### 4.4 Key Insight

This step converts:

---

### 4.5 Data Leakage Consideration

Some features use purchase information:

- cart_to_purchase_rate  
- time_to_first_purchase  

These are **excluded during model training** to avoid target leakage.

---

## 5. User-Level Feature Engineering

### 5.1 Overview

The user-level dataset aggregates all events into one row per user.

---

### 5.2 Feature Construction

Includes:

- Event counts (views, carts, purchases)
- Conversion rates (smoothed)
- Temporal patterns
- Diversity features
- Price statistics
- Binary flags

---

### 5.3 Feature Summary

| Feature Type | Description |
|------|-------------|
| Event counts | Engagement intensity |
| Conversion | Funnel efficiency |
| Temporal | Activity pattern |
| Diversity | Exploration behavior |
| Price | Sensitivity |
| Flags | Simple behavioral signals |

---

## 6. Relationship Between Datasets

The pipeline consists of three layers:

- Event-level → raw behavior  
- Feature table → enriched representation  
- User-level → model-ready data  

---

## 7. Modeling Perspective

### 7.1 Baseline

User-level dataset:
- Simple aggregation
- Coarse behavioral signals

---

### 7.2 Enhanced Representation

Event-derived features:
- Temporal dynamics  
- Session patterns  
- Conversion behavior  

---

### 7.3 Final Input

Final model uses:

---

## 8. Engineering Design

### 8.1 Scalability

- Handles 40M+ rows  
- Uses aggregation pipelines  

---

### 8.2 Modularity

- Event processing  
- Feature engineering  
- Modeling  

---

### 8.3 Reproducibility

- Dataset hosted on Hugging Face  
- Code available on GitHub  

---

## 9. Conclusion

This project demonstrates how large-scale behavioral data can be transformed into meaningful representations for predictive modeling.

Key contributions include:

- A multi-stage data processing pipeline  
- Rich behavioral feature engineering  
- Integration of global and user-level signals  

The approach provides a scalable and practical solution for user purchase prediction in real-world e-commerce systems.
