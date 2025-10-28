# AutoGluon Machine Learning Notebooks Collection

## Video Tutorials

# Name : Kalhar Mayurbhai Patel
# SJSU ID: 019140511

### Complete Playlist
Watch all 5 AutoGluon tutorials here: [AutoGluon Complete Series - YouTube Playlist](https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID)

### Individual Videos
1. [AutoGluon Tabular Quick Start](https://youtu.be/SIGcm1AMEi8) 
2. [AutoGluon Feature Engineering](https://youtu.be/9ywIfIy584k) 
3. [California Housing Prediction](https://youtu.be/MZzVYOLlxxU) 
4. [IEEE Fraud Detection](https://youtu.be/qKa8qXI-Hc0) 
5. [AutoGluon Multimodal](https://youtu.be/AF66LSDOHmk) 

---

## Overview

This collection contains 5 comprehensive AutoGluon notebooks demonstrating automated machine learning for various use cases. AutoGluon automates the entire ML pipeline including feature engineering, model selection, hyperparameter tuning, and ensemble creation.

## Notebooks

### 1. AutoGluon Tabular Quick Start
**File:** `auto_gluon_tabular_quick_start.ipynb`

**Objective:** Introduction to AutoGluon Tabular for automated machine learning

**Key Features:**
- Quick setup and installation
- Automatic data preprocessing
- Multiple model training with single line
- Model comparison and leaderboard
- Predictions and evaluation
- Feature importance analysis

**Dataset:** Sample tabular dataset for classification/regression

**What You'll Learn:**
- Basic AutoGluon workflow
- TabularPredictor setup
- Model fitting with minimal code
- Performance evaluation
- Making predictions on new data

**Runtime:** ~5-7 minutes

**Use Cases:** 
- Rapid prototyping
- Baseline model creation
- Quick model comparison

---

### 2. AutoGluon Feature Engineering
**File:** `autogluon_feature_engineering__1_.ipynb`

**Objective:** Deep dive into AutoGluon's automatic feature engineering capabilities

**Key Features:**
- Column type detection (numeric, categorical, datetime, text)
- Automatic feature transformations
- Datetime feature extraction (year, month, day, dayofweek)
- Categorical encoding
- Text feature processing (n-grams, special features)
- Missing value handling strategies
- Custom feature generators

**Dataset:** Multi-type dataset with various column types

**What You'll Learn:**
- How AutoGluon detects feature types
- Automatic feature engineering pipeline
- Custom preprocessing options
- Feature generator customization
- Handling different data types

**Runtime:** ~8-10 minutes

**Advanced Topics:**
- AutoMLPipelineFeatureGenerator
- PipelineFeatureGenerator customization
- Feature metadata specification
- Outlier removal
- Feature transformation

---

### 3. California Housing Price Prediction (Small Dataset)
**File:** `california_housing_autogluon_smal__1_.ipynb`

**Objective:** Fast housing price prediction with AutoGluon on small dataset

**Key Features:**
- California Housing dataset (500 samples for fast training)
- Regression task
- Exploratory data analysis
- Quick model training (<2 minutes)
- Model leaderboard
- Feature importance
- Predictions and visualizations

**Dataset:** California Housing (subset of 500 samples)

**What You'll Learn:**
- Regression with AutoGluon
- Fast prototyping on small datasets
- Model evaluation metrics (RMSE, R2, MAE)
- Feature correlation analysis
- Price predictions

**Runtime:** ~3-5 minutes (fast training mode)

**Models Trained:**
- LightGBM
- CatBoost
- Neural Networks
- Weighted Ensemble

**Visualizations:**
- Correlation heatmaps
- Scatter plots
- Feature importance
- Predicted vs Actual plots

---

### 4. IEEE Fraud Detection
**File:** `ieee_fraud_detection.ipynb`

**Objective:** Advanced fraud detection using AutoGluon with real-world imbalanced dataset

**Key Features:**
- IEEE-CIS Fraud Detection dataset
- Highly imbalanced binary classification
- Advanced preprocessing
- Class imbalance handling
- Multiple evaluation metrics
- Model interpretation
- Production deployment

**Dataset:** IEEE-CIS Fraud Detection (Kaggle competition data)

**What You'll Learn:**
- Handling imbalanced datasets
- Advanced classification techniques
- Feature selection for large datasets
- Model stacking for better performance
- Fraud detection patterns
- Production model deployment

**Runtime:** ~15-20 minutes (large dataset)

**Special Features:**
- Automatic class balancing
- Time-based features
- Transaction pattern analysis
- Anomaly scoring
- Model calibration

**Evaluation Metrics:**
- AUC-ROC
- Precision-Recall
- F1-Score
- Confusion Matrix
- Classification Report

**Advanced Techniques:**
- Bagging and stacking
- Feature engineering for fraud
- Hyperparameter tuning
- Model interpretation with SHAP

---

### 5. AutoGluon Multimodal
**File:** `multimodal.ipynb`

**Objective:** Multimodal learning combining text, images, and tabular data

**Key Features:**
- Text + Image + Tabular data fusion
- Automatic data type handling
- Transfer learning with pretrained models
- End-to-end multimodal training
- Cross-modal feature learning

**Dataset:** Multimodal dataset with mixed data types

**What You'll Learn:**
- Multimodal machine learning
- Handling heterogeneous data
- Text and image processing
- Feature fusion strategies
- Advanced AutoGluon capabilities

**Runtime:** ~10-15 minutes

**Data Types Supported:**
- Text (NLP with transformers)
- Images (computer vision)
- Tabular features (structured data)
- Mixed combinations

**Models Used:**
- BERT/DistilBERT for text
- ResNet/EfficientNet for images
- Neural networks for tabular
- Fusion layers

**Applications:**
- Product classification with images and descriptions
- Sentiment analysis with text and user data
- E-commerce recommendations
- Content moderation

---

## Installation

### Requirements
```bash
pip install autogluon
```

### Optional (for specific modules)
```bash
# For multimodal
pip install autogluon.multimodal

# For time series
pip install autogluon.timeseries

# For tabular with all features
pip install autogluon.tabular[all]
```

### Google Colab Setup
All notebooks are designed for Google Colab:
```python
!pip install autogluon
```

---

## Quick Start Guide

### Basic Usage Pattern

```python
from autogluon.tabular import TabularPredictor

# 1. Load data
train_data = pd.read_csv('train.csv')

# 2. Create predictor
predictor = TabularPredictor(label='target_column')

# 3. Train (AutoML happens here!)
predictor.fit(train_data)

# 4. Make predictions
predictions = predictor.predict(test_data)

# 5. Evaluate
predictor.leaderboard(test_data)
```

That's it! AutoGluon handles:
- Feature engineering
- Model selection
- Hyperparameter tuning
- Ensemble creation
- Best model selection

---

## Key Features Demonstrated

### 1. Auto ML Pipeline
- Automatic preprocessing
- Feature type detection
- Missing value handling
- Feature engineering
- Model selection

### 2. Model Training
- Multiple algorithms automatically
- Hyperparameter optimization
- Ensemble methods
- Stacking and bagging
- Neural architecture search

### 3. Evaluation
- Comprehensive metrics
- Leaderboard comparison
- Feature importance
- Model interpretation
- Performance visualization

### 4. Production Deployment
- Model persistence
- Fast inference
- Scalable predictions
- Easy API integration

---

## Datasets Summary

| Notebook | Dataset | Size | Task | Target |
|----------|---------|------|------|--------|
| Quick Start | Sample | Small | Classification/Regression | Various |
| Feature Engineering | Multi-type | Medium | Educational | Feature demo |
| California Housing | Housing | 500 samples | Regression | House prices |
| IEEE Fraud | Transactions | Large | Binary Classification | Fraud/Legitimate |
| Multimodal | Mixed data | Medium | Classification | Category |

---

## Performance Benchmarks

### Training Times (on Google Colab)
- Quick Start: 2-3 minutes
- Feature Engineering: 3-5 minutes
- California Housing: 2-3 minutes (small dataset)
- IEEE Fraud: 15-20 minutes (large dataset)
- Multimodal: 10-15 minutes

### Expected Accuracy
- California Housing: R² > 0.80, RMSE < $60k
- IEEE Fraud: AUC > 0.90, F1 > 0.70
- Multimodal: Accuracy > 0.85

---

## Common Use Cases

### Business Applications
1. **Customer Churn Prediction** (Quick Start pattern)
2. **Price Forecasting** (California Housing)
3. **Fraud Detection** (IEEE Fraud)
4. **Product Classification** (Multimodal)
5. **Risk Assessment** (Feature Engineering)

### Technical Applications
1. **Baseline Model Creation**
2. **Feature Engineering Research**
3. **Model Comparison Studies**
4. **Production ML Pipelines**
5. **AutoML Benchmarking**

---

## Notebook Execution Order

### Recommended Learning Path:

1. **Start:** `auto_gluon_tabular_quick_start.ipynb`
   - Learn basics of AutoGluon
   - Understand workflow
   - Get familiar with API

2. **Deep Dive:** `autogluon_feature_engineering__1_.ipynb`
   - Understand automatic feature engineering
   - Learn data type handling
   - Custom preprocessing

3. **Practice:** `california_housing_autogluon_smal__1_.ipynb`
   - Apply to real problem
   - Fast iteration
   - Complete workflow

4. **Advanced:** `ieee_fraud_detection.ipynb`
   - Handle complex datasets
   - Imbalanced classification
   - Production techniques

5. **Cutting Edge:** `multimodal.ipynb`
   - Multimodal learning
   - Latest AutoGluon features
   - Advanced use cases

---

## Tips and Best Practices

### 1. Data Preparation
- Ensure target column is clearly defined
- Handle extreme outliers manually if needed
- Check for data leakage
- Verify train/test split is appropriate

### 2. Model Training
- Start with default settings
- Use `time_limit` for faster iteration
- Try different `presets` (best_quality, high_quality, good_quality, medium_quality)
- Enable GPU if available

### 3. Evaluation
- Look at leaderboard for model comparison
- Check feature importance
- Validate on holdout set
- Use appropriate metrics for your problem

### 4. Production Deployment
- Save model with `predictor.save()`
- Load with `TabularPredictor.load()`
- Use `predictor.predict()` for inference
- Monitor performance over time

---

## Troubleshooting

### Common Issues

**Issue:** Out of memory during training
**Solution:** Reduce dataset size, use sampling, or increase available RAM

**Issue:** Training takes too long
**Solution:** Set `time_limit` parameter, use smaller `presets`, or reduce data

**Issue:** Poor model performance
**Solution:** Check data quality, try different presets, add feature engineering

**Issue:** Installation errors
**Solution:** Use correct Python version (3.8+), install in clean environment

---

## Additional Resources

### Official Documentation
- [AutoGluon Documentation](https://auto.gluon.ai/)
- [Tabular Tutorial](https://auto.gluon.ai/stable/tutorials/tabular/index.html)
- [Multimodal Guide](https://auto.gluon.ai/stable/tutorials/multimodal/index.html)
- [API Reference](https://auto.gluon.ai/stable/api/index.html)

### Community
- [GitHub Repository](https://github.com/autogluon/autogluon)
- [Discussion Forum](https://github.com/autogluon/autogluon/discussions)
- [Paper (arXiv)](https://arxiv.org/abs/2003.06505)

### Learning Resources
- [AutoGluon Tutorials](https://auto.gluon.ai/stable/tutorials/index.html)
- [Example Notebooks](https://github.com/autogluon/autogluon/tree/master/examples)
- [Blog Posts](https://auto.gluon.ai/stable/tutorials/index.html#blog-posts)
