========================================================================
             DIABETES HEALTH INDICATORS PREDICTION PIPELINE
                         PROJECT README FILE
========================================================================

Author / Project Framework: End-to-End Predictive Health Analytics
Notebook Version: Cat_(v7)(1).ipynb
Dataset Source: Alex Teboul's Diabetes Health Indicators (BRFSS 2015 via KaggleHub)
Target Variable: Diabetes_binary (0 = No Diabetes, 1 = Prediabetes/Diabetes)

------------------------------------------------------------------------
1. PROJECT OVERVIEW & GOALS
------------------------------------------------------------------------
This project builds an advanced data science and machine learning pipeline to 
analyze, cluster, and predict diabetes status based on survey responses from the 
Behavioral Risk Factor Surveillance System (BRFSS 2015). 

Because medical survey datasets are heavily imbalanced (far fewer diabetic cases 
than non-diabetic cases), standard accuracy metrics can be highly misleading. 
This project focuses on optimizing classification models using custom threshold-based 
scoring systems that prioritize a high Recall rate (to catch as many positive cases 
as possible) while preserving a robust, acceptable baseline level of Precision and 
Overall Accuracy.

Additionally, the pipeline employs advanced unsupervised learning techniques (K-Means 
and Gaussian Mixture Models) to identify latent patient health profiles, which are 
subsequently visualized using multidimensional radar charts.

------------------------------------------------------------------------
2. SYSTEM REQUIREMENTS & DEPENDENCIES
------------------------------------------------------------------------
To successfully execute this notebook, install the required packages using pip:

    pip install pandas numpy matplotlib seaborn scipy scikit-learn imblearn xgboost lightgbm ipywidgets

Core Library Breakdown:
  - Data Processing & Stats:  pandas, numpy, scipy
  - Visualizations:           matplotlib, seaborn, openpyxl (for charts if required)
  - Unsupervised & Preprocess: scikit-learn (KMeans, GMM, StandardScaler, SelectKBest)
  - Imbalance Management:     imblearn (SMOTE - Synthetic Minority Over-sampling Technique)
  - Supervised Classifiers:   scikit-learn, xgboost, lightgbm
  - Interactive Widgets:      ipywidgets (for dynamic user interfaces and slider elements)

------------------------------------------------------------------------
3. PIPELINE ARCHITECTURE & WORKFLOW STAGES
------------------------------------------------------------------------

STAGE 1: DATA INGESTION & QUALITY CONTROL
  * File loaded: `diabetes_binary_health_indicators_BRFSS2015.csv`
  * Baseline Dimensions: 253,680 records across 21 structural features.
  * Quality Control: Identifies and removes 24,206 exact duplicate records to prevent 
    data leakage and artificially inflated performance scores during validation.
  * Type Alignment: Explicitly parses features into correct datatypes:
    - Numerical Columns: 'BMI', 'MentHlth' (Mental Health days), 'PhysHlth' (Physical Health days).
    - Categorical/Ordinal Columns: 'HighBP', 'HighChol', 'Smoker', 'Stroke', 'GenHlth', 
      'Age', 'Education', 'Income', 'DiffWalk', etc.

STAGE 2: FEATURE SELECTION & TRANSFORMATION
  * Employs `SelectKBest` driven by an ANOVA F-value (`f_classif`) scoring function to 
    rank feature correlation and variance with respect to the binary target.
  * Constructs robust preprocessing pipelines using Scikit-Learn's `ColumnTransformer`. 
    Numerical features undergo `StandardScaler` transformations, ensuring high-variance 
    inputs do not disproportionately bias downstream distance-based models (e.g., GMM or KNN).

STAGE 3: UNSUPERVISED CLUSTERING & PATIENT PROFILING
  To understand broader health patterns independent of the target label, the pipeline 
  implements two advanced clustering methodologies:
  * K-Means Clustering: Profiled over k=4 clusters to isolate behavioral patterns 
    (combining indicators like physical activity, walking difficulties, and income groups).
  * Gaussian Mixture Models (GMM): Formulates a probabilistic soft-clustering structure 
    at k=3. Segment distributions show asymmetric group splits (e.g., Cluster 1 encapsulating 
    ~173,401 elements).
  * Visual Analytics: Outputs custom-built multi-axis Radar Charts to visualize 
    cluster profiles across normalized dimensions.

STAGE 4: IMBALANCE CORRECTION & PREDICTIVE MODELING
  * Address Class Imbalance: Employs SMOTE (Synthetic Minority Over-sampling Technique) 
    within cross-validation boundaries to ensure models train on balanced class 
    representations without leaking validation data information.
  * Model Benchmarking Matrix: Iteratively trains and builds cross-validation matrices for:
    - Logistic Regression
    - Random Forest Classifier
    - K-Nearest Neighbors (KNN)
    - XGBoost (Extreme Gradient Boosting)
    - LightGBM (Light Gradient Boosting Machine)

STAGE 5: MODEL FILTERING & HIGHLIGHTS
  * Employs custom criteria functions: `recall_then_accuracy` and `recall_then_balanced_accuracy`.
  * Setting a baseline constraint of minimum Accuracy >= 70% and minimum Recall >= 70%, 
    gradient-boosted tree structures emerge as the top performers.
  * Evaluation Summary Matrix:
    - XGBoost Classifier:  Mean Recall ~78.0% | Mean Accuracy ~70.6%
    - LightGBM Classifier: Mean Recall ~77.8% | Mean Accuracy ~71.0%

------------------------------------------------------------------------
4. HOW TO RUN THE NOTEBOOK
------------------------------------------------------------------------
1. Place the dataset file (`diabetes_binary_health_indicators_BRFSS2015.csv`) in the 
   same directory as the `Cat_(v7)(1).ipynb` notebook file.
2. Open your terminal or Anaconda prompt, navigate to the folder, and run:
   
    jupyter notebook
   
3. Open `Cat_(v7)(1).ipynb` and click "Run All" or execute cells sequentially from 
   top to bottom.
4. Interact with the custom ipywidgets layouts embedded in the visualization and 
   model filtering sections to examine how different decision thresholds impact 
   model selection.

------------------------------------------------------------------------
5. PROJECT DIRECTORY METADATA
------------------------------------------------------------------------
- Source Code:         Cat_(v7)(1).ipynb (Jupyter Notebook Format)
- Target Data Dependencies:  diabetes_binary_health_indicators_BRFSS2015.csv
- Expected File Outputs: Generated inline matplotlib/seaborn plots, evaluation metric 
                         matrices, and interactive UI views.

========================================================================
                    [ END OF PROJECT README.TXT ]
========================================================================
