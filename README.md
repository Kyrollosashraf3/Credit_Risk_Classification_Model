# Credit_Risk_Classification_Model
kaggle competition

# Home Credit Default Risk Analysis

This notebook performs an exploratory data analysis (EDA) and data cleaning process for the Home Credit Default Risk dataset, sourced from Kaggle. The objective is to understand the factors influencing loan repayment default and prepare the data for potential machine learning model development.

## Table of Contents
1. [Data Loading](#data-loading)
2. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Missing Value Handling](#missing-value-handling)

## 1. Data Loading
The dataset `application_train.csv` is loaded from the Kaggle competition using `kagglehub`. The raw data provides information about clients' demographics, credit history, and other relevant features.

## 2. Data Cleaning and Preprocessing
A `wrangle` function is defined and applied to the DataFrame (`df`). This function performs several crucial preprocessing steps:
- **Outlier Treatment**: Caps `CNT_CHILDREN` to 4 and `AMT_INCOME_TOTAL` to 800,000.
- **Feature Engineering**: 
  - Creates `TOTAL_DOCS` and `TOTAL_DOCS_BIN` from `FLAG_DOCUMENT` columns.
  - Converts `DAYS_BIRTH` to `AGE` in years.
  - Converts `DAYS_EMPLOYED` to `YEARS_EMPLOYED`.
  - Creates `IS_PENSIONER` flag.
  - Derives ratio features like `INCOME_PER_PERSON`, `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `EMPLOYMENT_RATIO`, and `GOODS_CRIDET_RATIO`.
- **Missing Value Imputation (Initial)**:
  - Handles specific cases for `OCCUPATION_TYPE` related to 'Pensioner' and 'Unemployed' income types.
  - Fills remaining `OCCUPATION_TYPE` NaNs with 'Missing' and creates `IS_OCCUPATION_MISSING` flag.
  - Replaces the special `-365243` value in `YEARS_EMPLOYED` with NaN and imputes with the median.
- **Column Dropping**: Removes original `FLAG_DOCUMENT` columns, `SK_ID_CURR`, `REGION_RATING_CLIENT_W_CITY`, `WEEKDAY_APPR_PROCESS_START`, `DAYS_BIRTH`, and `DAYS_EMPLOYED` to reduce redundancy and multicollinearity.

## 3. Exploratory Data Analysis (EDA)
Extensive EDA is performed to understand data distributions, relationships, and potential predictive power of features:
- **Categorical Feature Analysis**: 
  - Identifies high-cardinality columns.
  - Analyzes the distribution of `NAME_CONTRACT_TYPE`, `TARGET`, `CODE_GENDER`, `FLAG_OWN_CAR`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `OCCUPATION_TYPE`, and `ORGANIZATION_TYPE`.
  - Visualizes default rates for various categorical features using pivot tables and bar plots.
  - Corrects `CODE_GENDER` by replacing 'XNA' with 'M'.
- **Numerical Feature Analysis**: 
  - Examines the distribution of `EXT_SOURCE_2` and `AMT_GOODS_PRICE` using histograms and box plots.
  - Analyzes `CNT_CHILDREN` for outliers and caps values.
  - Bins numerical features into quantiles (`pd.qcut`) and analyzes default rates across these bins.
  - Plots target ratios for top features using `plotly.express` for interactive visualization.
- **Time-Related Features**: Explores `AGE` and `YEARS_EMPLOYED` distributions and their relation to the `TARGET` variable.
- **Ratio Features**: Visualizes the relationship between engineered ratio features (e.g., `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`) and `TARGET` using box plots.
- **Flag Variables**: Analyzes the impact of various `FLAG` columns on default rates.

## 4. Missing Value Handling
The notebook employs a structured approach to missing value treatment, encompassing:
- **`missing_analysis_with_pd` function**: Calculates the percentage of missing values and the default rate within missing and non-missing subsets, as well as the difference (`diff`) from the overall default rate. This helps prioritize imputation strategies based on the predictive signal of missingness.
- **Identification of Missing Groups**: Columns are categorized into:
  - **High Missing, Low Impact (`high_miss_cols`)**: Columns with >60% missing values and a small `diff` in default rate are dropped.
  - **Important Missing (`important_missing_cols`)**: Columns with >30% missing values and a significant `diff` are flagged and imputed.
  - **Other Missing (`cols_to_impute`)**: All other columns with missing values are flagged and imputed.
- **Feature Engineering for Housing Columns**: For property-related columns (identified by 'AVG', 'MEDI', 'MODE' in their names), `HOUSING_MEAN` and `HOUSING_STD` aggregate features are created to capture information before dropping the original sparse columns.
- **Imputation Strategy**: Numeric columns are imputed with the median, and categorical columns with the mode. A missing indicator (`_MISS` suffix) is created for each imputed column.
- **Final Verification**: Confirms that all specified columns have been successfully imputed with no remaining missing values.



-------------------------------------------------------------------
In The Second Notebook -  notebook, we performed the following steps:

1.  **Data Loading:** 'Home Credit Default Risk' competition data was loaded using `kagglehub`.
2.  **Data Wrangling:** The `wrangle` function was applied to clean the data, handle outliers, create new features, convert time-based columns to years, and process income and occupation types.
3.  **Missing Value Handling:** The `full_missing_pipeline` function was used to deal with missing values, which included analyzing missing value percentages, grouping columns with significant missingness, dropping some columns, creating missing value flags, and imputing missing values (with the median for numerical and mode for categorical values).
4.  **Data Splitting:** The data was split into training and testing sets (a separate validation set was later used for decision tree model tuning).
5.  **Model Training and Evaluation:**

    *   **Logistic Regression:** A Pipeline including OneHotEncoder, StandardScaler, and a Logistic Regression model was built.
    *   **Random Forest:** A Pipeline including OneHotEncoder and a Random Forest model was built.
    *   **CatBoost:** Data was prepared for the CatBoost model, which handles categorical features directly.

During the experimentation process, I tried different data processing methods and their impact on model performance, including:

*   **Experiment 1 (My prepare - depth 6):** With my initial settings, including `wrangle` and `full_missing_pipeline` functions, CatBoost achieved **0.7631 AUC**.
*   **Experiment 2 (My prepare):** This experiment was a repeat of Experiment 1 with the same settings, yielding a CatBoost AUC of **0.7627**.
*   **Experiment 3 (0.05 importance, 50%):** After adjusting missing value handling criteria (0.05 importance and 50% missing values), CatBoost performance decreased to **0.7585 AUC**.
*   **Experiment 4 (only wrangle):** Using only the `wrangle` function without advanced missing value handling from `full_missing_pipeline`, CatBoost yielded **0.7627 AUC**.
*   **Experiment 5 (0.005 importance, 60% - NO MISS FLAGS):** With other missing value handling criteria (0.005 importance and 60% missing values, without adding missing flag columns), CatBoost yielded **0.7618 AUC**.

**Final Result:**

After a series of experiments and adjustments to the data processing pipeline, I settled on the **CatBoost** model as the best-performing one in this case. The best result was achieved using my initial setup ('My prepare') for CatBoost, where the **Valid ROC-AUC** reached **0.7627**.



