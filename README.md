# IMDb Rating Prediction from Movie Reviews

# Table of Contents

- [IMDb Rating Prediction from Movie Reviews](#imdb-rating-prediction-from-movie-reviews)
- [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Research Question](#research-question)
  - [Installation \& Usage](#installation--usage)
  - [Technologies \& Libraries](#technologies--libraries)
  - [Datasets](#datasets)
    - [Data Structure](#data-structure)
  - [Project Pipeline](#project-pipeline)
    - [Step 1: Data Collection \& Cleaning](#step-1-data-collection--cleaning)
    - [Step 2: Data Cleaning \& Management](#step-2-data-cleaning--management)
    - [Step 3: Visualizations](#step-3-visualizations)
    - [Step 4: Modeling](#step-4-modeling)
    - [Linear Regressions](#linear-regressions)
    - [TF-IDF \& Linear Regression](#tf-idf--linear-regression)
    - [Random Forest](#random-forest)
  - [Model Comparison Summary](#model-comparison-summary)
  - [Team Members](#team-members)

## Project Overview

This project investigates whether IMDb ratings can be predicted from reviews and explores how metadata and reviewer information enhance prediction accuracy.

This is a data science project for the ECON-2206 Data Management course at HEC Liège, combining natural language processing (NLP), machine learning, and data engineering techniques.

## Research Question

Can IMBd ratings be accurately predicted using the textual content of movie reviews and does adding movie metadata and reviewer information improve prediction accuracy compared to sentiment analysis alone?

## Installation & Usage

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook final_project.ipynb
   ```

2. Execute cells sequentially:
   - **Step 1**: Data Cleaning & Collection
   - **Step 2**: Data Management
   - **Step 3**: Visualizations
   - **Step 4**: Modelling

## Technologies & Libraries

**Data Processing & Analysis**:
- `polars`: High-performance data loading and manipulation
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing

**Text Processing & NLP**:
- `nltk`: VADER sentiment analyzer
- `textblob`: Subjectivity analysis
- `unidecode`: Unicode normalization
- `sklearn.feature_extraction.text.TfidfVectorizer`: Feature extraction

**Machine Learning**:
- `sklearn.linear_model.LinearRegression`: Linear regression models
- `sklearn.ensemble.RandomForestRegressor`: Ensemble learning method
- `sklearn.preprocessing.StandardScaler`: Feature scaling
- `sklearn.preprocessing.OneHotEncoder`: Categorical encoding
- `sklearn.compose.ColumnTransformer`: Pipeline feature preprocessing
- `sklearn.pipeline.Pipeline`: Model pipelines
- `sklearn.model_selection.train_test_split`: Data splitting
- `sklearn.metrics`: Performance evaluation (RMSE, R²)

**Visualization**:
- `matplotlib.pyplot`: Plotting library
- `seaborn`: Statistical data visualization

## Datasets 

This project utilizes two primary data sources, merged on **normalized movie titles**.

- **Movie Reviews**: The review data is obtained through Kaggle and is stored across multiple parts of JSON files, all files are read and concatenated into a single Polars Dataframe. (https://www.kaggle.com/datasets/ebiswas/imdb-review-dataset?resource=download)

- **Movie Metadata**: The movies metadata are obtained from IMDb Non-Commercial Datasets. The IMDb datasets used are (`title.basics.tsv` which contains the titles, years, genres and runtime and `title.ratings.tsv` which contains the average ratings) and are merged on the IMDb identifier  (https://datasets.imdbws.com/)

After cleaning and merging, a **sample of 500,000 reviews** was used for analysis and modeling. This final dataset is referred to as `imdb_reviews_500k.csv`.

Due to GitHub's file size limits, the raw data files and the final dataset are not included in this repository. The dataset can be reproduced by downloading the datasets and running the scripts in the jupyter notebook.

### Data Structure

The final merged dataset contains the following columns:

- `reviewid`: Unique identifier for each review
- `reviewer`: Username of the reviewer
- `movie`: Movie title as it appears in the review database
- `rating`: User's rating (1-10 scale)
- `reviewsummary`: Summary/headline of the review
- `reviewdate`: Date the review was posted
- `reviewdetail`: Full text of the review
- `cleantitle`: Title with parenthetical information removed
- `titlenorm`: Normalized and standardized title for matching
- `tconst`: IMDb identifier for the movie
- `titleType`: Type of content (filtered to "movie" only)
- `primaryTitle`: Official IMDb movie title
- `startYear`: Year the movie was released
- `runtimeMinutes`: Duration of the film in minutes
- `genres`: Primary genre classification
- `averageRating`: IMDb's average rating for the movie
- `numVotes`: Number of votes the movie received

## Project Pipeline

### Step 1: Data Collection & Cleaning

**Process**:
- Imported review data from multiple JSON files using Polars for efficiency
- Applied text cleaning to remove parentheses and special characters from titles
- Normalized titles (lowercase, removed accents, punctuation) to facilitate matching
- Loaded IMDb metadata from TSV files using Pandas
- Merged datasets on normalized titles and IMDb identifiers (`tconst`)
- Removed unmatched reviews
- Sampled 500,000 reviews due to having millions of reviews that we have in the original database.

---

### Step 2: Data Cleaning & Management

**Features Created**:

**Text-Based Features**:
- `reviewlengthchar`: Character count of the review
- `reviewlengthwords`: Word count of the review
- `reviewemphasis`: Count of exclamation and question marks
- `reviewvocabrichness`: Ratio of unique words to total words, used to capture the linguistic diversity of a review.

**Sentiment Features**:
- `sentimentcompound`: VADER sentiment score (-1 to 1, normalized), used to capture the sentiment of the review (positive or negative)
- `sentimentsubjectivity`: TextBlob subjectivity score (0 to 1)
- `subjectivityxsentiment`: Interaction term between sentiment and subjectivity

**Temporal Features**:
- `movieage`: Years since movie release (2025 - startYear)
- `isclassic`: Binary indicator (1 if released before 1980)
- `isrecent`: Binary indicator (1 if released after 2015)

**Reviewer Features**:
- `reviewernreviews`: Total reviews posted by reviewer
- `revieweravgrating`: Average rating given by reviewer
- `reviewerbias`: Deviation from dataset mean rating
- `is_heavy_reviewer`: Top 5% of reviewers
- `reviewergroup`: Categorical grouping, from casual to heavy reviewers (1 review, 2-5, 6-20, 21-100, 100+)

**Movie Features**:
- `primarygenre`: Primary genre category
- Metadata features: `averageRating`, `numVotes`, `runtimeMinutes`

---

### Step 3: Visualizations

**Visualizations Created**:
1. Distribution of IMDb ratings (histogram with KDE)
2. Sentiment (VADER) vs. rating (scatter plot with regression line)
3. Review length vs. rating (boxplot)
4. Review emphasis vs. rating (boxplot)
5. Review subjectivity vs. rating (boxplot)
6. Reviewer group vs. rating (boxplot)
7. Reviewer bias vs. rating (scatter plot)
8. Mean rating per genre (barplot)
9. Correlation heatmap of all numerical variables

---

### Step 4: Modeling

**Models Tested**: For the modeling, three different methods were used: Linear Regression, TF-IDF & Linear Regression and Random Forest

### Linear Regressions

**Model 1: Linear Regression (NLP Only)**

The first model uses only textual variables.

- **Features added**: `sentimentcompound`, `sentimentsubjectivity`, `subjectivityxsentiment`
- **Purpose**: Test if review text alone with only the sentiment and subjectivity predicts ratings
- **Result**: RMSE = 2.67, R² = 0.196 (poor predictive power)
- **Conclusion**: The sentiment and subjectivity of the text alone is insufficient for rating prediction. 

**Model 2: Linear Regression with Metadata**

The second model uses textual variables and metadata from IMDb.

- **Additions**: `movieage`, `averageRating`, `numVotes`, `primarygenre`
- **Purpose**: Test if movie context improves predictions
- **Result**: RMSE = 2.64, R² = 0.212 (minimal improvement)
- **Conclusion**: Movie metadata provides limited additional predictive value

**Model 3: Linear Regression with Metadata and Reviewer data**

The third model uses reviewer data; in addition to textual and metadata variables.

- **Additions**: `reviewerbias`, `reviewernreviews`, `reviewergroup`
- **Purpose**: Assess reviewer behavior's impact on predictions
- **Result**: RMSE = 1.41, R² = 0.776 (strong predictive power)
- **Conclusion**: Reviewer data is a major explanatory variable explaining the rating.

### TF-IDF & Linear Regression

 Unlike previous NLP models using pre-computed sentiment scores, this model directly analyzes the importance and uniqueness of words within review texts.

**Configuration**:

- Vectorizer: TfidfVectorizer with maximum 5,000 features
- Minimum document frequency: 5 (words appearing in fewer than 5 reviews excluded)
- Preprocessing: English stopword removal
- Regression: Linear Regression on the TF-IDF feature matrix

**Results**:

- RMSE: 1.975
- R² Score: 0.561

**Conclusion**:

The TF-IDF model explains 56.1% of variance in ratings, representing a substantial improvement over the linear regression with only textual variables (RMSE: 2.67, R²: 0.196). However, it remains inferior to the linear regression with metadata and reviewer data (R²: 0.776) because the model needs context and the behavior of the reviewer dominates.

### Random Forest

Random Forest is a non-linear program, used to capture potential non-linear relationships that could not be portrayed in earlier models.

**Configuration**:
- Number of estimators: 200 trees
- Max depth: None (trees grow until all samples are classified)
- Random state: 42 (for reproducibility)

**Results**:
- RMSE: **1.348**
- R² Score: **0.795**

**Conclusion**: The Random Forest achieves the best predictive accuracy of all models tested, explaining 79.5% of variance in movie ratings while its prediction is off by only 1.35 points.

## Model Comparison Summary

| Model | Features | RMSE | R² | Use Case |
|-------|----------|------|-----|----------|
| **Simple Linear Regression** | Sentiment only | 2.673 | 0.196 | Baseline; demonstrates text weakness |
| **Linear + Metadata** | + Movie context | 2.643 | 0.212 | Shows that metadata has minimal value |
| **Full Linear Regression** | + Reviewer behavior | 1.409 | 0.776 | Shows that reviewer data is a major explanotary variable |
| **TF-IDF Linear Regression** | Raw text vectorization | 1.975 | 0.561 | Word importance analysis |
| **Random Forest** | All features (non-linear) | 1.348 | 0.795 | Confirms that reviewer-related features largely impacts rating prediction|

## Team Members

* Megan Hardy (s191198)
* Hideaki Fukuyama (s221591)
