# IMDb Rating Prediction from Movie Reviews

## Project Overview

This project investigates whether IMDb ratings can be predicted from reviews and explores how metadata and reviewer information enhance prediction accuracy.

This is a data science project for the ECON-2206 Data Management course at HEC Liège, combining natural language processing (NLP), machine learning, and data engineering techniques.

---

## Research Question

Can IMBd ratings be accurately predicted using the textual content of movie reviews and does adding movie metadata and reviewer information improve prediction accuracy compared to sentiment analysis alone?

---

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



### Random Forest