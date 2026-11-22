# Comment Category Prediction

Multi-class classification of user comments into one of **4 categories**, using the comment text together with engagement signals, emoticon and topic indicators, and timestamp metadata.

Built for the **Comment Category Prediction Challenge**, a Kaggle community competition hosted as part of the IIT Madras BS in Data Science and Applications programme (Jan – Mar 2026).

## Results

| Metric | Score |
|---|---|
| Private leaderboard score | **0.81644** |
| Public leaderboard score | 0.82245 |
| Final rank | **544 of 2,744 teams (top 20%)** |
| Validation accuracy (20% stratified hold-out) | 0.9158 |
| Validation macro F1 | 0.83 |

The leaderboard uses the competition's custom metric; validation accuracy is measured on my own hold-out split.

## Data

- 198,000 labelled training rows and 102,000 test rows
- **Text:** the raw comment
- **Engagement:** upvotes and downvotes
- **Indicators:** three emoticon groups, four topic indicators (race, religion, gender, disability) and two hidden platform features
- **Metadata:** creation timestamp and parent post ID
- **Target:** one of 4 internal handling categories (imbalanced)

The dataset is not included in this repository, in line with the competition rules.

## Approach

**1. Exploratory data analysis**
Label distribution, votes by category, comment-length distribution and a correlation heatmap of numeric features.

**2. Feature engineering**
- Timestamp split into year, month, day, hour and day of week
- Text statistics: character length, word count, exclamation and question-mark counts
- Missing topic indicators mapped to a `missing` category; other numeric columns cleaned
- Upvote and downvote outliers capped at the 99th percentile, learned from training data only (no leakage into validation or test)

**3. Text features: dual TF-IDF**
- Word-level TF-IDF: unigrams and bigrams, 15,000 features
- Character-level TF-IDF: 3- to 5-character n-grams, 10,000 features

**4. Final model**
LightGBM (`num_leaves=63`) with custom class weights to improve recall on the two minority classes, trained inside a scikit-learn `ColumnTransformer` pipeline.

**5. Models explored**
Dummy baseline, Logistic Regression, SGD, Complement Naive Bayes, KNN, SVM, Decision Tree, Random Forest, MLP, XGBoost, LightGBM, CatBoost, PCA/SVD + Logistic Regression, and averaging and stacking ensembles. These are kept, commented out, in the notebook's appendix.

### Validation accuracy across experiments

| Model | Validation accuracy |
|---|---|
| Complement Naive Bayes | 0.7026 |
| Tuned Random Forest | 0.7831 |
| Support Vector Machine | 0.7888 |
| K-Nearest Neighbors | 0.8128 |
| PCA + Logistic Regression | 0.8765 |
| SGDClassifier (tuned) | 0.8811 |
| Decision Tree | 0.8914 |
| Logistic Regression (best CV) | 0.8997 |
| CatBoost | 0.9040 |
| Tuned MLP | 0.9059 |
| Stacking Ensemble | 0.9096 |
| Custom-weighted XGBoost | 0.9133 |
| Soft-voting Ensemble (XGBoost + LightGBM) | 0.9135 |
| Dual TF-IDF LightGBM (variants) | 0.9158 – 0.9176 |

The submitted model is the dual TF-IDF LightGBM (validation 0.9158, private leaderboard 0.81644).

## Repository structure

```
comment_category_prediction.ipynb   # full notebook with outputs
requirements.txt                    # Python packages used
README.md
```

## How to run

**On Kaggle (easiest):** create a notebook in the competition, attach the competition data, upload this notebook and run all cells. A full run takes about 22 minutes on CPU.

**Locally:** put `train.csv`, `test.csv` and `Sample.csv` in a folder, update the three file paths in section 2 of the notebook, then:

```bash
pip install -r requirements.txt
jupyter notebook comment_category_prediction.ipynb
```

## Tech stack

Python, pandas, NumPy, scikit-learn, LightGBM, Matplotlib, Seaborn
