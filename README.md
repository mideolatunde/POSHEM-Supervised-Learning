#  POSHEM Supervised Learning for Spotify Music Classification Using Audio Features

## &#128279; Project Overview
This project applies supervised machine learning techniques to classify Spotify tracks based on their audio 
characteristics. Using Spotify’s audio feature metrics—such as danceability, energy, loudness, and valence—the 
model learns patterns that distinguish songs into predefined labels (e.g., 'sad', 'happy', 'energetic', 'calm') 

## &#128279; Authors
* Ayomide Olatunde

## &#128279; Table of Contents
* [Authors](#-authors)
* [Table of Contents](#-table-of-contents)
* [Dataset Description](#-dataset-description)
* [Tools & Libraries Used](#-tools--libraries-used)
* [Workflow Implemented](#-workflow-implemented)
* [Exploratory Data Analysis](#-exploratory-data-analysis)
* [Model Performance](#-model-performance)
* [Key Insights](#-key-insights)
* [Visualizations](#-visualizations)
* [Learning Outcomes](#-learning-outcomes)
* [How to Run the project](#-how-to-run-the-project)
* [Repository Structure](#-repository-structure)
* [License](#-license)
* [Contact](#-contact)


## &#128194; Dataset Description
**Source**: 278k_song_labelled.csv
**Records**: 277938 entries
**Columns**: 13 

#### **Features Used**
- duration (ms) - Length of the song in milliseconds 
- danceability - Suitability of a track for dancing (0–1)  
- energy - Perceived intensity and activity (0–1) 
- loudness - Overall loudness in decibels (dB) 
- speechiness - Presence of spoken words (0–1) 
- acousticness - Probability the track is acous c (0–1) 
- instrumentalness - Likelihood of no vocals (0–1) 
- liveness - Presence of a live audience (0–1) 
- valence - Musical positivity or mood (0–1) 
- tempo - Beats per minute (BPM) 
- spec_rate - Derived spectral/audio rate feature 

**Target Variable**
- labels - 'sad': 0, 'happy': 1, 'energetic': 2, 'calm': 3

## &#128736; Tools & Libraries Used

#### **Programming Language** : `Python 3.10+` 

#### Libraries
- `pandas` – Data manipulation & cleaning

- `numpy` - Numerical analysis

- `matplotlib & seaborn` – Data Visualization

- `scikit-learn (Logistic Regression, DecisionTree, RandomForest, SVM)` - Machine Learning

- `LightGBM` - Machine Learning

- `Streamlit & joblib` - Deployment

#### Environment
- `Jupyter Notebook` - Interactive analysis & documentation

- `github`


## &#129529; Workflow Implemented

The following steps were performed for this project:
1. Exploratory Data Analysis (EDA) 
- Distribution analysis of audio features 
- Correlation analysis between features and target label
- Class balance inspection 
- Visualization of feature patterns per label 

2. Data Preprocessing 
- Handled missing values (no missing values)
- Manual Oversampling to fix class imbalance
- Train-Test-Split (80/20)
- Feature scaling (StandardScaler) 

3. Model Training (Supervised Learning)  
Trained the features on the following models:
- Logistic Regression 
- Decision Tree 
- Random Forest 
- Support Vector Machine (SVM) 
- LightGBM

4. Model Evaluation  
Models were evaluated using: 
- Accuracy, Precision, Recall, and F1-Score 
- Confusion Matrix plot
- ROC-AUC (for multi-class classification) 
- Cross-validation for robustness (5-fold stratified cross-validation)

5. Deployment


## 📈 Exploratory Data Analysis  
- The distribution of the Duration, Speechiness, Acousticness, Instrumentalness, Liveness and Spec-rate features are right-skewed while Valence is slightly right-skewed.
- The distribution of Loudness is left-skewed while Energy and Danceability are slightly left-skewed.
- The Tempo Column is normally distributed.
- Instrumentalness has the most correlation to the target variable (0.54)
- Visualization of feature patterns per label shows that there's alot of outliers

## &#127942; Model Performance
| Model Name | Accuracy Score | F1 Score | Recall Score | ROC-AUC Score | Mean CV | S.D CV |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.846730 | 0.846730 | 0.845728 | 0.969690 | 0.846984 | 0.000955 |
| Decision Tree | 0.959938 | 0.959938 | 0.959950 | 0.973292 | 0.953053 | 0.000495 |
| Random Forest | 0.975277 | 0.975277 | 0.975304 | 0.999091 | 0.971260 | 0.000757 |
| Support Vector | 0.816546 | 0.816546 | 0.814112 | NaN	| 0.817422 | 0.001019 |
| LightGBM | 0.967314 | 0.967314 | 0.967335 | 0.998466 | 0.967414 | 0.000379 |



## 🔍 Key Insights


## &#127919; Real World Application


## &#128161; Learning Outcomes
Through this project, I demonstrated proficiency in:

#### Technical Skills
- Data cleaning, analysis and preprocessing
- Exploratory data analysis techniques
- Correlation analysis and interpretation
- Data visualization (Matplotlib, Seaborn)
- Python programming for data science
- Jupyter Notebook documentation


#### Tools Mastered
- Python (Pandas, NumPy, Matplotlib, Seaborn, SciPy, ScikitLearn, Streamlit, Joblib, LightGBM)
- Jupyter Notebook
- Using different models  

## &#128640; How to Run the Project
1. **Python Installation** (3.10 or higher)
```windows cmd
py -m python --version
```

2. **Required Libraries**
```windows cmd
py -m pip install pandas numpy matplotlib seaborn scipy jupyter lab openpyxl scikit-learn lightgbm joblib streamlit
```

#### Step-by-Step Instructions

1. **Clone/Download Repository**
- Download the project folder
- Ensure all files are in the same directory

2. **Launch Jupyter Notebook**
```windows cmd
py -m jupyter lab
```

3. **Open the Notebook**
- Open `Spotify Music.ipynb` In Jupyter.

4. **Run all cells at once or Run cells individually**

#### Troubleshooting

**Issue**: "File not found" error
**Solution**: Ensure `278k_song_labelled.csv` is in the same directory as the notebook.

**Issue**: Import errors
**Solution**: Install missing libraries using `py -m pip install [library-name]`

5. **Run Streamlit App**
``` ```

## &#128193; Repository structure
```
Poshem_Sales_Project/
├──charts/ # Visualization charts
├── 278k_labelled_uri.csv
├── 278k_song_labelled.csv  #Dataset
├── README.md   #README  
├── Spotify Music.ipynb  #Main Project Workbook
├── app.py #App 
├── models.pkl #Trained models for app
└── scaler.pkl
```


## &#128222; Contact
- **Email**: ayomideeli2002@gmail.com
- **LinkedIn**: https://linkedin.com/in/ayomide-olatunde-2859141a8
- **GitHub**: https://github.com/mideolatunde

- **Institution**: Poshem Technologies Institute
- **Course**: Python Data Challenge
- **Project Supervisor**: Simon E. Akhamie, CEO



