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


## 📈 Exploratory Data Analysis  



### Accident Frequency Analysis
- Examined accident distribution over time (day, month, year) to identify high-risk periods.

### Demographic Insights 
- Analyze the relationship between driver characteristics (age, experience, gender, educational level) and accident occurrences.

### Environmental Factors
- Investigate the impact of weather, road surface conditions, road surface types and lightning on accident severity.

### Accident Severity Analysis
- Categorize accidents by severity levels (slight, serious, fatal).

### Common Causes of Accidents
- Determine primary factors leading to road accidents (Types of Collisions, types of junction, area of occurrence, types of vehicle).

### Correlation Analysis
-  Identify relationships between drivers age band and accident severity, with age band casuality and severity.

## &#127942; Model Performance



## 🔍 Key Insights

###  Accidents OverTime
- **Total Number of accidents**: 12,316 
- **Total Number of casualties**: 19,067 casualties
- **Total Number of vehicles involved**: 25,133 vehicles

### Peak accident periods
- **Most Frequent Accident Time**: 17:00
- **Most Frequent Day**: Friday

### Demographic insights
- **Highest Gender Involvement**: Male (11437, 92.86%), 
- **Highest Age Band Involvement**: 18-30, 4271
- **Highest driving experience Involvement**: Junior high school, 7619
- **Highest educational level Involvement**: 5-10yr, 3363 followed by 2-5yr, 2613

### Environmental Factors causing highest severity
- **Road surface type**: Asphalt road types 
- **Road surface condition**: Dry road condition followed by wet or damp roads
- **Light Condition**: Daylight condition
- **Weather Condition**: Normal weather condition

### Accident Severity Analysis
- **Slight Injury**: 10415 (84.56%)
- **Serious Injury**: 1743 (14.15%)
- **Fatal Injury**: 158 (1.28%)

### Common Causes of Accidents
- No distancing: 2263
- Changing lane to the right: 1808
- Changing lane to the left: 1473

### Other frequent causes
- **Types of Collision**: Vehicle with vehicle collision 
- **Types of junction**: Y-junction
- **Area of occurrence**: Office areas
- **Types of vehicle**: Automobile  

### Correlation Analysis
- Drivers age band has a correlation of 0.013 with Accident severity
- Casualty age band has a correlation of 0.77 with Casualty severity 

## &#128200; Visualizations
The analysis includes:

- Multiple Bar charts of the days, driver characteristics, environmental Information and other information (*check out charts*)
- A pie chart showing the different accident severity levels.
- Subplot of barcharts showing the number of accidents per driver's age, sex, education level and driving experience.
- Line plot showing the timely occurrence of accidents.
- Heat map showing the correlation between age band of driver and casualty and severity of driver and casualty.

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
- Python (Pandas, NumPy, Matplotlib, Seaborn, SciPy)
- Jupyter Notebook
- Data Visualization enhances clarity and business decision-making
- Statistical analysis methods

## &#128640; How to Run the Project
1. **Python Installation** (3.8 or higher)
```windows cmd
py -m python --version
```

2. **Required Libraries**
```windows cmd
py -m pip install pandas numpy matplotlib seaborn scipy jupyter lab openpyxl
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
- Open `RTA Project code.ipynb` In Jupyter.

4. **Run all cells at once or Run cells individually**

#### Troubleshooting

**Issue**: "File not found" error
**Solution**: Ensure `RTA Dataset_with_DD.xlsx` is in the same directory as the notebook.

**Issue**: Import errors
**Solution**: Install missing libraries using `py -m pip install [library-name]`


## &#128193; Repository structure
```
Poshem_Sales_Project/
├──charts/ # Visualization charts
├── RTA Accidents Report.docx
├── README.md # README
├── RTA Project code # Main project workbook
└── RTA Dataset_with_DD.xlsx  # Dataset
```


## &#128222; Contact
- **Email**: ayomideeli2002@gmail.com
- **LinkedIn**: https://linkedin.com/in/ayomide-olatunde-2859141a8
- **GitHub**: https://github.com/mideolatunde

- **Institution**: Poshem Technologies Institute
- **Course**: Python Data Challenge
- **Project Supervisor**: Simon E. Akhamie, CEO



