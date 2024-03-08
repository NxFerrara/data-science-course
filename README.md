# Data Science Course Capstone Project

This project aims to explore the relationships between sensation seeking, movie experiences, personality types, and more, through data-driven analysis.

## Project Overview

In this capstone, we delve into a dataset containing ratings for 400 movies from 1097 research participants, alongside self-assessments on sensation seeking behaviors, responses to personality questions, self-reported movie experience ratings, and demographic data. This project aims to uncover hidden patterns and relationships within the complex dataset using statistical and machine learning techniques.

## Installation and Setup

To run the analysis, ensure you have Python installed on your system. The project utilizes NumPy, Pandas, Matplotlib, Scikit-Learn, and SciPy. You can install all required packages using:

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

## Data Description

The dataset, `movieReplicationSet.csv`, features:

- **Movie Ratings:** Ratings for 400 movies (0 to 4, with missing data).
- **Sensation Seeking:** Self-assessments on 21 sensation seeking behaviors (1-5).
- **Personality Questions:** Responses to 43 personality questions (1-5).
- **Movie Experience:** Self-reported ratings on movie experiences (1-5).
- **Demographics:** Information on gender identity, only child status, and social viewing preferences.

## Analysis Workflow

The analysis comprises the following key steps:

1. **Data pre-processing**: Import the dataset and process it for analysis.
2. **Data Analysis**: Perform correlation analysis, Principal Component Analysis (PCA), and hypothesis testing to uncover relationships within the data.
3. **Machine Learning**: Utilize techniques like KMeans Clustering and Random Forest Regression to predict outcomes based on personality factors.
4. **Evaluate**: Validate the models and make final conclusions about the dataset.
 
## Key Findings

- Exploration of relationships between sensation seeking and movie experience.
- Identification of distinct personality types based on PCA.
- Analysis of gender differences in movie ratings, particularly for 'Shrek (2001)'.
- Investigation into whether being an only child affects enjoyment of 'The Lion King (1994)'.
- Examination of viewer preferences for watching movies alone or socially.
