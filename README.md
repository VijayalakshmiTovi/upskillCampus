🌾 Prediction of Agriculture Crop Production in India

📌 Project Overview

**Prediction of Agriculture Crop Production in India** is a Machine Learning project developed to estimate agricultural crop production based on different agricultural factors such as State, District, Crop Year, Season, Crop and Cultivated Area.

The project uses a Random Forest Regression algorithm to predict crop production and provides an interactive web application developed using **Streamlit**.

The main objective is to demonstrate how Machine Learning can be applied to agricultural data analysis and crop production prediction.

## 🎯 Objectives

- Analyse agricultural crop production data.
- Understand the relationship between crop production and different agricultural factors.
- Preprocess the dataset for Machine Learning.
- Develop a prediction model using Random Forest Regression.
- Evaluate model performance using MAE, RMSE and R².
- Build an interactive web application for crop production prediction.
- Display data analysis and model performance through visualisations.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning model |
| Random Forest Regressor | Crop production prediction |
| Streamlit | Web application |
| Plotly | Interactive visualisations |
| Matplotlib | Data visualisation |
| Joblib | Saving and loading the trained model |

## 📂 Project Structure

```text
Agriculture-Crop-Production-Prediction/
│
├── app.py
├── requirements.txt
│
├── data/
│   └── crop_production_starter_dataset.csv
│
├── models/
│   ├── crop_model.joblib
│   └── metrics.json
│
└── README.md
```

## 📊 Dataset Description

The dataset currently contains **60 sample records and 7 columns**.

| Column | Description |
|---|---|
| State | State name |
| District | District name |
| Crop_Year | Year of crop production |
| Season | Agricultural season |
| Crop | Crop name |
| Area | Cultivated area |
| Production | Crop production (target variable) |

**Note:** The current dataset is a small illustrative sample dataset used for project development and demonstration. It is not an official or real-time agricultural dataset.

## ⚙️ Methodology

The project follows these steps:

1. **Data Collection:** Load the agricultural crop production CSV dataset.
2. **Data Preprocessing:** Inspect and prepare the dataset for model training.
3. **Feature Selection:** Use State, District, Crop Year, Season, Crop and Area as input features.
4. **Model Training:** Train a Random Forest Regression model.
5. **Model Evaluation:** Evaluate the model using MAE, RMSE and R².
6. **Model Saving:** Save the trained model using Joblib.
7. **Web Application:** Integrate the model into a Streamlit application.

## 🤖 Machine Learning Algorithm

### Random Forest Regression

Random Forest Regression is an ensemble machine learning algorithm that combines multiple decision trees to produce a numerical prediction.

In this project, it is used to estimate crop production based on the selected agricultural input features.

## 📈 Model Performance

The current model evaluation results are:

| Evaluation Metric | Result |
|---|---:|
| MAE | 108681.79 |
| RMSE | 129101.07 |
| R² Score | 0.447 |

These results are preliminary and are based on the small illustrative sample dataset. They should not be interpreted as verified real-world prediction accuracy.

## 🌐 Web Application Features

The Streamlit application provides the following features:

- **Home:** Introduction and overview of the project.
- **Predict Production:** Enter agricultural details and obtain an estimated crop production value.
- **Data Analysis:** Explore dataset information through charts and visualisations.
- **Model Performance:** View the model evaluation metrics.
- **About:** Information about the project.

## 💻 Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/VijayalakshmiTovi/upskillCampus.git
```

### 2. Open the project folder

```bash
cd upskillCampus
```

### 3. Install the required libraries

```bash
py -m pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
py -m streamlit run app.py
```

### 5. Open the application

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 🔮 Future Enhancements

- Use larger and verified agricultural datasets from official sources.
- Include more crops, states, districts and historical years.
- Add rainfall, temperature and soil-related features where reliable data is available.
- Compare different machine learning algorithms.
- Improve prediction reliability through additional validation.
- Add downloadable prediction reports.
- Improve the user interface and application features.

## 🎓 Project Information

**Project Title:** Prediction of Agriculture Crop Production in India

**Student Name:** Vijayalakshmi

**Project Domain:** Data Science and Machine Learning

**Machine Learning Algorithm:** Random Forest Regression

**Frontend / Web Framework:** Streamlit

---

## 👩‍💻 Author

**Vijayalakshmi

Machine Learning | Data Science | Python

## 📌 Conclusion

This project demonstrates the application of Machine Learning in agriculture by developing a prototype for crop production prediction. It combines data preprocessing, model training, model evaluation and an interactive Streamlit application.

The project can be further improved using larger, verified agricultural datasets and additional environmental factors to support more reliable crop production analysis.
