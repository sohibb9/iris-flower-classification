# Iris Flower Classification 🌸

An end-to-end machine learning project that predicts an Iris flower species from four physical measurements. It includes exploratory data analysis, a trained Logistic Regression model, saved preprocessing artifacts, and an interactive Streamlit application.


## Demo

The app accepts four measurements in centimeters:

- Sepal length
- Sepal width
- Petal length
- Petal width

It applies the saved `StandardScaler`, predicts the flower species with the trained Logistic Regression model, and displays the predicted-class confidence and class probabilities.

## Predicted classes

- Iris Setosa
- Iris Versicolor
- Iris Virginica

## Model performance

The supplied notebook records **93.33% test accuracy** on its held-out test set. The recorded confusion matrix is:

| Actual \\ Predicted | Setosa | Versicolor | Virginica |
|---|---:|---:|---:|
| Setosa | 10 | 0 | 0 |
| Versicolor | 0 | 9 | 1 |
| Virginica | 0 | 1 | 9 |

This is the result recorded in the notebook; retraining with a different split or configuration may produce different results.

## Tech stack

- Python
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Repository structure

```text
iris-flower-classification/
├── app.py
├── Iris.ipynb
├── iris_model.joblib
├── iris_scaler.joblib
├── requirements.txt
├── .gitignore
├── README.md
├── Feature Distributions.png
└── Feature Distributions  sns.png
```

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/sohibb9/iris-flower-classification.git
cd iris-flower-classification
```

### 2. Create and activate a virtual environment

**Windows:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the app

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Select **Create app**.
4. Choose repository `sohibb9/iris-flower-classification`, branch `main`, and main file `app.py`.
5. Deploy. Keep `app.py`, `requirements.txt`, `iris_model.joblib`, and `iris_scaler.joblib` in the repository root.

## Notes

- Scikit-learn is pinned to `1.7.2` to match the version documented when the model artifacts were saved.
- Only load Joblib files from sources you trust: serialized Python objects can execute code when loaded.
- This project is an educational machine learning demonstration.

## Author

**Sohib Mohamed**  
AI Engineering Student | Python | Data Analysis | Machine Learning

GitHub: [@sohibb9](https://github.com/sohibb9)
