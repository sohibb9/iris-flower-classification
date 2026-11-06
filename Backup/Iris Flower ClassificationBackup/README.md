# Iris Flower Classification 🌸

A machine learning project that predicts the species of an Iris flower from four measurements using **Logistic Regression**. The project includes exploratory data analysis, model evaluation, saved preprocessing/model artifacts, and an interactive Streamlit application.

## Live Demo

> Add your deployed Streamlit app URL here after deployment.

## Project Overview

The application takes four measurements (in centimeters):

- Sepal length
- Sepal width
- Petal length
- Petal width

It standardizes the input with the saved `StandardScaler`, then uses a trained Logistic Regression model to predict one of three species:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

The app also displays the model's predicted-class confidence and the probability distribution across all three species.

## Model Performance

The notebook's recorded test accuracy is **93.33%** on a held-out test set. The confusion matrix recorded in the notebook is:

| Actual \\ Predicted | Setosa | Versicolor | Virginica |
|---|---:|---:|---:|
| Setosa | 10 | 0 | 0 |
| Versicolor | 0 | 9 | 1 |
| Virginica | 0 | 1 | 9 |

These results describe the experiment in the notebook; performance may vary with a different split or training setup.

## Tech Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn (exploratory analysis)
- Scikit-learn (StandardScaler and Logistic Regression)
- Joblib (model persistence)
- Streamlit (interactive web app)
- Jupyter Notebook

## Project Structure

```text
Iris Flower Classification/
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

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/iris-flower-classification.git
cd iris-flower-classification
```

### 2. Create and activate a virtual environment (recommended)

**Windows PowerShell:**

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

### 4. Run the app

```bash
streamlit run app.py
```

The app will open in your browser, usually at `http://localhost:8501`.

## Deployment on Streamlit Community Cloud

1. Push this project to a public GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with GitHub and select **Create app**.
4. Choose the repository, branch, and `app.py` as the main file.
5. Deploy. Keep `app.py`, `requirements.txt`, `iris_model.joblib`, and `iris_scaler.joblib` in the repository root.
6. Add the public app URL to the **Live Demo** section above.

## Notes

- The model and scaler files were saved using scikit-learn 1.7.2, so the requirements pin that version for compatibility.
- Only load `.joblib` files from sources you trust; serialized Python objects can execute code when loaded.
- This is an educational classification demo, not a botanical or scientific diagnostic tool.

## Author

**Sohib Mohamed**  
AI Engineering Student | Python | Data Analysis | Machine Learning

- GitHub: [sohibb9](https://github.com/sohibb9)
