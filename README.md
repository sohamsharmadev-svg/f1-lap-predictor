# 🏎️ F1 Lap Time Predictor

A machine learning web app that predicts **Formula 1 lap times** using real race data from 2011–2020.

Built with **XGBoost**, **SHAP explainability**, and deployed via **Streamlit**.

---

## 🚀 Live Demo
👉 **[https://f1-lap-predictor-ml.streamlit.app/](https://f1-lap-predictor-ml.streamlit.app/)**

---

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| Mean Absolute Error | 4.37 seconds |
| R² Score | 0.7266 |
| Training samples | 470,000+ laps |

---

## ✨ Features
- 🔮 **Predict lap times** by selecting driver, country, lap number & race round
- 🧠 **SHAP explainability** — understand *why* the model made each prediction
- 📈 **Historical comparison** — see how your prediction compares to real lap data
- 🎛️ **Interactive dashboard** — built with Streamlit, no setup needed

---

## 🛠️ Tech Stack
- **Python** — core language
- **XGBoost** — regression model
- **SHAP** — model explainability
- **Streamlit** — web dashboard
- **Pandas / NumPy** — data processing
- **Scikit-learn** — model evaluation

---

## 📁 Project Structure
```
f1-lap-predictor/
├── app.py              # Streamlit dashboard
├── prepare_data.py     # Data cleaning & merging
├── train_model.py      # Model training
├── f1_cleaned.csv      # Cleaned dataset
├── model.pkl           # Trained XGBoost model
├── le_driver.pkl       # Driver label encoder
├── le_country.pkl      # Country label encoder
└── requirements.txt    # Dependencies
```

---

## ⚙️ Run Locally

```bash
git clone https://github.com/sohamsharmadev-svg/f1-lap-predictor.git
cd f1-lap-predictor
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## 📦 Data Source
[Formula 1 World Championship Dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) — Kaggle
