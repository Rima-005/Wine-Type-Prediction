# 🍷 Wine Type Prediction using Machine Learning

## 📌 Project Overview

The Wine Type Prediction System is a Machine Learning web application that predicts the type of wine based on its chemical properties. The model is trained using the Wine dataset and deployed using the Flask framework.

---

## 🚀 Features

- Predicts wine type using Machine Learning.
- Trained on the Wine dataset.
- Compares multiple ML algorithms.
- Uses the best-performing Random Forest model.
- Simple and user-friendly Flask web interface.

---

## 🛠️ Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- HTML
- CSS

---

## 📂 Project Structure

```
Wine-Type-Prediction/
│── dataset/
│   └── wine_clean.csv
│
│── model/
│   └── wine_model.pkl
│
│── static/
│   └── style.css
│
│── templates/
│   ├── index.html
│   └── result.html
│
│── app.py
│── train_model.py
│── requirements.txt
│── README.md
```

---

## 🤖 Machine Learning Models Used

- Logistic Regression
- Decision Tree
- Random Forest ✅
- Support Vector Machine (SVM)

### Model Accuracy

| Model | Accuracy |
|--------|---------:|
| Logistic Regression | 97.22% |
| Decision Tree | 94.44% |
| Random Forest | **100.00%** |
| SVM | 80.56% |

Random Forest achieved the highest accuracy and was selected as the final model.

---

## ▶️ How to Run

1. Clone the repository

```bash
git clone <repository-link>
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Train the model (optional)

```bash
python train_model.py
```

4. Run the Flask application

```bash
python app.py
```

5. Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 📷 Application Workflow

1. Enter the chemical properties of the wine.
2. Click the **Predict** button.
3. The model predicts the wine class.
4. The prediction is displayed on the result page.

---

## 👩‍💻 Author

**Rima**

Machine Learning Internship Project