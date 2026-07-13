from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
# Load the trained model
model = joblib.load("model/wine_model.pkl")
@app.route("/")
def home():
    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)