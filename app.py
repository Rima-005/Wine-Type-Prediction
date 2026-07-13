from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("model/wine_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [
        float(request.form["Alcohol"]),
        float(request.form["Malic_Acid"]),
        float(request.form["Ash"]),
        float(request.form["Alcalinity_of_Ash"]),
        float(request.form["Magnesium"]),
        float(request.form["Total_Phenols"]),
        float(request.form["Flavanoids"]),
        float(request.form["Nonflavanoid_Phenols"]),
        float(request.form["Proanthocyanins"]),
        float(request.form["Color_Intensity"]),
        float(request.form["Hue"]),
        float(request.form["OD280_OD315"]),
        float(request.form["Proline"])
    ]

    prediction = model.predict([features])

    return render_template("result.html", prediction=prediction[0])


if __name__ == "__main__":
    app.run(debug=True)