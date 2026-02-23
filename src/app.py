from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("amazon_rating_prediction_model.pkl")


@app.route("/")
def home():
    return render_template("home.html")
@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/predict-page")
def predict_page():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")
        category = request.form.get("category")

        previous_sales = float(request.form.get("previous_sales", 0))
        price = float(request.form.get("price", 0))
        discount = float(request.form.get("discount", 0))
        marketing = float(request.form.get("marketing", 0))

        prediction = model.predict([[previous_sales, price, discount, marketing]])
        prediction_value = round(prediction[0] * 1000, 2)

        return render_template(
            "predict.html",
            prediction=prediction_value,
            previous_sales=previous_sales,
            price=price,
            discount=discount,
            marketing=marketing,
            start_date=start_date,
            end_date=end_date,
            category=category
        )

    except Exception as e:
        return render_template("predict.html", prediction=str(e))



   



if __name__ == "__main__":
    app.run(debug=True)
