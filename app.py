from flask import Flask, render_template, request
import pandas as pd
from models.classifier import model

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", title="Home")

@app.route("/form", methods=["GET", "POST"])
def form():
    if request.method == "POST":
        data = {}

        sepal_length = float(request.form.get('sepal_length'))
        sepal_width = float(request.form.get('sepal_width'))
        petal_length = float(request.form.get('petal_length'))
        petal_width = float(request.form.get('petal_width'))

        data['SepalLengthCm'] = sepal_length
        data['SepalWidthCm'] = sepal_width
        data['PetalLengthCm'] = petal_length
        data['PetalWidthCm'] = petal_width
        
        df = pd.DataFrame(data, index=[0])
        
        # méthode classement de notre modèle
        result, probabilities = model.classement(df)
        
        flowers ={
            "C'est un iris setosa": "./static/setosa.jpeg",
            "C'est un iris versicolor": "./static/versicolor.jpeg",
            "C'est un iris virginica": "./static/virginica.jpeg"
        }

        # Vérifiez si result est dans les clés de flowers
        image = flowers.get(result, "")
        
        return render_template("form.html", result=result, title="Result", image=image, probabilities=probabilities)

    else:
        return render_template("form.html", title="Form")

if __name__ == '__main__':
    app.run(debug=True, port=3000)
