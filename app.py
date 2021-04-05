import numpy as np 
from flask import Flask, render_template, jsonify, request
import pickle

app = Flask(__name__)
model = pickle.load(open('Jupyter Notebook\Stroke Predictor\stroke-predictor.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    web_prediction = model.predict(final_features)

    output = web_prediction[0]
    if output == 0:
        return render_template('predict.html', prediction_text='Person is more likely to not be affected by a Heart Stroke')
    else:
        return render_template('predict_precaution.html', prediction_text='Person is more likely to be affected by a Heart Stroke')

@app.route('/about-us')
def aboutUS():
    return render_template('about-us.html')

@app.route('/precaution')
def precaution():
    return render_template('precaution.html')

if __name__ == "__main__":
    app.run(debug=True)