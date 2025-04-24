from flask import Flask
from flask import request
from flask import render_template

import os
import json

import joblib


class Model:

    def __init__(self):
        # Load the model
        # https://www.kaggle.com/datasets/subhajournal/phishingemails?resource=download
        self.model = joblib.load('./assets/phishing_model.pkl')

    def predict(self, input):
        # Make a prediction
        prediction = self.model.predict([input])[0]
        return prediction


model = Model()

app = Flask(__name__)
app.model = model


@app.route('/', methods=['GET'])
def index():
    user_input = request.args.get(
        'input',
        default='',
        type=str,
    )
    prediction = app.model.predict(user_input)

    tfidif_vector = app.model.model['tfidf'].transform([user_input])

    # Get the feature names
    feature_names = app.model.model['tfidf'].get_feature_names_out()

    # Get the tf-idf scores for the input
    tfidf_scores = tfidif_vector.toarray()[0]

    # Create a dictionary of feature names and their corresponding tf-idf scores
    tfidf_dict = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names))}

    # Sort the dictionary by tf-idf scores in descending order
    sorted_tfidf_dict = dict(sorted(tfidf_dict.items(), key=lambda item: item[1], reverse=True))

    # Get the top 10 features
    top_10_features = dict(list(sorted_tfidf_dict.items())[:min(10, len(user_input.split()))])

    output = {
        'input': user_input,
        'prediction': prediction,
        'reason': "The top 10 features that contributed to the prediction are: " + str(top_10_features),
    }
    return app.response_class(
        response=json.dumps(output),
        status=200,
        mimetype='application/json',
    )


def run():
    if not os.path.exists(app.static_folder):
        print(f"WARNING: static folder {app.static_folder} not found")
        print("Current working directory:", os.getcwd())
        print("Directory contents:", os.listdir(os.getcwd()))

    app.run(debug=True)


if __name__ == "__main__":
    run()
