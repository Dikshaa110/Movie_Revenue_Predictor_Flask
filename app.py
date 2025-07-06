from flask import Flask, render_template, request
import pandas as pd
import pickle           # ✅ Used to load model_features.pkl
import cloudpickle      # ✅ Used to load your .pkl model

app = Flask(__name__)

# ✅ Load the trained model (saved using cloudpickle)
with open("movie_revenue_model.pkl", "rb") as f:
    model = cloudpickle.load(f)

# ✅ Load the feature column names used during training
with open('model_features.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        budget = float(request.form['budget'])
        popularity = float(request.form['popularity'])
        runtime = float(request.form['runtime'])
        rating = float(request.form['vote_average'])
        vote_count = int(request.form['vote_count'])
        num_genres = int(request.form['num_genres'])
        num_keywords = int(request.form['num_keywords'])
        num_companies = int(request.form['num_companies'])
        num_countries = int(request.form['num_countries'])
        num_languages = int(request.form['num_languages'])
        original_language = request.form['original_language']

        # Initialize all features to 0
        input_dict = dict.fromkeys(feature_columns, 0)

        # Set numeric features
        input_dict['budget'] = budget
        input_dict['popularity'] = popularity
        input_dict['runtime'] = runtime
        input_dict['vote_average'] = rating
        input_dict['vote_count'] = vote_count
        input_dict['num_genres'] = num_genres
        input_dict['num_keywords'] = num_keywords
        input_dict['num_companies'] = num_companies
        input_dict['num_countries'] = num_countries
        input_dict['num_languages'] = num_languages

        # Set one-hot encoded language column if applicable
        lang_col = f'original_language_{original_language}'
        if lang_col in input_dict:
            input_dict[lang_col] = 1

        # Convert to DataFrame and predict
        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]

        print("Input features:", input_df.to_dict(orient='records'))
        print(f"Predicted Revenue: ${prediction:,.2f}")

        return render_template('form.html', prediction=round(prediction, 2))

    except Exception as e:
        print("Prediction Error:", str(e))
        return render_template('form.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
