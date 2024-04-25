from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Create Flask app
app = Flask(__name__)

# Loading DataFrame df_sampled here
df=pd.read_csv('searcheng_cleaned.csv')

# Initialize CountVectorizer
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_content'])

# Function to search subtitles
def search_subtitles(query, vectorizer, X):
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, X)
    top_indices = np.argsort(cosine_similarities[0])[::-1]
    top_matching_movies = df.iloc[top_indices]['name'].tolist()
    return top_matching_movies[:10]

# Route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for handling the search query
@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    top_matching_movies = search_subtitles(query, vectorizer, X)
    return render_template('output.html', movies=top_matching_movies)

# Route for accessing subtitles of a specific movie
@app.route('/subtitle/<movie_name>')
def get_subtitle(movie_name):
    movie_row = df[df['name'] == movie_name].iloc[0]
    subtitle = movie_row['cleaned_content']
    return subtitle

if __name__ == '__main__':
    app.run(debug=True)