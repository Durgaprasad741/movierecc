import streamlit as st
import pandas as pd
import difflib as df
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Function to load the dataset
def load_data():
    # Load the movie dataset (Ensure it has the necessary columns)
    movies_data = pd.read_csv('movies.csv')
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    
    # Fill missing values with empty strings
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')
    
    # Combine selected features into a single string
    combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']
    return movies_data, combined_features

# Function to get movie poster URL from OMDb API
def get_movie_poster_url(movie_title):
    # OMDb API URL with your API key
    api_key = '6ba4b48e'  # Your OMDb API key
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    
    # Debugging: Print the raw response from OMDb to check for any issues
    print(f"OMDb Response for {movie_title}: {data}")
    
    if data.get('Response') == 'True':
        poster_url = data.get('Poster')
        if poster_url and poster_url != 'N/A':
            return poster_url  # Return the poster URL if available
        else:
            return None  # Poster URL is not available or 'N/A'
    else:
        return None  # If the movie was not found or an error occurred

# Function to get movie recommendations based on user input
def get_movie_recommendations(movie_name, movies_data, combined_features, num_recommendations=10):
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)

    # Lowercase the movie_name for matching
    movie_name = movie_name.lower()
    list_of_all_titles = [title.lower() for title in movies_data['title'].tolist()]
    
    find_close_match = df.get_close_matches(movie_name, list_of_all_titles)
    
    if not find_close_match:
        return []

    close_match = find_close_match[0]

    # Find the index of the matched movie
    index_of_the_movie = movies_data[movies_data.title.str.lower() == close_match]['index'].values[0]

    # Get similarity scores for all movies
    similarity_score = list(enumerate(similarity[index_of_the_movie]))

    # Sort movies based on similarity score in descending order
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # Collect top recommendations
    recommendations = []
    i = 1
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data.iloc[index]['title']
        poster_url = get_movie_poster_url(title_from_index)  # Fetch poster URL using OMDb API
        if i <= num_recommendations:
            recommendations.append((title_from_index, poster_url))
            i += 1
        else:
            break
    return recommendations

# Streamlit UI
def main():
    st.title('Movie Recommendation System')
    st.write('Enter the name of a movie, and get movie recommendations based on its features.')

    # Load data
    movies_data, combined_features = load_data()

    # User input for movie name
    movie_name = st.text_input('Enter your favourite movie name:')

    # User input for number of recommendations
    num_recommendations = st.slider('How many recommendations would you like?', min_value=1, max_value=20, value=10)

    if movie_name:
        # Get recommendations
        recommendations = get_movie_recommendations(movie_name, movies_data, combined_features, num_recommendations)

        # Display recommendations with poster images
        if recommendations:
            st.write(f"**Top {num_recommendations} recommended movies similar to '{movie_name.capitalize()}':**")
            for idx, (movie, poster_url) in enumerate(recommendations, 1):
                st.write(f"{idx}. {movie}")
                if poster_url:  # Check if poster URL exists
                    st.image(poster_url, width=200)  # Display the poster image with a fixed width
                else:
                    st.write("No poster available.")
        else:
            st.write("Sorry, no close match found. Please try again with another movie name.")
    
if __name__ == '__main__':
    main()
