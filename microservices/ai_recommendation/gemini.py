from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import requests
import random
import logging
import os
import json
from pymongo import MongoClient
import urllib.parse
import google.generativeai as genai
from collections import Counter
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration from .env file
CONTENT_API_BASE_URL = "http://34.47.135.240:8080/content-aggregation"
GEMINI_API_KEY = "AIzaSyB91dDRx0apjiCbYTQyxaunrk-0u7Gprbs"

# MongoDB connection without authentication
try:
    # Try without authentication first
    mongo_uri_options = [
        "mongodb://firetv:password@34.47.135.240:27017"
    ]

    mongo_client = None
    for uri in mongo_uri_options:
        try:
            test_client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            test_client.admin.command('ping')
            mongo_client = test_client
            break
        except Exception as e:
            continue

    if not mongo_client:
        mongo_client = None

except Exception as e:
    mongo_client = None

# Database and collection setup
user_interactions_collection = None
if mongo_client is not None:
    db = mongo_client["firetv_content"]
    user_interactions_collection = db['user_interaction']
    print("user interaction collection:")
    print(user_interactions_collection)
else:
    db = None
    user_interactions_collection = None

# Initialize Gemini AI
# Initialize Gemini AI
if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-pro')
    except Exception as e:
        gemini_model = None
else:
    gemini_model = None


# Time-based genre mapping (4 periods as requested)
TIME_PERIODS = {
    "morning": {"start": 6, "end": 12, "genres": ["comedy", "family", "animation", "romance"]},
    "afternoon": {"start": 12, "end": 18, "genres": ["action", "adventure", "documentary", "drama"]},
    "evening": {"start": 18, "end": 22, "genres": ["drama", "romance", "thriller", "mystery"]},
    "night": {"start": 22, "end": 6, "genres": ["horror", "sci-fi", "thriller", "mystery"]}
}

# Pydantic models
class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: int = 10

class MovieRecommendation(BaseModel):
    content_id: str
    movie_name: str
    match_rate: float

class RecommendationResponse(BaseModel):
    recommendations: List[MovieRecommendation]
    total: int
    strategy: str
    timestamp: str

class UserInteraction(BaseModel):
    user_id: str
    event_type: str  # watch, like, dislike, click
    content_type: str  # movie, tvshow
    content_id: str
    content_platform: str  # netflix, primevideo, hotstar
    context_data: Dict[str, Any]
    watch_progress: Optional[float] = None
    timestamp: Optional[datetime] = None

# Sample user interaction data for fallback when MongoDB is not available
SAMPLE_USER_DATA = {
    "01": [
        {
            "user_id": "01",
            "event_type": "watch",
            "content_id": "tmdb_movie_1233413",
            "content_platform": "netflix",
            "context_data": {
                "title": "Sinners",
                "rating": 7.6,
                "mood_tags": ["scary", "edge-of-seat", "suspenseful"],
                "genres": ["horror", "thriller"],
                "release_date": "2025-04-16"
            },
            "watchProgress": 0.85,
            "timestamp": datetime.now(timezone.utc)
        },
        {
            "user_id": "01",
            "event_type": "like",
            "content_id": "tmdb_movie_456",
            "content_platform": "primevideo",
            "context_data": {
                "title": "Action Hero",
                "rating": 8.2,
                "mood_tags": ["exciting", "adrenaline"],
                "genres": ["action", "adventure"],
                "release_date": "2025-03-10"
            },
            "timestamp": datetime.now(timezone.utc)
        }
    ]
}

def get_current_time_period() -> str:
    """Get current time period based on hour (4 periods as requested)"""
    current_hour = datetime.now().hour

    for period, config in TIME_PERIODS.items():
        start, end = config["start"], config["end"]

        if period == "night":  # Handle night period (22-6)
            if current_hour >= start or current_hour < end:
                return period
        else:
            if start <= current_hour < end:
                return period

    return "evening"  # Default fallback

def get_random_genre_for_new_user() -> str:
    """Get random genre for new user based on current time (as requested)"""
    period = get_current_time_period()
    genres = TIME_PERIODS[period]["genres"]
    selected_genre = random.choice(genres)
    return selected_genre

def fetch_all_movies_from_apis() -> List[Dict]:
    """Fetch movies from local content aggregation APIs"""
    all_content = []

    # Your actual API endpoints
    platforms = [
        "http://34.47.135.240:8080/content-aggregation/getContentNetflix",
        "http://34.47.135.240:8080/content-aggregation/getContentPrime",
        "http://34.47.135.240:8080/content-aggregation/getContentHotstar",
    ]

    for platform_url in platforms:
        try:
            response = requests.get(platform_url, timeout=30)

            if response.status_code == 200:
                data = response.json()

                # Handle different response formats
                if isinstance(data, dict) and 'content' in data:
                    content_list = data['content']
                    all_content.extend(content_list)
                elif isinstance(data, list):
                    all_content.extend(data)

        except Exception as e:
            continue

    # Add fallback content if no external content available
    if not all_content:
        all_content = get_fallback_movies()

    return all_content



def get_fallback_movies() -> List[Dict]:
    """Fallback movies when external APIs are unavailable"""
    return [
        {
            "id": "movie_1", "title": "The Dark Knight", "genres": ["action", "thriller"],
            "mood_tags": ["intense", "gripping"], "rating": 9.0, "platform": "netflix"
        },
        {
            "id": "movie_2", "title": "Inception", "genres": ["sci-fi", "thriller"],
            "mood_tags": ["mind-bending", "complex"], "rating": 8.8, "platform": "primevideo"
        },
        {
            "id": "movie_3", "title": "The Conjuring", "genres": ["horror", "thriller"],
            "mood_tags": ["scary", "suspenseful"], "rating": 7.5, "platform": "hotstar"
        },
        {
            "id": "movie_4", "title": "Toy Story", "genres": ["animation", "family"],
            "mood_tags": ["heartwarming", "fun"], "rating": 8.3, "platform": "netflix"
        },
        {
            "id": "movie_5", "title": "The Notebook", "genres": ["romance", "drama"],
            "mood_tags": ["emotional", "romantic"], "rating": 7.8, "platform": "primevideo"
        },
        {
            "id": "movie_6", "title": "Avengers: Endgame", "genres": ["action", "adventure"],
            "mood_tags": ["epic", "heroic"], "rating": 8.4, "platform": "hotstar"
        },
        {
            "id": "movie_7", "title": "Interstellar", "genres": ["sci-fi", "drama"],
            "mood_tags": ["emotional", "epic"], "rating": 8.6, "platform": "netflix"
        },
        {
            "id": "movie_8", "title": "The Hangover", "genres": ["comedy"],
            "mood_tags": ["hilarious", "fun"], "rating": 7.7, "platform": "primevideo"
        }
    ]

def get_user_interactions(user_id: str) -> List[Dict]:
    """Get user interactions from MongoDB or fallback data"""
    try:
        if user_interactions_collection is not None:
            cursor = user_interactions_collection.find(
                {"user_id": user_id}
            ).sort("timestamp", -1).limit(100)

            interactions = list(cursor)
            return interactions
        else:
            # Fallback to sample data
            interactions = SAMPLE_USER_DATA.get(user_id, [])
            return interactions

    except Exception as e:
        # Return sample data as fallback
        return SAMPLE_USER_DATA.get(user_id, [])

def is_new_user(user_id: str) -> bool:
    """Check if user is new (less than 5 interactions)"""
    try:
        if user_interactions_collection is not None:
            print("tried")
            print("user_id: ",user_id)
            count = user_interactions_collection.count_documents({"user_id": user_id})
            is_new = count < 5
            return is_new
        else:
            # Fallback: check sample data
            fallback_count = len(SAMPLE_USER_DATA.get(user_id, []))
            return fallback_count < 5
    except Exception as e:
        return True

async def get_personal_recommendations(request):
    """Get personalized movie recommendations - USER INTERACTIONS HAVE ABSOLUTE PRIORITY"""
    try:

        # Always check user interactions first
        user_interactions = get_user_interactions(request.user_id)

        if user_interactions and len(user_interactions) > 0:
            # EXISTING USER: User interaction preferences have ABSOLUTE PRIORITY

            # Get all available movies from database only
            all_movies = fetch_all_movies_from_apis()

            # Extract user's preferred genres and moods from interactions
            user_genres = []
            user_moods = []
            for interaction in user_interactions:
                context = interaction.get('context_data', {})
                if context.get('genres'):
                    user_genres.extend(context['genres'])
                if context.get('mood_tags'):
                    user_moods.extend(context['mood_tags'])

            # Get most common genres and moods
            from collections import Counter
            genre_counts = Counter(user_genres)
            mood_counts = Counter(user_moods)

            preferred_genres = [genre for genre, count in genre_counts.most_common(5)]  # Increased to 5
            preferred_moods = [mood for mood, count in mood_counts.most_common(5)]     # Increased to 5


            # PRIORITY 1: Exact genre and mood match
            exact_match_movies = []
            for movie in all_movies:
                movie_genres = movie.get('genres', [])
                movie_moods = movie.get('mood_tags', [])

                if isinstance(movie_genres, str):
                    movie_genres = [movie_genres]
                if isinstance(movie_moods, str):
                    movie_moods = [movie_moods]

                # Check if movie matches user's preferences (both genre AND mood)
                genre_match = any(genre.lower() in [g.lower() for g in movie_genres] for genre in preferred_genres)
                mood_match = any(mood.lower() in [m.lower() for m in movie_moods] for mood in preferred_moods)

                if genre_match and mood_match:
                    exact_match_movies.append(movie)

            # PRIORITY 2: Genre match only (if not enough exact matches)
            genre_only_movies = []
            if len(exact_match_movies) < request.num_recommendations:
                for movie in all_movies:
                    if movie in exact_match_movies:
                        continue  # Skip already added movies

                    movie_genres = movie.get('genres', [])
                    if isinstance(movie_genres, str):
                        movie_genres = [movie_genres]

                    genre_match = any(genre.lower() in [g.lower() for g in movie_genres] for genre in preferred_genres)
                    if genre_match:
                        genre_only_movies.append(movie)

            # PRIORITY 3: Mood match only (if still not enough)
            mood_only_movies = []
            total_matched = len(exact_match_movies) + len(genre_only_movies)
            if total_matched < request.num_recommendations:
                for movie in all_movies:
                    if movie in exact_match_movies or movie in genre_only_movies:
                        continue  # Skip already added movies

                    movie_moods = movie.get('mood_tags', [])
                    if isinstance(movie_moods, str):
                        movie_moods = [movie_moods]

                    mood_match = any(mood.lower() in [m.lower() for m in movie_moods] for mood in preferred_moods)
                    if mood_match:
                        mood_only_movies.append(movie)

            # Combine recommendations in priority order
            recommended_movies = exact_match_movies + genre_only_movies + mood_only_movies

            # If we have recommendations based on user interactions
            if recommended_movies:
                # Select top recommendations
                num_to_select = min(request.num_recommendations, len(recommended_movies))
                selected_movies = recommended_movies[:num_to_select]  # Take in priority order

                recommendations = []
                for i, movie in enumerate(selected_movies):
                    # Higher match rate for exact matches
                    if movie in exact_match_movies:
                        match_rate = round(random.uniform(0.85, 0.95), 2)
                    elif movie in genre_only_movies:
                        match_rate = round(random.uniform(0.75, 0.85), 2)
                    else:
                        match_rate = round(random.uniform(0.65, 0.75), 2)

                    recommendations.append(MovieRecommendation(
                        content_id=movie.get('id', movie.get('_id', 'unknown')),
                        movie_name=movie.get('title', 'Unknown Title'),
                        match_rate=match_rate
                    ))

                strategy = "user_interaction_priority_based"

            else:
                # ONLY if NO movies match user preferences, then use any available content
                all_movies = fetch_all_movies_from_apis()

                num_to_select = min(request.num_recommendations, len(all_movies))
                if num_to_select > 0:
                    selected_movies = random.sample(all_movies, num_to_select)
                else:
                    selected_movies = []

                recommendations = []
                for movie in selected_movies:
                    recommendations.append(MovieRecommendation(
                        content_id=movie.get('id', movie.get('_id', 'unknown')),
                        movie_name=movie.get('title', 'Unknown Title'),
                        match_rate=round(random.uniform(0.5, 0.7), 2)
                    ))

                strategy = "fallback_all_content"

        else:
            # NEW USER: ONLY THEN use time-based recommendations
            selected_genre = get_random_genre_for_new_user()
            current_period = get_current_time_period()


            # Get all available movies from database only
            all_movies = fetch_all_movies_from_apis()

            # Filter by selected genre for this time period
            genre_movies = []
            for movie in all_movies:
                movie_genres = movie.get('genres', [])
                if isinstance(movie_genres, str):
                    movie_genres = [movie_genres]

                if any(selected_genre.lower() in genre.lower() for genre in movie_genres):
                    genre_movies.append(movie)

            # If no movies in selected genre, use all movies
            if not genre_movies:
                genre_movies = all_movies

            # Randomly select movies and assign random match rates
            num_to_select = min(request.num_recommendations, len(genre_movies))
            if num_to_select > 0:
                selected_movies = random.sample(genre_movies, num_to_select)
            else:
                selected_movies = []

            recommendations = []
            for movie in selected_movies:
                recommendations.append(MovieRecommendation(
                    content_id=movie.get('id', movie.get('_id', 'unknown')),
                    movie_name=movie.get('title', 'Unknown Title'),
                    match_rate=round(random.uniform(0.6, 0.9), 2)
                ))

            strategy = f"new_user_time_based_{current_period}_{selected_genre}"

        return {
                "recommendations":recommendations,
                "total":len(recommendations),
                "strategy":strategy,
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating recommendations")
