# microservices/ai-recommendation-service/temporal_recommendation_engine.py
import numpy as np
import asyncio
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import pymongo
import math

mongo_url = "mongodb://firetv:password@mongo:27017"
client = pymongo.MongoClient(mongo_url)

class TemporalPreferenceLearner:
    def __init__(self, db):
        self.db = db
        self.user_temporal_preferences = {}
        
    def learn_user_temporal_preferences(self, user_id: str) -> Dict[str, Dict[str, float]]:
        """Learn user's temporal genre preferences from their viewing history"""
        try:
            # Get user's viewing history with timestamps
            user_interactions = list(self.db.get_collection("user_interaction").find({
                "user_id": user_id,
                "event_type": {"$in": ["watch", "like"]}  # Positive interactions only
            }).limit(200))
            
            if not user_interactions:
                return self._get_default_temporal_preferences()
            
            # Initialize temporal genre counters
            temporal_genre_scores = {
                'morning': defaultdict(float),
                'afternoon': defaultdict(float), 
                'evening': defaultdict(float),
                'night': defaultdict(float)
            }
            
            temporal_counts = {
                'morning': 0, 'afternoon': 0, 'evening': 0, 'night': 0
            }
            
            for interaction in user_interactions:
                timestamp = interaction['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                time_window = self._get_time_window(timestamp)
                
                content = self.db.get_collection("content").find_one({"id": interaction['content_id']})
                if not content:
                    continue
                
                interaction_weight = self._calculate_interaction_weight(interaction)
                
                genres = content.get('genres', [])
                for genre in genres:
                    temporal_genre_scores[time_window][genre.lower()] += interaction_weight
                
                temporal_counts[time_window] += 1
            
            # Normalize scores to get preferences (0-1 scale)
            normalized_preferences = {}
            for time_window, genre_scores in temporal_genre_scores.items():
                if temporal_counts[time_window] == 0:
                    normalized_preferences[time_window] = {}
                    continue
                
                # Convert to preference scores (higher = more preferred)
                total_score = sum(genre_scores.values())
                if total_score > 0:
                    normalized_preferences[time_window] = {
                        genre: score / total_score 
                        for genre, score in genre_scores.items()
                    }
                else:
                    normalized_preferences[time_window] = {}
            
            # Cache the learned preferences
            self.user_temporal_preferences[user_id] = normalized_preferences
            
            print(f"Learned temporal preferences for user {user_id}")
            return normalized_preferences
            
        except Exception as e:
            print(f"Error learning temporal preferences: {e}")
            return self._get_default_temporal_preferences()
    
    def _calculate_interaction_weight(self, interaction: Dict) -> float:
        """Calculate weight for interaction based on type and engagement"""
        event_type = interaction['event_type']
        watch_progress = interaction.get('watchProgress', 0.0)
        
        # Base weights
        base_weights = {
            'watch': watch_progress,  # Weight by completion
            'like': 1.0,
            'dislike': -1.0,
        }
        
        base_weight = base_weights.get(event_type, 0.5)
        
        # Time decay - recent interactions matter more
        timestamp = interaction['timestamp']
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        days_ago = (datetime.utcnow() - timestamp.replace(tzinfo=None)).days
        time_decay = np.exp(-days_ago / 30.0)  # 30-day half-life
        
        return base_weight * max(0.1, time_decay)
    
    def _get_time_window(self, timestamp: datetime) -> str:
        """Get time window from timestamp"""
        hour = timestamp.hour
        
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    def _get_default_temporal_preferences(self) -> Dict[str, Dict[str, float]]:
        """Return empty preferences for new users"""
        return {
            'morning': {},
            'afternoon': {},
            'evening': {},
            'night': {}
        }
    
    def get_temporal_genre_boost(self, user_id: str, content_genres: List[str], 
                               current_time_window: str) -> float:
        """Get personalized temporal boost for content based on learned preferences"""
        try:
            # Get or learn user preferences
            if user_id not in self.user_temporal_preferences:
                self.learn_user_temporal_preferences(user_id)
            
            user_prefs = self.user_temporal_preferences.get(user_id, {})
            time_prefs = user_prefs.get(current_time_window, {})
            
            if not time_prefs:
                return 1.0  # No boost if no preferences learned
            
            # Calculate boost based on genre overlap
            genre_boosts = []
            for genre in content_genres:
                genre_lower = genre.lower()
                preference_score = time_prefs.get(genre_lower, 0.0)
                
                # Convert preference score to boost (1.0 - 1.5x)
                boost = 1.0 + (preference_score * 0.5)
                genre_boosts.append(boost)
            
            # Return average boost
            return np.mean(genre_boosts) if genre_boosts else 1.0
            
        except Exception as e:
            print(f"Error calculating temporal boost: {e}")
            return 1.0

class TemporalRecommendationEngine:
    def __init__(self):
        self.db = client.get_database("firetv_content")
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.content_vectors = None
        self.content_ids = []
        self.preference_learner = TemporalPreferenceLearner(self.db)

        # Event weights for building user preferences
        self.event_weights = {
            'click': 0.2,
            'like': 0.8,
            'dislike': -0.7,
            'watch': 1.0,
        }
        
        # Time window definitions
        self.time_windows = {
            'morning': (6, 12),
            'afternoon': (12, 17),
            'evening': (17, 22),
            'night': (22, 6)
        }

    def initialize(self):
        """Initialize the recommendation engine"""
        try:
            all_content = list(self.db.get_collection("content").find())

            content_texts = []
            self.content_ids = []

            for content in all_content:
                text_features = [
                    content.get('title', ''),
                    ' '.join(content.get('genres', [])),
                    ' '.join(content.get('mood_tags', [])),
                    content.get('overview', ''),
                    content.get('platform', '')
                ]

                content_text = ' '.join(filter(None, text_features))
                content_texts.append(content_text)
                self.content_ids.append(str(content['id']))

            self.content_vectors = self.vectorizer.fit_transform(content_texts)
            print(f"Initialized temporal recommendation engine with {len(self.content_ids)} content items")

        except Exception as e:
            print(f"Error initializing recommendation engine: {e}")
            raise

    def get_time_window(self, timestamp: datetime = None) -> str:
        """Get current time window"""
        if timestamp is None:
            timestamp = datetime.now()
        
        hour = timestamp.hour
        
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'

    def calculate_time_decay(self, timestamp: datetime, decay_rate: float = 0.1) -> float:
        """Calculate exponential time decay factor"""
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        # Calculate days since interaction
        days_ago = (datetime.utcnow() - timestamp.replace(tzinfo=None)).days
        
        decay_factor = math.exp(-decay_rate * days_ago)
        return max(0.01, decay_factor)

    def calculate_temporal_interaction_score(self, event_type: str, watch_progress: float, 
                                           timestamp: datetime, current_time_window: str) -> float:
        """Enhanced interaction score with temporal factors"""
        # Base interaction score
        base_score = self.calculate_interaction_score(event_type, watch_progress)
        
        # Time decay factor
        time_decay = self.calculate_time_decay(timestamp)
        
        # Time window relevance boost
        interaction_time_window = self.get_time_window(timestamp)
        time_window_boost = 1.2 if interaction_time_window == current_time_window else 1.0
        
        # Combine all factors
        final_score = base_score * time_decay * time_window_boost
        return final_score

    def calculate_interaction_score(self, event_type: str, watch_progress: float = 0.0) -> float:
        """Calculate weighted score for user interaction"""
        base_weight = self.event_weights.get(event_type, 0.0)

        if event_type == 'watch':
            return base_weight * watch_progress
        else:
            return base_weight

    def get_temporal_user_profile_vector(self, user_id: str, current_time_window: str = None):
        """Create time-aware user profile vector"""
        try:
            # Get current time window if not provided
            if current_time_window is None:
                current_time_window = self.get_time_window()
            
            # Get user interactions with timestamps
            user_interactions = list(self.db.get_collection("user_interaction").find({
                "user_id": user_id
            }).sort("timestamp", -1).limit(100))

            if not user_interactions:
                return None, {}

            # Build temporal content scores
            content_scores = {}
            time_window_scores = {window: 0 for window in self.time_windows.keys()}

            for interaction in user_interactions:
                content_id = interaction['content_id']
                event_type = interaction['event_type']
                watch_progress = interaction.get('watchProgress', 0.0)
                timestamp = interaction['timestamp']

                # Calculate temporal interaction score
                temporal_score = self.calculate_temporal_interaction_score(
                    event_type, watch_progress, timestamp, current_time_window
                )

                # Track time window preferences
                interaction_window = self.get_time_window(timestamp)
                time_window_scores[interaction_window] += temporal_score

                # Accumulate content scores
                if content_id in content_scores:
                    content_scores[content_id] += temporal_score
                else:
                    content_scores[content_id] = temporal_score

            # Filter positive content
            positive_content = {cid: score for cid, score in content_scores.items() if score > 0}

            if not positive_content:
                return None, time_window_scores

            # Get content details for positively-scored items
            positive_content_ids = list(positive_content.keys())
            liked_content = list(self.db.get_collection("content").find({
                "id": {"$in": positive_content_ids}
            }))

            # Create weighted text representation with temporal boosting
            weighted_texts = []
            for content in liked_content:
                content_id = str(content['id'])
                content_score = positive_content[content_id]

                # Create text representation
                text_features = [
                    content.get('title', ''),
                    ' '.join(content.get('genres', [])),
                    ' '.join(content.get('mood_tags', [])),
                    content.get('overview', ''),
                    content.get('platform', '')
                ]
                content_text = ' '.join(filter(None, text_features))

                # Apply temporal weighting
                repetitions = max(1, int(content_score * 5))  # Increased scale factor
                weighted_texts.extend([content_text] * repetitions)

            # Combine all weighted preferences
            combined_user_text = ' '.join(weighted_texts)
            user_vector = self.vectorizer.transform([combined_user_text])

            return user_vector, time_window_scores

        except Exception as e:
            print(f"Error creating temporal user profile vector: {e}")
            return None, {}

    def get_temporal_recommendations(self, user_id: str, num_recommendations: int = 10,
                                   time_context: str = None) -> List[Dict]:
        """Get time-aware recommendations"""
        try:
            # Determine current time context
            if time_context is None:
                time_context = self.get_time_window()

            # Get temporal user profile
            profile_result = self.get_temporal_user_profile_vector(user_id, time_context)

            if profile_result[0] is None:
                return self.get_popular_content(num_recommendations)

            user_vector, time_window_scores = profile_result

            # Calculate cosine similarity
            similarities = cosine_similarity(user_vector, self.content_vectors).flatten()

            # Get user interaction history for filtering
            user_interactions = list(self.db.get_collection("user_interaction").find({
                "user_id": user_id
            }))

            interacted_content = set()
            disliked_content = set()

            for interaction in user_interactions:
                content_id = interaction['content_id']
                interacted_content.add(content_id)

                if interaction['event_type'] in ['dislike', 'skip']:
                    disliked_content.add(content_id)
                elif interaction['event_type'] == 'watch' and interaction.get('watchProgress', 0) < 0.1:
                    disliked_content.add(content_id)

            # Generate recommendations with temporal scoring
            recommendations = []
            similar_indices = np.argsort(similarities)[::-1]

            for idx in similar_indices:
                content_id = self.content_ids[idx]

                if content_id in interacted_content or content_id in disliked_content:
                    continue

                content = self.db.get_collection('content').find_one({"id": content_id})
                if content:
                    base_similarity = float(similarities[idx])

                    # Apply temporal context boost using learned preferences
                    temporal_boost = self.preference_learner.get_temporal_genre_boost(
                        user_id, content.get('genres', []), time_context
                    )
                    final_score = base_similarity * temporal_boost

                    recommendations.append({
                        'content_id': content_id,
                        'title': content.get('title', ''),
                        'platform': content.get('platform', ''),
                        'genres': content.get('genres', []),
                        'rating': content.get('rating', 0),
                        'similarity_score': final_score,
                        'base_similarity': base_similarity,
                        'temporal_boost': temporal_boost,
                        'time_context': time_context,
                        'reason': self._generate_temporal_reason(user_id, content, time_context, final_score)
                    })

                if len(recommendations) >= num_recommendations:
                    break

            # Sort by final temporal score
            recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
            print_rec = recommendations[:num_recommendations]
            user_recommendations = self.db.get_collection(f"user_{user_id}_recommendations")
            user_recommendations.delete_many({})
            user_recommendations.insert_many(print_rec)
            # self.db.get_collection("user_recommendations").delete_many({'user_id':user_id})
            # self.db.get_collection("user_recommendations").insert_many(print_rec)
            
            # Save results to file
            # json_data = json.dumps(recommendations[:num_recommendations], indent=2)
            # with open('temporal_recommendations.json', 'w') as f:
            #     f.write(json_data)
                
            return recommendations[:num_recommendations]

        except Exception as e:
            print(f"Error getting temporal recommendations: {e}")
            return []

    def _generate_temporal_reason(self, user_id: str, content: Dict, 
                                time_context: str, score: float) -> str:
        """Generate personalized temporal explanation"""
        # Get user's learned preferences for this time
        user_prefs = self.preference_learner.user_temporal_preferences.get(user_id, {})
        time_prefs = user_prefs.get(time_context, {})
        
        if time_prefs:
            # Find matching genres
            content_genres = content.get('genres', [])
            matching_genres = []
            
            for genre in content_genres:
                if genre.lower() in time_prefs and time_prefs[genre.lower()] > 0.1:
                    matching_genres.append(genre)
            
            if matching_genres:
                return f"Matches your {time_context} preference for {', '.join(matching_genres[:2])}"
        
        # Fallback to generic message
        time_messages = {
            'morning': "Good for morning viewing",
            'afternoon': "Perfect for afternoon break", 
            'evening': "Great for evening entertainment",
            'night': "Ideal for late-night watching"
        }
        
        return time_messages.get(time_context, "Recommended for you")

    def get_popular_content(self, num_recommendations: int = 10) -> List[Dict]:
        """Fallback: return popular content for new users"""
        try:
            popular_content = list(self.db.get_collection('content').find().sort("rating", -1).limit(num_recommendations))

            recommendations = []
            for content in popular_content:
                recommendations.append({
                    'content_id': str(content['id']),
                    'title': content.get('title', ''),
                    'platform': content.get('platform', ''),
                    'genres': content.get('genres', []),
                    'rating': content.get('rating', 0),
                    'similarity_score': 0.8,
                    'reason': "Popular content"
                })

            return recommendations

        except Exception as e:
            print(f"Error getting popular content: {e}")
            return []

    def analyze_user_temporal_patterns(self, user_id: str) -> Dict:
        """Analyze user's temporal viewing patterns"""
        try:
            # Learn preferences first
            preferences = self.preference_learner.learn_user_temporal_preferences(user_id)
            
            # Get interaction counts by time window
            interactions = list(self.db.get_collection("user_interaction").find({
                "user_id": user_id
            }))
            
            time_window_counts = {'morning': 0, 'afternoon': 0, 'evening': 0, 'night': 0}
            
            for interaction in interactions:
                timestamp = interaction['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                time_window = self.get_time_window(timestamp)
                time_window_counts[time_window] += 1
            
            analysis = {
                'user_id': user_id,
                'temporal_patterns': {},
                'peak_viewing_time': max(time_window_counts.items(), key=lambda x: x[1])[0] if any(time_window_counts.values()) else None,
                'preferred_genres_by_time': preferences,
                'interaction_counts_by_time': time_window_counts
            }
            
            # Analyze patterns for each time window
            for time_window, genre_prefs in preferences.items():
                if genre_prefs:
                    top_genres = sorted(genre_prefs.items(), key=lambda x: x[1], reverse=True)[:3]
                    analysis['temporal_patterns'][time_window] = {
                        'top_genres': [genre for genre, score in top_genres],
                        'genre_scores': dict(top_genres),
                        'diversity_score': len(genre_prefs)
                    }
            
            # Save analysis to file
            json_data = json.dumps(analysis, indent=2)
            with open('temporal_analysis.json', 'w') as f:
                f.write(json_data)
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing temporal patterns: {e}")
            return {}

    def get_user_interaction_summary(self, user_id: str) -> Dict:
        """Get enhanced summary with temporal insights"""
        try:
            interactions = list(self.db.get_collection('user_interaction').find({
                "user_id": user_id
            }))

            summary = {
                'total_interactions': len(interactions),
                'event_type_counts': {},
                'avg_watch_progress': 0.0,
                'preferred_genres': {},
                'preferred_platforms': {},
                'overall_engagement_score': 0.0,
                'temporal_activity': {'morning': 0, 'afternoon': 0, 'evening': 0, 'night': 0}
            }

            total_score = 0.0
            watch_progresses = []

            for interaction in interactions:
                event_type = interaction['event_type']
                summary['event_type_counts'][event_type] = summary['event_type_counts'].get(event_type, 0) + 1

                # Track temporal activity
                timestamp = interaction['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                time_window = self.get_time_window(timestamp)
                summary['temporal_activity'][time_window] += 1

                # Calculate interaction score
                watch_progress = interaction.get('watchProgress', 0.0)
                if event_type == 'watch':
                    watch_progresses.append(watch_progress)

                interaction_score = self.calculate_interaction_score(event_type, watch_progress)
                total_score += interaction_score

                # Analyze content preferences
                content = self.db.get_collection('content').find_one({"id": interaction['content_id']})
                if content and interaction_score > 0:
                    for genre in content.get('genres', []):
                        summary['preferred_genres'][genre] = summary['preferred_genres'].get(genre, 0) + interaction_score

                    platform = content.get('platform', '')
                    if platform:
                        summary['preferred_platforms'][platform] = summary['preferred_platforms'].get(platform, 0) + interaction_score

            if watch_progresses:
                summary['avg_watch_progress'] = np.mean(watch_progresses)

            if len(interactions) > 0:
                summary['overall_engagement_score'] = total_score / len(interactions)

            # Save summary to file
            json_data = json.dumps(summary, indent=2)
            with open('user_summary.json', 'w') as f:
                f.write(json_data)
                
            return summary

        except Exception as e:
            print(f"Error getting user interaction summary: {e}")
            return {}

# if __name__ == "__main__":
#     engine = TemporalRecommendationEngine()
#     engine.initialize()
#     
#     # Test user ID
#     user_id = "01"
#     
#     print("=== Getting Temporal Recommendations ===")
#     recommendations = engine.get_temporal_recommendations(user_id, num_recommendations=10)
#     print(f"Generated {len(recommendations)} recommendations")
#     
#     print("\n=== Analyzing Temporal Patterns ===")
#     analysis = engine.analyze_user_temporal_patterns(user_id)
#     print(f"Peak viewing time: {analysis.get('peak_viewing_time', 'Unknown')}")
#     
#     print("\n=== User Interaction Summary ===")
#     summary = engine.get_user_interaction_summary(user_id)
#     print(f"Total interactions: {summary.get('total_interactions', 0)}")
#     print(f"Temporal activity: {summary.get('temporal_activity', {})}")
#     
#     print("\n=== Testing Different Time Contexts ===")
#     for time_context in ['morning', 'afternoon', 'evening', 'night']:
#         recs = engine.get_temporal_recommendations(user_id, num_recommendations=10, time_context=time_context)
#         print(f"{time_context.capitalize()}: {len(recs)} recommendations")
#         if recs:
#             print(f"  Top recommendation: {recs[0]['title']} (boost: {recs[0].get('temporal_boost', 1.0):.2f})")
