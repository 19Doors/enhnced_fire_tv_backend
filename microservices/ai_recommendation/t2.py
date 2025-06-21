import numpy as np
import asyncio
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pymongo

mongo_url = "mongodb://firetv:password@34.47.135.240:27017"
client = pymongo.MongoClient(mongo_url)

class RecommendationEngine:
    def __init__(self):
        self.db = client.get_database("firetv_content")
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.content_vectors = None
        self.content_ids = []

        self.event_weights = {
            'click': 0.2,
            'like': 0.8,
            'dislike': -0.7,
            'watch': 1.0,
        }

    async def initialize(self):
        try:
            all_content = self.db.get_collection("content").find().to_list(length=None)

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
            print(f"Initialized recommendation engine with {len(self.content_ids)} content items")

        except Exception as e:
            print(f"Error initializing recommendation engine: {e}")
            raise

    def calculate_interaction_score(self, event_type: str, watch_progress: float = 0.0) -> float:
        base_weight = self.event_weights.get(event_type, 0.0)

        if event_type == 'watch':
            return base_weight * watch_progress
        else:
            return base_weight

    async def get_user_profile_vector(self, user_id: str):
        try:
            user_interactions = self.db.get_collection("user_interaction").find({
                "user_id": user_id
            }).sort("timestamp", -1).limit(100).to_list(length=100)

            if not user_interactions:
                return None

            content_scores = {}

            for interaction in user_interactions:
                content_id = interaction['content_id']
                event_type = interaction['event_type']
                watch_progress = interaction.get('watchProgress', 0.0)

                interaction_score = self.calculate_interaction_score(
                    event_type, watch_progress
                )

                time_decay = self._calculate_time_decay(interaction['timestamp'])
                final_score = interaction_score * time_decay

                if content_id in content_scores:
                    content_scores[content_id] += final_score
                else:
                    content_scores[content_id] = final_score

            positive_content = {cid: score for cid, score in content_scores.items() if score > 0}

            if not positive_content:
                return None

            positive_content_ids = list(positive_content.keys())
            print("postive_ids:",positive_content_ids)
            liked_content = self.db.get_collection("content").find({
                "id": {"$in": positive_content_ids}
            }).to_list(length=None)

            weighted_texts = []
            for content in liked_content:
                content_id = str(content['id'])
                content_score = positive_content[content_id]

                text_features = [
                    content.get('title', ''),
                    ' '.join(content.get('genres', [])),
                    ' '.join(content.get('mood_tags', [])),
                    content.get('overview', ''),
                    content.get('platform', '')
                ]
                content_text = ' '.join(filter(None, text_features))

                # Weight the text by user's preference score
                # Repeat text based on score (higher score = more influence)
                repetitions = max(1, int(content_score * 3))  # Scale factor
                weighted_texts.extend([content_text] * repetitions)

            # Combine all weighted preferences into single text
            combined_user_text = ' '.join(weighted_texts)

            # Transform to vector using same vectorizer
            user_vector = self.vectorizer.transform([combined_user_text])

            return user_vector

        except Exception as e:
            print(f"Error creating user profile vector: {e}")
            return None

    def _calculate_time_decay(self, timestamp) -> float:
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

        days_ago = (datetime.utcnow() - timestamp.replace(tzinfo=None)).days

        # Exponential decay: recent = 1.0, 30 days ago = ~0.5, 90 days ago = ~0.1
        decay_factor = np.exp(-days_ago / 30.0)
        return max(0.1, decay_factor)  # Minimum weight of 0.1

    async def get_recommendations(self, user_id: str, num_recommendations: int = 10) -> List[Dict]:
        """Get recommendations using weighted cosine similarity"""
        try:
            # Get user profile vector
            user_vector = await self.get_user_profile_vector(user_id)

            if user_vector is None:
                # Return popular content for new users
                return await self.get_popular_content(num_recommendations)

            # Calculate cosine similarity between user and all content
            similarities = cosine_similarity(user_vector, self.content_vectors).flatten()

            # Get user's interaction history to exclude and apply negative filtering
            user_interactions = self.db.get_collection("user_interaction").find({
                "user_id": user_id
            }).to_list(length=None)

            # Build exclusion and negative preference sets
            interacted_content = set()
            disliked_content = set()

            for interaction in user_interactions:
                content_id = interaction['content_id']
                interacted_content.add(content_id)

                if interaction['event_type'] in ['dislike', 'skip']:
                    disliked_content.add(content_id)
                elif interaction['event_type'] == 'watch' and interaction.get('watchProgress', 0) < 0.1:
                    # Very low watch progress indicates disinterest
                    disliked_content.add(content_id)

            # Get top similar content (excluding interacted and disliked)
            recommendations = []

            # Sort by similarity score
            similar_indices = np.argsort(similarities)[::-1]

            for idx in similar_indices:
                content_id = self.content_ids[idx]

                # Skip if already interacted with or disliked
                if content_id in interacted_content or content_id in disliked_content:
                    continue

                # Get content details
                content = self.db.get_collection('content').find_one({"id": content_id})
                print(content)
                if content:
                    similarity_score = float(similarities[idx])

                    # Apply additional filtering based on user's negative preferences
                    # if await self._should_filter_content(user_id, content):
                    #     continue

                    recommendations.append({
                        'content_id': content_id,
                        'title': content.get('title', ''),
                        'platform': content.get('platform', ''),
                        'genres': content.get('genres', []),
                        'rating': content.get('rating', 0),
                        'similarity_score': similarity_score,
                        'reason': self._generate_recommendation_reason(user_id, content, similarity_score)
                    })

                if len(recommendations) >= num_recommendations:
                    break

            json_de = json.dumps(recommendations, indent=2);
            with open('a.json', 'w') as f:
                f.write(json_de)
            return recommendations

        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []

    # async def _should_filter_content(self, user_id: str, content: Dict) -> bool:
    #     """Check if content should be filtered based on user's negative preferences"""
    #     try:
    #         # Get user's disliked genres/platforms
    #         negative_interactions = await self.db.user_interactions.find({
    #             "user_id": user_id,
    #             "event_type": {"$in": ["dislike", "skip"]}
    #         }).to_list(length=50)
    #
    #         # Count negative interactions by genre and platform
    #         disliked_genres = {}
    #         disliked_platforms = {}
    #
    #         for interaction in negative_interactions:
    #             interacted_content = await self.db.content.find_one({"_id": interaction['content_id']})
    #             if interacted_content:
    #                 # Count genre dislikes
    #                 for genre in interacted_content.get('genres', []):
    #                     disliked_genres[genre] = disliked_genres.get(genre, 0) + 1
    #
    #                 # Count platform dislikes
    #                 platform = interacted_content.get('platform', '')
    #                 if platform:
    #                     disliked_platforms[platform] = disliked_platforms.get(platform, 0) + 1
    #
    #         # Filter if content has heavily disliked genres/platforms
    #         content_genres = content.get('genres', [])
    #         content_platform = content.get('platform', '')
    #
    #         # If user has disliked this genre/platform more than 3 times, filter it
    #         for genre in content_genres:
    #             if disliked_genres.get(genre, 0) >= 3:
    #                 return True
    #
    #         if disliked_platforms.get(content_platform, 0) >= 3:
    #             return True
    #
    #         return False
    #
    #     except Exception as e:
    #         logger.error(f"Error filtering content: {e}")
    #         return False

    def _generate_recommendation_reason(self, user_id: str, content: Dict, similarity_score: float) -> str:
        if similarity_score > 0.8:
            return f"Highly matches your preferences (similarity: {similarity_score:.2f})"
        elif similarity_score > 0.6:
            return f"Good match for your taste (similarity: {similarity_score:.2f})"
        elif similarity_score > 0.4:
            return f"You might enjoy this (similarity: {similarity_score:.2f})"
        else:
            return f"Something new to explore (similarity: {similarity_score:.2f})"

    async def get_popular_content(self, num_recommendations: int = 10) -> List[Dict]:
        """Fallback: return popular content for new users"""
        try:
            popular_content = self.db.get_collection('content').find().sort("rating", -1).limit(num_recommendations).to_list(length=num_recommendations)

            recommendations = []
            for content in popular_content:
                recommendations.append({
                    'content_id': str(content['_id']),
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

    async def get_user_interaction_summary(self, user_id: str) -> Dict:
        """Get summary of user's interaction patterns"""
        try:
            interactions = self.db.get_collection('user_interaction').find({
                "user_id": user_id
            }).to_list(length=None)

            summary = {
                'total_interactions': len(interactions),
                'event_type_counts': {},
                'avg_watch_progress': 0.0,
                'preferred_genres': {},
                'preferred_platforms': {},
                'overall_engagement_score': 0.0
            }

            total_score = 0.0
            watch_progresses = []

            for interaction in interactions:
                event_type = interaction['event_type']
                summary['event_type_counts'][event_type] = summary['event_type_counts'].get(event_type, 0) + 1

                # Calculate interaction score
                watch_progress = interaction.get('watchProgress', 0.0)
                if event_type == 'watch':
                    watch_progresses.append(watch_progress)

                interaction_score = self.calculate_interaction_score(event_type, watch_progress)
                total_score += interaction_score

                # Analyze content preferences
                content = self.db.get_collection('content').find_one({"id": interaction['content_id']})
                if content and interaction_score > 0:  # Only count positive interactions
                    for genre in content.get('genres', []):
                        summary['preferred_genres'][genre] = summary['preferred_genres'].get(genre, 0) + interaction_score

                    platform = content.get('platform', '')
                    if platform:
                        summary['preferred_platforms'][platform] = summary['preferred_platforms'].get(platform, 0) + interaction_score

            if watch_progresses:
                summary['avg_watch_progress'] = np.mean(watch_progresses)

            if len(interactions) > 0:
                summary['overall_engagement_score'] = total_score / len(interactions)

            soo = json.dumps(summary, indent=2);
            with open('b.json', 'w') as f:
                f.write(soo)
            return summary

        except Exception as e:
            print(f"Error getting user interaction summary: {e}")
            return {}

t = RecommendationEngine()
asyncio.run(t.initialize())
asyncio.run(t.get_user_profile_vector(user_id="01"))
asyncio.run(t.get_recommendations(user_id='01'))
asyncio.run(t.get_user_interaction_summary(user_id='01'))
