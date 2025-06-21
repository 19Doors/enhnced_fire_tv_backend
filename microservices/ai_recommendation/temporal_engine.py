# microservices/ai-recommendation-service/temporal_engine.py
import numpy as np
from datetime import datetime, time
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class TemporalIntelligenceEngine:
    def __init__(self):
        self.time_windows = {
            "morning": (6, 12),
            "afternoon": (12, 17),
            "evening": (17, 22),
            "night": (22, 6)
        }
        
    def get_time_window(self, timestamp: datetime) -> str:
        """Determine time window from timestamp"""
        hour = timestamp.hour
        
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    def extract_temporal_features(self, user_event: Dict[str, Any]) -> Dict:
        """Extract temporal features from user event"""
        timestamp = user_event.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        time_window = self.get_time_window(timestamp)
        
        # Extract mood and genre signals from context_data
        context_data = user_event.get('context_data', {})
        mood_tags = context_data.get('mood_tags', [])
        genres = context_data.get('genres', [])
        
        # Calculate engagement quality
        engagement_quality = self._calculate_engagement_quality(
            user_event.get('event_type'), 
            user_event.get('watchProgress', 0.0)
        )
        
        return {
            'time_window': time_window,
            'mood_tags': mood_tags,
            'genres': genres,
            'engagement_quality': engagement_quality,
            'platform': user_event.get('content_platform'),
            'content_type': user_event.get('content_type')
        }
    
    def _calculate_engagement_quality(self, event_type: str, watch_progress: float) -> float:
        """Calculate engagement quality score"""
        base_scores = {
            'click': 0.2,
            'like': 0.8,
            'dislike': -0.5,
            'watch': watch_progress or 0.0
        }
        
        return base_scores.get(event_type, 0.0)
    
    def create_temporal_preference(self, time_window: str, temporal_features: Dict) -> Dict:
        """Create new temporal preference structure"""
        return {
            'time_window': time_window,
            'preferred_moods': temporal_features['mood_tags'],
            'preferred_genres': temporal_features['genres'],
            'avg_watch_progress': temporal_features['engagement_quality'],
            'content_count': 1,
            'last_updated': datetime.utcnow()
        }
    
    def update_temporal_preferences(self, 
                                  current_prefs: Optional[Dict],
                                  temporal_features: Dict) -> Dict:
        """Update temporal preferences with new data"""
        
        if current_prefs is None:
            # Initialize new temporal preference
            return self.create_temporal_preference(
                temporal_features['time_window'], 
                temporal_features
            )
        
        # Update existing preferences with weighted average
        alpha = 0.3  # Learning rate
        
        # Update mood preferences
        updated_moods = self._update_preference_list(
            current_prefs.get('preferred_moods', []),
            temporal_features['mood_tags'],
            alpha
        )
        
        # Update genre preferences
        updated_genres = self._update_preference_list(
            current_prefs.get('preferred_genres', []),
            temporal_features['genres'],
            alpha
        )
        
        # Update average watch progress
        current_avg = current_prefs.get('avg_watch_progress', 0.0)
        new_avg_progress = (
            (1 - alpha) * current_avg + 
            alpha * temporal_features['engagement_quality']
        )
        
        return {
            'time_window': temporal_features['time_window'],
            'preferred_moods': updated_moods,
            'preferred_genres': updated_genres,
            'avg_watch_progress': new_avg_progress,
            'content_count': current_prefs.get('content_count', 0) + 1,
            'last_updated': datetime.utcnow()
        }
    
    def _update_preference_list(self, current_list: List[str], 
                               new_items: List[str], alpha: float) -> List[str]:
        """Update preference list with new items"""
        # Simple approach: add new items if they're not already present
        # In production, you'd use more sophisticated weighting
        updated_list = list(set(current_list + new_items))
        return updated_list[:10]  # Keep top 10 preferences
    
    def get_temporal_relevance_score(self, 
                                   content_mood_tags: List[str],
                                   content_genres: List[str],
                                   user_temporal_prefs: Dict) -> float:
        """Calculate how relevant content is for current time context"""
        
        user_moods = user_temporal_prefs.get('preferred_moods', [])
        user_genres = user_temporal_prefs.get('preferred_genres', [])
        
        mood_overlap = len(set(content_mood_tags) & set(user_moods))
        genre_overlap = len(set(content_genres) & set(user_genres))
        
        mood_score = mood_overlap / max(len(user_moods), 1)
        genre_score = genre_overlap / max(len(user_genres), 1)
        
        # Weight mood more heavily for temporal relevance
        temporal_score = 0.6 * mood_score + 0.4 * genre_score
        
        return min(temporal_score, 1.0)
    
    def analyze_viewing_pattern(self, user_events: List[Dict]) -> Dict[str, Any]:
        """Analyze user's viewing patterns across different time windows"""
        time_window_stats = {
            "morning": {"count": 0, "avg_engagement": 0.0, "top_genres": []},
            "afternoon": {"count": 0, "avg_engagement": 0.0, "top_genres": []},
            "evening": {"count": 0, "avg_engagement": 0.0, "top_genres": []},
            "night": {"count": 0, "avg_engagement": 0.0, "top_genres": []}
        }
        
        for event in user_events:
            temporal_features = self.extract_temporal_features(event)
            time_window = temporal_features['time_window']
            
            if time_window in time_window_stats:
                stats = time_window_stats[time_window]
                stats["count"] += 1
                stats["avg_engagement"] += temporal_features['engagement_quality']
                
                # Track genres
                for genre in temporal_features['genres']:
                    if genre not in stats["top_genres"]:
                        stats["top_genres"].append(genre)
        
        # Calculate averages
        for window_stats in time_window_stats.values():
            if window_stats["count"] > 0:
                window_stats["avg_engagement"] /= window_stats["count"]
            window_stats["top_genres"] = window_stats["top_genres"][:5]  # Top 5 genres
        
        return time_window_stats
    
    def predict_optimal_recommendation_time(self, user_temporal_prefs: Dict[str, Dict]) -> str:
        """Predict the best time window for recommendations based on user patterns"""
        best_window = "evening"  # Default
        best_score = 0.0
        
        for time_window, prefs in user_temporal_prefs.items():
            # Score based on engagement and activity
            engagement_score = prefs.get('avg_watch_progress', 0.0)
            activity_score = min(prefs.get('content_count', 0) / 10.0, 1.0)  # Normalize
            
            combined_score = 0.7 * engagement_score + 0.3 * activity_score
            
            if combined_score > best_score:
                best_score = combined_score
                best_window = time_window
        
        return best_window
    
    def get_mood_transition_patterns(self, user_events: List[Dict]) -> Dict[str, List[str]]:
        """Analyze how user moods transition throughout the day"""
        mood_transitions = {}
        
        # Sort events by timestamp
        sorted_events = sorted(user_events, key=lambda x: x.get('timestamp', ''))
        
        for i in range(len(sorted_events) - 1):
            current_features = self.extract_temporal_features(sorted_events[i])
            next_features = self.extract_temporal_features(sorted_events[i + 1])
            
            current_window = current_features['time_window']
            next_window = next_features['time_window']
            
            if current_window != next_window:
                if current_window not in mood_transitions:
                    mood_transitions[current_window] = []
                
                # Track mood tags that transition
                for mood in current_features['mood_tags']:
                    if mood not in mood_transitions[current_window]:
                        mood_transitions[current_window].append(mood)
        
        return mood_transitions
