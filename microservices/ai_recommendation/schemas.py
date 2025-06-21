from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

class EventType(str, Enum):
    CLICK = "click"
    LIKE = "like"
    DISLIKE = "dislike"
    WATCH = "watch"

class ContentType(str, Enum):
    MOVIE = "movie"
    TV_SHOW = "tv_show"

class Platform(str, Enum):
    NETFLIX = "netflix"
    HOTSTAR = "hotstar"
    PRIMEVIDEO = "primevideo"

class TimeWindow(str, Enum):
    MORNING = "morning"      # 6-12
    AFTERNOON = "afternoon"  # 12-17
    EVENING = "evening"      # 17-22
    NIGHT = "night"         # 22-6

class UserEvent(BaseModel):
    user_id: str
    event_type: EventType
    content_type: ContentType
    content_id: str
    content_platform: Platform
    context_data: Dict[str, Any]
    timestamp: datetime
    watch_progress: Optional[float] = 0.0

class ContentDNA(BaseModel):
    content_id: str
    title: str
    genres: List[str]
    mood_tags: List[str]
    rating: float
    platform: Platform
    content_type: ContentType
    release_date: Optional[str] = None
    overview: Optional[str] = None
    emotional_tone: Optional[str] = None

class TemporalPreference(BaseModel):
    time_window: TimeWindow
    preferred_moods: List[str]
    preferred_genres: List[str]
    avg_watch_progress: float
    content_count: int
    last_updated: datetime

class UserProfile(BaseModel):
    user_id: str
    temporal_preferences: Dict[TimeWindow, TemporalPreference]
    overall_preferences: Dict[str, float]
    platform_affinity: Dict[Platform, float]
    created_at: datetime
    updated_at: datetime

class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: int = 10
    time_context: Optional[TimeWindow] = None
    platform_filter: Optional[List[Platform]] = None

class Recommendation(BaseModel):
    content_id: str
    title: str
    platform: Platform
    similarity_score: float
    confidence: float
    reason: str
    mood_match: float
    temporal_relevance: float
    genres: List[str]
    mood_tags: List[str]

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Recommendation]
    time_context: TimeWindow
    generation_strategy: str
    total_count: int
    generated_at: datetime
