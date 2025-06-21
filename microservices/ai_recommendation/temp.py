# microservices/ai-recommendation-service/services/temporal_intelligence_engine.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Embedding, Dropout, Input, Concatenate
from keras.optimizers import Adam
import joblib
import redis
import asyncio

logger = logging.getLogger(__name__)

class TemporalIntelligenceEngine:
    def __init__(self):
        self.time_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.preference_predictor = None
        self.sequence_model = None
        self.attention_model = None
        self.is_initialized = False
        
        # Time window mappings
        self.time_windows = {
            'early_morning': (5, 8),    # 5-8 AM
            'morning': (8, 12),         # 8-12 PM  
            'afternoon': (12, 17),      # 12-5 PM
            'evening': (17, 21),        # 5-9 PM
            'night': (21, 24),          # 9 PM-12 AM
            'late_night': (0, 5)        # 12-5 AM
        }
        
    async def initialize_models(self):
        """Initialize all ML models for temporal intelligence"""
        try:
            logger.info("Initializing Temporal Intelligence ML Models...")
            
            # Initialize sequence-based preference model (LSTM)
            await self._initialize_sequence_model()
            
            # Initialize attention-based temporal model
            await self._initialize_attention_model()
            
            # Initialize preference prediction model
            await self._initialize_preference_predictor()
            
            self.is_initialized = True
            logger.info("Temporal Intelligence Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing temporal models: {e}")
            raise
    
    async def _initialize_sequence_model(self):
        """Initialize LSTM-based sequence model for temporal patterns"""
        # LSTM model for learning temporal viewing sequences
        self.sequence_model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(10, 50)),  # 10 time steps, 50 features
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # Preference score
        ])
        
        self.sequence_model.compile(
            optimizer=Adam(learning_rate=0.01),
            loss='mse',
            metrics=['mae']
        )

    async def _initialize_attention_model(self):
        """Initialize attention-based temporal model"""
        # Input layers
        sequence_input = Input(shape=(10, 30), name='sequence_input')
        time_input = Input(shape=(6,), name='time_input')
        
        # LSTM with attention mechanism
        lstm_out = LSTM(64, return_sequences=True)(sequence_input)
        
        # Simple attention mechanism
        attention_weights = Dense(64, activation='tanh')(lstm_out)
        attention_weights = Dense(1, activation='softmax')(attention_weights)
        
        # FIX: Use Keras operations instead of TensorFlow functions
        from keras import ops
        attended_output = ops.sum(lstm_out * attention_weights, axis=1)
        
        # Alternative fix using Lambda layer:
        # attended_output = Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1))([lstm_out, attention_weights])
        
        # Combine with time features
        time_dense = Dense(32, activation='relu')(time_input)
        combined = Concatenate()([attended_output, time_dense])
        
        # Final prediction layers
        dense1 = Dense(64, activation='relu')(combined)
        dense2 = Dense(32, activation='relu')(dense1)
        output = Dense(10, activation='softmax', name='preference_output')(dense2)
        
        self.attention_model = Model(inputs=[sequence_input, time_input], outputs=output)
        self.attention_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    async def _initialize_preference_predictor(self):
        """Initialize Random Forest for preference prediction"""
        self.preference_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    
    def extract_temporal_features(self, user_events: List[Dict]) -> np.ndarray:
        """Extract comprehensive temporal features using ML techniques"""
        if not user_events:
            return np.array([])
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(user_events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Time window encoding
        df['time_window'] = df['hour'].apply(self._get_time_window_numeric)
        
        # Sequential features
        df = df.sort_values('timestamp')
        df['time_since_last'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
        df['session_position'] = df.groupby(df['timestamp'].dt.date).cumcount()
        
        # Engagement features
        df['watch_progress'] = df['watchProgress'].fillna(0)
        df['engagement_score'] = self._calculate_engagement_score(df)
        
        # Content temporal features
        content_features = self._extract_content_temporal_features(df)
        
        # Combine all features
        feature_columns = [
            'hour', 'day_of_week', 'month', 'is_weekend', 'time_window',
            'time_since_last', 'session_position', 'engagement_score'
        ]
        
        features = df[feature_columns].values
        
        # Add content features
        if len(content_features) > 0:
            features = np.hstack([features, content_features])
        
        return features
    
    def _get_time_window_numeric(self, hour: int) -> int:
        """Convert hour to numeric time window"""
        for i, (window, (start, end)) in enumerate(self.time_windows.items()):
            if window == 'late_night':
                if hour >= start or hour < end:
                    return i
            else:
                if start <= hour < end:
                    return i
        return 0  # Default to early morning
    
    def _calculate_engagement_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate engagement score using ML techniques"""
        engagement_features = []
        
        for _, row in df.iterrows():
            event_type = row.get('event_type', 'watch')
            watch_progress = row.get('watch_progress', 0)
            
            # Base engagement scores
            base_scores = {
                'click': 0.2,
                'like': 0.8,
                'dislike': -0.3,
                'watch': watch_progress
            }
            
            base_score = base_scores.get(event_type, 0)
            
            # Time-based engagement adjustment
            hour = row['hour']
            if 8 <= hour <= 10 or 19 <= hour <= 22:  # Peak engagement hours
                base_score *= 1.2
            elif 0 <= hour <= 6:  # Low engagement hours
                base_score *= 0.8
            
            engagement_features.append(base_score)
        
        return pd.Series(engagement_features)
    
    def _extract_content_temporal_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract content-specific temporal features"""
        content_features = []
        
        for _, row in df.iterrows():
            context_data = row.get('context_data', {})
            
            # Genre temporal features
            genres = context_data.get('genres', [])
            genre_features = self._encode_genres_temporally(genres, row['hour'])
            
            # Mood temporal features  
            mood_tags = context_data.get('mood_tags', [])
            mood_features = self._encode_moods_temporally(mood_tags, row['hour'])
            
            # Platform temporal features
            platform = row.get('content_platform', '')
            platform_features = self._encode_platform_temporally(platform, row['hour'])
            
            # Combine content features
            combined_features = np.concatenate([
                genre_features, mood_features, platform_features
            ])
            content_features.append(combined_features)
        
        return np.array(content_features) if content_features else np.array([])
    
    def _encode_genres_temporally(self, genres: List[str], hour: int) -> np.ndarray:
        """Encode genres with temporal context"""
        # Define temporal-genre associations
        temporal_genre_weights = {
            'morning': {'comedy': 1.2, 'family': 1.1, 'documentary': 1.0},
            'afternoon': {'drama': 1.1, 'romance': 1.0, 'comedy': 0.9},
            'evening': {'action': 1.2, 'thriller': 1.1, 'drama': 1.0},
            'night': {'horror': 1.2, 'thriller': 1.1, 'mystery': 1.0}
        }
        
        time_window = self._get_time_window_name(hour)
        weights = temporal_genre_weights.get(time_window, {})
        
        # Create genre feature vector
        genre_vector = np.zeros(10)  # Support top 10 genres
        genre_mapping = {
            'comedy': 0, 'drama': 1, 'action': 2, 'thriller': 3, 'romance': 4,
            'horror': 5, 'family': 6, 'documentary': 7, 'mystery': 8, 'sci-fi': 9
        }
        
        for genre in genres:
            if genre.lower() in genre_mapping:
                idx = genre_mapping[genre.lower()]
                weight = weights.get(genre.lower(), 1.0)
                genre_vector[idx] = weight
        
        return genre_vector
    
    def _encode_moods_temporally(self, mood_tags: List[str], hour: int) -> np.ndarray:
        """Encode mood tags with temporal context"""
        # Temporal-mood associations
        temporal_mood_weights = {
            'morning': {'uplifting': 1.3, 'energetic': 1.2, 'positive': 1.1},
            'afternoon': {'relaxing': 1.1, 'moderate': 1.0, 'engaging': 0.9},
            'evening': {'intense': 1.2, 'dramatic': 1.1, 'engaging': 1.2},
            'night': {'calm': 1.1, 'mysterious': 1.2, 'atmospheric': 1.1}
        }
        
        time_window = self._get_time_window_name(hour)
        weights = temporal_mood_weights.get(time_window, {})
        
        # Create mood feature vector
        mood_vector = np.zeros(8)  # Support 8 mood categories
        mood_mapping = {
            'uplifting': 0, 'intense': 1, 'relaxing': 2, 'dramatic': 3,
            'energetic': 4, 'calm': 5, 'mysterious': 6, 'engaging': 7
        }
        
        for mood in mood_tags:
            if mood.lower() in mood_mapping:
                idx = mood_mapping[mood.lower()]
                weight = weights.get(mood.lower(), 1.0)
                mood_vector[idx] = weight
        
        return mood_vector
    
    def _encode_platform_temporally(self, platform: str, hour: int) -> np.ndarray:
        """Encode platform with temporal context"""
        # Platform temporal usage patterns
        platform_temporal_weights = {
            'morning': {'netflix': 1.1, 'hotstar': 1.2, 'primevideo': 0.9},
            'afternoon': {'hotstar': 1.2, 'netflix': 1.0, 'primevideo': 1.1},
            'evening': {'netflix': 1.2, 'primevideo': 1.1, 'hotstar': 1.0},
            'night': {'netflix': 1.3, 'primevideo': 1.1, 'hotstar': 0.9}
        }
        
        time_window = self._get_time_window_name(hour)
        weights = platform_temporal_weights.get(time_window, {})
        
        # Create platform feature vector
        platform_vector = np.zeros(3)  # 3 platforms
        platform_mapping = {'netflix': 0, 'hotstar': 1, 'primevideo': 2}
        
        if platform.lower() in platform_mapping:
            idx = platform_mapping[platform.lower()]
            weight = weights.get(platform.lower(), 1.0)
            platform_vector[idx] = weight
        
        return platform_vector
    
    def _get_time_window_name(self, hour: int) -> str:
        """Get time window name from hour"""
        for window, (start, end) in self.time_windows.items():
            if window == 'late_night':
                if hour >= start or hour < end:
                    return window
            else:
                if start <= hour < end:
                    return window
        return 'morning'
    
    async def predict_temporal_preferences(self, user_events: List[Dict], 
                                         target_time_window: str) -> Dict[str, float]:
        """Predict user preferences for specific time window using ML models"""
        if not self.is_initialized:
            await self.initialize_models()
        
        # Extract features
        features = self.extract_temporal_features(user_events)
        if len(features) == 0:
            return {}
        
        # Use sequence model for temporal prediction
        if len(features) >= 10:
            # Prepare sequence data
            sequence_data = self._prepare_sequence_data(features)
            
            # Predict using LSTM
            lstm_predictions = self.sequence_model.predict(sequence_data)
            
            # Use attention model for refined predictions
            time_features = self._extract_time_features(target_time_window)
            attention_predictions = self.attention_model.predict([
                sequence_data, 
                np.array([time_features])
            ])
            
            # Combine predictions
            combined_predictions = self._combine_predictions(
                lstm_predictions, attention_predictions
            )
            
            return self._convert_predictions_to_preferences(combined_predictions)
        
        else:
            # Use simpler model for limited data
            return self._fallback_prediction(features, target_time_window)
    
    def _prepare_sequence_data(self, features: np.ndarray) -> np.ndarray:
        """Prepare sequence data for LSTM model"""
        # Take last 10 interactions
        if len(features) < 10:
            # Pad with zeros if insufficient data
            padded = np.zeros((10, features.shape[1]))
            padded[-len(features):] = features
            return padded.reshape(1, 10, features.shape[1])
        else:
            return features[-10:].reshape(1, 10, features.shape[1])
    
    def _extract_time_features(self, time_window: str) -> np.ndarray:
        """Extract time-specific features"""
        current_time = datetime.now()
        
        time_features = np.array([
            current_time.hour / 24.0,  # Normalized hour
            current_time.weekday() / 7.0,  # Normalized day of week
            current_time.month / 12.0,  # Normalized month
            1.0 if current_time.weekday() >= 5 else 0.0,  # Is weekend
            list(self.time_windows.keys()).index(time_window) / len(self.time_windows),  # Time window
            np.sin(2 * np.pi * current_time.hour / 24)  # Cyclical hour
        ])
        
        return time_features
    
    def _combine_predictions(self, lstm_pred: np.ndarray, 
                           attention_pred: np.ndarray) -> np.ndarray:
        """Combine LSTM and attention model predictions"""
        # Weighted combination
        lstm_weight = 0.6
        attention_weight = 0.4
        
        # Normalize attention predictions to match LSTM output shape
        attention_normalized = np.mean(attention_pred, axis=1, keepdims=True)
        
        combined = lstm_weight * lstm_pred + attention_weight * attention_normalized
        return combined
    
    def _convert_predictions_to_preferences(self, predictions: np.ndarray) -> Dict[str, float]:
        """Convert model predictions to preference dictionary"""
        # Map predictions to preference categories
        preference_categories = [
            'comedy_preference', 'drama_preference', 'action_preference',
            'romance_preference', 'thriller_preference', 'documentary_preference',
            'family_preference', 'horror_preference', 'sci_fi_preference',
            'mystery_preference'
        ]
        
        # Ensure we have enough predictions
        if len(predictions.flatten()) < len(preference_categories):
            predictions = np.tile(predictions, (len(preference_categories), 1))[:len(preference_categories)]
        
        preferences = {}
        for i, category in enumerate(preference_categories):
            if i < len(predictions.flatten()):
                preferences[category] = float(predictions.flatten()[i])
            else:
                preferences[category] = 0.5  # Default neutral preference
        
        return preferences
    
    def _fallback_prediction(self, features: np.ndarray, 
                           time_window: str) -> Dict[str, float]:
        """Fallback prediction for limited data"""
        # Use simple heuristics based on time window
        time_based_preferences = {
            'morning': {
                'comedy_preference': 0.8, 'family_preference': 0.7,
                'documentary_preference': 0.6, 'drama_preference': 0.4
            },
            'afternoon': {
                'drama_preference': 0.7, 'romance_preference': 0.6,
                'documentary_preference': 0.5, 'comedy_preference': 0.5
            },
            'evening': {
                'action_preference': 0.8, 'thriller_preference': 0.7,
                'drama_preference': 0.6, 'mystery_preference': 0.5
            },
            'night': {
                'thriller_preference': 0.7, 'horror_preference': 0.6,
                'mystery_preference': 0.6, 'drama_preference': 0.5
            }
        }
        
        return time_based_preferences.get(time_window, {
            'comedy_preference': 0.5, 'drama_preference': 0.5,
            'action_preference': 0.5, 'romance_preference': 0.5
        })
    
    async def analyze_user_temporal_patterns(self, user_events: List[Dict]) -> Dict[str, Any]:
        """Analyze individual user's temporal viewing patterns"""
        if not user_events:
            return {}
        
        # Extract features
        features = self.extract_temporal_features(user_events)
        if len(features) == 0:
            return {}
        
        # Create DataFrame for analysis
        df = pd.DataFrame(user_events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Analyze patterns by time window
        time_window_analysis = {}
        
        for window_name, (start, end) in self.time_windows.items():
            if window_name == 'late_night':
                window_mask = (df['hour'] >= start) | (df['hour'] < end)
            else:
                window_mask = (df['hour'] >= start) & (df['hour'] < end)
            
            window_data = df[window_mask]
            
            if len(window_data) > 0:
                # Calculate engagement metrics
                avg_engagement = window_data['watchProgress'].mean()
                
                # Extract most common genres and moods
                all_genres = []
                all_moods = []
                
                for _, row in window_data.iterrows():
                    context_data = row.get('context_data', {})
                    all_genres.extend(context_data.get('genres', []))
                    all_moods.extend(context_data.get('mood_tags', []))
                
                # Count frequencies
                genre_counts = pd.Series(all_genres).value_counts().head(3).to_dict()
                mood_counts = pd.Series(all_moods).value_counts().head(3).to_dict()
                
                time_window_analysis[window_name] = {
                    'session_count': len(window_data),
                    'avg_engagement': float(avg_engagement),
                    'top_genres': genre_counts,
                    'top_moods': mood_counts,
                    'avg_watch_progress': float(window_data['watchProgress'].mean())
                }
        
        return {
            'user_temporal_patterns': time_window_analysis,
            'overall_engagement': float(df['watchProgress'].mean()),
            'most_active_time_window': max(time_window_analysis.items(), 
                                         key=lambda x: x[1]['session_count'])[0] if time_window_analysis else None,
            'peak_engagement_window': max(time_window_analysis.items(), 
                                        key=lambda x: x[1]['avg_engagement'])[0] if time_window_analysis else None
        }
    
    async def get_temporal_recommendation_weights(self, user_events: List[Dict], 
                                                current_time_window: str) -> Dict[str, float]:
        """Get recommendation weights based on temporal patterns"""
        if not user_events:
            return self._get_default_weights(current_time_window)
        
        # Analyze user patterns
        patterns = await self.analyze_user_temporal_patterns(user_events)
        
        if current_time_window not in patterns.get('user_temporal_patterns', {}):
            return self._get_default_weights(current_time_window)
        
        window_data = patterns['user_temporal_patterns'][current_time_window]
        
        # Calculate weights based on user's historical preferences
        weights = {}
        
        # Genre weights
        top_genres = window_data.get('top_genres', {})
        total_genre_count = sum(top_genres.values()) if top_genres else 1
        
        for genre, count in top_genres.items():
            weights[f"{genre.lower()}_preference"] = count / total_genre_count
        
        # Mood weights
        top_moods = window_data.get('top_moods', {})
        total_mood_count = sum(top_moods.values()) if top_moods else 1
        
        for mood, count in top_moods.items():
            weights[f"{mood.lower()}_mood_weight"] = count / total_mood_count
        
        # Engagement weight
        weights['engagement_factor'] = min(window_data.get('avg_engagement', 0.5), 1.0)
        
        return weights
    
    def _get_default_weights(self, time_window: str) -> Dict[str, float]:
        """Get default weights for time window"""
        default_weights = {
            'morning': {
                'comedy_preference': 0.8, 'family_preference': 0.6,
                'uplifting_mood_weight': 0.7, 'engagement_factor': 0.7
            },
            'afternoon': {
                'drama_preference': 0.6, 'documentary_preference': 0.5,
                'relaxing_mood_weight': 0.6, 'engagement_factor': 0.6
            },
            'evening': {
                'action_preference': 0.7, 'thriller_preference': 0.6,
                'intense_mood_weight': 0.7, 'engagement_factor': 0.8
            },
            'night': {
                'thriller_preference': 0.6, 'mystery_preference': 0.5,
                'atmospheric_mood_weight': 0.6, 'engagement_factor': 0.7
            }
        }
        
        return default_weights.get(time_window, {
            'comedy_preference': 0.5, 'engagement_factor': 0.6
        })
    
    async def save_models(self, model_path: str):
        """Save trained models"""
        try:
            # Save sklearn models
            joblib.dump(self.scaler, f"{model_path}/scaler.pkl")
            if self.preference_predictor:
                joblib.dump(self.preference_predictor, f"{model_path}/preference_predictor.pkl")
            
            # Save TensorFlow models
            if self.sequence_model:
                self.sequence_model.save(f"{model_path}/sequence_model.h5")
            if self.attention_model:
                self.attention_model.save(f"{model_path}/attention_model.h5")
            
            logger.info("Temporal intelligence models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
    
    async def load_models(self, model_path: str):
        """Load pre-trained models"""
        try:
            # Load sklearn models
            self.scaler = joblib.load(f"{model_path}/scaler.pkl")
            self.preference_predictor = joblib.load(f"{model_path}/preference_predictor.pkl")
            
            # Load TensorFlow models
            self.sequence_model = tf.keras.models.load_model(f"{model_path}/sequence_model.h5")
            self.attention_model = tf.keras.models.load_model(f"{model_path}/attention_model.h5")
            
            self.is_initialized = True
            logger.info("Temporal intelligence models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

test = TemporalIntelligenceEngine()
asyncio.run(test.initialize_models())
