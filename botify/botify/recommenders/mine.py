import random
from typing import Dict, List
from .recommender import Recommender

class Mine(Recommender):
    def __init__(self, tracks_redis, artists_redis, catalog, fallback, 
                 memory_size=50, skip_threshold=0.3):
        self.tracks_redis = tracks_redis
        self.artists_redis = artists_redis
        self.catalog = catalog
        self.fallback = fallback
        self.memory_size = memory_size
        self.skip_threshold = skip_threshold
        self.user_history: Dict[int, List[int]] = {}
        
    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        track_data = self.tracks_redis.get(prev_track)
        if not track_data:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
            
        track = self.catalog.from_bytes(track_data)
        
        history = self.user_history.setdefault(user, [])
        history.append(prev_track)
        if len(history) > self.memory_size:
            history.pop(0)
            
        if prev_track_time < self.skip_threshold:
            return self._recommend_diverse(track.artist, history)
        else:
            return self._recommend_similar(track.artist, prev_track, history)
    
    def _recommend_similar(self, artist: int, prev_track: int, history: List[int]) -> int:
        artist_data = self.artists_redis.get(artist)
        if not artist_data:
            return self.fallback.recommend_next(None, None, 0.0)
            
        tracks = self.catalog.from_bytes(artist_data)
        candidates = [t for t in tracks if t != prev_track and t not in history]
        
        return random.choice(candidates) if candidates else self.fallback.recommend_next(None, None, 0.0)
    
    def _recommend_diverse(self, prev_artist: int, history: List[int]) -> int:
        all_artists = list(self.artists_redis.keys())
        random.shuffle(all_artists)
        
        for artist in all_artists:
            if artist == prev_artist:
                continue
                
            artist_data = self.artists_redis.get(artist)
            if not artist_data:
                continue
                
            tracks = self.catalog.from_bytes(artist_data)
            candidates = [t for t in tracks if t not in history]
            
            if candidates:
                return random.choice(candidates)
                
        return self.fallback.recommend_next(None, None, 0.0)