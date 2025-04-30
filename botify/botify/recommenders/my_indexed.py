import random
import faiss
import numpy as np
import json

from .recommender import Recommender

class MyIndexed(Recommender):
    def __init__(
        self,
        recommendations_redis,
        catalog,
        fallback,
        user_info_redis,
        recent_tracks_redis,
    ):
        self.recommendations_redis = recommendations_redis
        self.catalog = catalog
        self.fallback = fallback
        self.user_info_redis = user_info_redis
        self.recent_tracks_redis = recent_tracks_redis

    def _get_recent_tracks(self, user: int):
        key = f"recent_tracks:{user}"
        recent = self.recent_tracks_redis.lrange(key, 0, 10)
        return set(map(int, recent or []))

    def _save_recent_track(self, user: int, track: int):
        key = f"recent_tracks:{user}"
        self.recent_tracks_redis.lpush(key, track)
        self.recent_tracks_redis.ltrim(key, 0, 10)

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        recent_tracks = self._get_recent_tracks(user)
        recommendations = self.recommendations_redis.get(user)
        recs = list(self.catalog.from_bytes(recommendations))

        if prev_track is None:
            track = recs[0]
            self._save_recent_track(user, track)
            return track

        if prev_track_time <= 0.7:
            user_data_raw = self.user_info_redis.get(user)
            if user_data_raw:
                fav_tracks = list(self.catalog.from_bytes(recommendations))
                random.shuffle(fav_tracks)
                for track in fav_tracks:
                    if track not in recent_tracks:
                        self._save_recent_track(user, track)
                        return track

        for idx in recs:
            track = int(idx)
            if track not in recent_tracks:
                self._save_recent_track(user, track)
                return track

        return self.fallback.recommend_next(user, prev_track, prev_track_time)
