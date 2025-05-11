import pickle
import random

from .recommender import Recommender
from collections import defaultdict, deque

class CustomRecommender(Recommender):
    HISTORY_KEY = "user:{uid}:history"
    HISTORY_SIZE = 100  # сколько последних треков храним
    TOP_K = 20  # обрезаем кандидатов
    THRESHOLD = 0.6  # доля прослушивания для "лайка"

    W_ARTIST = 0.45
    W_SIMILAR = 0.35
    W_FRESH = 0.20

    def __init__(self, tracks_redis, artists_redis, catalog, fallback):
        self.tracks_redis = tracks_redis
        self.artists_redis = artists_redis
        self.catalog = catalog
        self.fallback = fallback

    def _get_track(self, tid):
        data = self.tracks_redis.get(tid)
        return self.catalog.from_bytes(data) if data else None

    def _push_history(self, user, track_id):
        key = self.HISTORY_KEY.format(uid=user)
        history = self.tracks_redis.lrange(key, 0, -1) or []
        history = [int(t) for t in history]
        history.insert(0, track_id)
        history = history[: self.HISTORY_SIZE]
        pipe = self.tracks_redis.pipeline()
        pipe.delete(key)
        if history:
            pipe.rpush(key, *history)
        pipe.execute()

    def _in_history(self, user, track_id):
        key = self.HISTORY_KEY.format(uid=user)
        return self.tracks_redis.lpos(key, track_id) is not None

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        prev_track_obj = self._get_track(prev_track)
        if not prev_track_obj:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        if prev_track_time >= self.THRESHOLD:
            self._push_history(user, prev_track)

        candidates = defaultdict(float)

        artist_tracks = self.artists_redis.get(prev_track_obj.artist)
        if artist_tracks:
            artist_tracks = self.catalog.from_bytes(artist_tracks)
            for tid in artist_tracks:
                if tid != prev_track:
                    candidates[tid] += self.W_ARTIST

        similar = getattr(prev_track_obj, "similar", [])
        for tid in similar:
            if tid != prev_track:
                candidates[tid] += self.W_SIMILAR

        if not candidates:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        fresh_candidates = [
            (tid, score + self.W_FRESH)
            if not self._in_history(user, tid) else (tid, score)
            for tid, score in candidates.items()
        ]

        top = sorted(fresh_candidates, key=lambda x: -x[1])[: self.TOP_K]
        total = sum(s for _, s in top)
        probs = [s / total for _, s in top]
        chosen = random.choices([tid for tid, _ in top], probs, k=1)[0]

        return int(chosen)