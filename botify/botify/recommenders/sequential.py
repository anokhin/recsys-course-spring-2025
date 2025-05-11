from typing import List

from .recommender import Recommender


class Sequential(Recommender):
    def __init__(self, recommendations_redis, catalog, fallback):
        self.recommendations_redis = recommendations_redis
        self.fallback = fallback
        self.catalog = catalog

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        recommendations = self.recommendations_redis.get(user)

        if recommendations is not None:
            recommendations = list(self.catalog.from_bytes(recommendations))
            next_rec_index = self.get_track_index(prev_track, recommendations)
            return recommendations[(next_rec_index + 1) % len(recommendations)]
        else:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

    def get_track_index(self, track: int, recommendations: List[int]) -> int:
        for ix, recommendation in enumerate(recommendations):
            if recommendation == track:
                return ix
        return 0
