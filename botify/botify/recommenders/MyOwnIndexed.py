from .recommender import Recommender


class MyOwnIndexed(Recommender):
    def __init__(self, recommendations_redis, catalog, fallback):
        self.recommendations_redis = recommendations_redis
        self.fallback = fallback
        self.catalog = catalog

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        recommendations = self.recommendations_redis.get(user)

        if recommendations is not None:
            recommendations = list(self.catalog.from_bytes(recommendations))
            if prev_track not in recommendations:
                return  recommendations[0]
            prev_track_index = recommendations.index(prev_track)
            return recommendations[(prev_track_index + 1) % len(recommendations)]
                
        else:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)
