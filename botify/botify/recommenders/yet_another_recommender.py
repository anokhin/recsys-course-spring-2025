import random

from gensim.models import KeyedVectors

from .recommender  import Recommender


class YetAnotherRecommender(Recommender):
    KV = KeyedVectors.load('userbody.kv', mmap='r')

    def __init__(self, tracks_redis, catalog, fallback, topn=10):
        self.tracks_redis = tracks_redis
        self.catalog = catalog
        self.fallback = fallback
        self.topn = topn

    def recommend_next(self, user, prev_track, prev_time):
        track_data = self.tracks_redis.get(prev_track)

        track = self.catalog.from_bytes(track_data)
        track_id = str(track.track)

        if track_id not in YetAnotherRecommender.KV:
            return self.fallback.recommend_next(user, prev_track, prev_time)

        top_tracks = YetAnotherRecommender.KV.most_similar(track_id, topn=self.topn)
        top_tracks, _ = zip(*top_tracks)

        top_track = random.choices(top_tracks, k=1)[0]

        top_track_data = self.tracks_redis.get(top_track)

        return self.catalog.from_bytes(top_track_data).track
