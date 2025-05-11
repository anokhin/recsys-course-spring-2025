import numpy as np
from pathlib import Path
from .recommender import Recommender   # homework interface
from gensim.models import KeyedVectors

class LightFMRecommender(Recommender):

    MAX_TRACK_REPEAT = 2
    MAX_ARTIST_REPEAT = 3

    def __init__(self,
                 catalog,
                 tracks_redis,
                 fallback,
                 topn=10,
                 p=0.1,
                 model_dir="data"):
        artefacts = Path(model_dir)
        self.kv      = KeyedVectors.load(str(artefacts/"w2v.kv"))
        self.topn = topn
        self.p = p

        self.fallback = fallback

        self.catalog, self.tracks_redis = catalog, tracks_redis

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        if str(prev_track) in self.kv.key_to_index:
            candidates = self.kv.most_similar(
                str(prev_track),
                topn=self.topn,
                restrict_vocab=None,
            )

            idx = np.random.geometric(p=self.p, size=1)[0]
            if idx >= len(candidates):
                return self.fallback.recommend_next(user, prev_track, prev_track_time)

            return int(candidates[idx][0])
        return self.fallback.recommend_next(user, prev_track, prev_track_time)