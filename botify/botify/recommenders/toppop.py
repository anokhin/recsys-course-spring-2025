import json

import numpy as np

from typing import List

from .recommender import Recommender


class TopPop(Recommender):
    @staticmethod
    def load_from_json(path: str) -> List[int]:
        with open(path, "r") as f:
            return json.load(f)

    def __init__(self, top_tracks: List[int], fallback: Recommender):
        self.top_tracks = top_tracks
        self.fallback = fallback

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        if self.top_tracks:
            return int(np.random.choice(self.top_tracks))

        return self.fallback.recommend_next(user, prev_track, prev_track_time)
