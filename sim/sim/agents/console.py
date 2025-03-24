from typing import Dict, Optional
from urllib.parse import urlunsplit

import requests

from .recommender import Recommender
from ..envs import RemoteRecommenderConfig

SCHEME = "http"


class ConsoleRecommender(Recommender):
    """Provide recommendations manually, ftw"""

    def __init__(self, config: RemoteRecommenderConfig):
        self.config = config

    def recommend(self, observation: Dict[str, int], reward: float, done: bool) -> int:
        previous_track_info = self.load_track_info(observation["track"])
        print(
            f"Got previous track {self.format(previous_track_info)} for user {observation['user']} with reward {reward}"
        )

        recommendation = None
        while recommendation is None:
            print("Enter recommended track:")
            recommendation = self.parse_input()

        return recommendation

    def parse_input(self) -> Optional[int]:
        try:
            recommendation = int(input())
        except ValueError as va:
            return None

        track_info = self.load_track_info(recommendation)
        if track_info is None:
            print(f"Could not load track {recommendation}")
            return None
        else:
            print(f"Recommending track {self.format(track_info)}")
            return recommendation

    def load_track_info(self, track) -> Optional[Dict]:
        url = urlunsplit(
            (SCHEME, f"{self.config.host}:{self.config.port}", f"track/{track}", {}, "")
        )
        response = requests.get(url)

        if response.status_code != 200:
            return None

        return response.json()

    def format(self, track_info):
        return f"'{track_info['title']}' by '{track_info['artist']}'"

    def __repr__(self):
        return "console"
