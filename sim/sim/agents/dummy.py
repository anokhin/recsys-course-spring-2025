from typing import Dict

from .recommender import Recommender


class DummyRecommender(Recommender):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def recommend(self, observation: Dict[str, int], reward: float, done: bool) -> int:
        return self.action_space.sample()

    def __repr__(self):
        return "dummy"
