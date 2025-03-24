from typing import Dict


class Recommender(object):
    def recommend(self, observation: Dict[str, int], reward: float, done: bool) -> int:
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        pass
