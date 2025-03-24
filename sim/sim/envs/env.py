import gym
import numpy as np
from gym.spaces import Discrete, Dict

from .config import RecEnvConfig
from .track import TrackCatalog
from .user import UserCatalog


class RecEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, config: RecEnvConfig):
        super(RecEnv, self).__init__()
        self.config = config

        self.track_catalog = TrackCatalog(config.track_catalog_config)
        self.user_catalog = UserCatalog(config.user_catalog_config)

        # At each step you suggest a track, so each action is a single track ID
        self.action_space = Discrete(self.track_catalog.size())

        # We need to provide a user ID to the recommender and the initial track
        self.observation_space = Dict(
            user=Discrete(self.user_catalog.size()),
            track=Discrete(self.track_catalog.size()),
        )

        self.user = None
        self.session = None

        self.reset()

    def step(self, recommendation: int):
        assert self.action_space.contains(recommendation), str(recommendation)
        playback_time = self.user.consume(
            recommendation, self.session, self.track_catalog
        )
        return self.session.observe(), playback_time, self.session.finished, None

    def reset(self):
        self.user = self.user_catalog.sample_user()
        self.session = self.user.new_session(self.track_catalog)
        return self.session.observe()

    def render(self, mode="human", close=False):
        print(f"Current session: {self.session}")

    def seed(self, seed=None):
        np.random.seed(seed)
