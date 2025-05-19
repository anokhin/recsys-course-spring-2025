from typing import List, Dict, Any
import random
import json

import numpy as np
from scipy import stats

from .random import Random
from .recommender import Recommender


class MVP(Recommender):
    THRESHOLD = 0.35

    def __init__(
            self,
            tracks_redis,
            artists_redis,
            catalog,
            users_ugd,
            session_tracks,
            lfm_recs,
            dpp_recs

    ):
        self.tracks_redis = tracks_redis
        self.artists_redis = artists_redis
        self.catalog = catalog
        self.users_ugd = users_ugd
        self.fallback = Random(tracks_redis)
        self.session_tracks = session_tracks
        self.lfm_recs = lfm_recs
        self.dpp_recs = dpp_recs

    def recommend_next(
            self,
            user_id,
            prev_track_id,
            prev_track_time
    ):

        """
        Recommend the next track for a user.
        If the previous track was played beyond the THRESHOLD, recommend within the same genre;
        otherwise, recommend a track from a different genre/from lfm recommendations.
        """
        track = self._get_track(prev_track_id)

        profile = self._get_user_profile(user_id)

        general_genre = self._select_general_genre(track.genre, profile)

        session_key = f"session_tracks:{user_id}"
        entry = {
            "track_id": prev_track_id,
            "time": prev_track_time,
            "genre": track.genre
        }

        self.session_tracks.rpush(session_key, json.dumps(entry))
        session = self._get_session_info(user_id)

        lfm_rec = self.lfm_recs.get(user_id)
        personalized_track = random.choice(list(self.catalog.from_bytes(lfm_rec)))

        # bad effect
        # if len(session) % 5 == 0:
        #     dpp_rec = self.dpp_recs.get(user_id)
        #     return random.choice(list(self.catalog.from_bytes(dpp_rec)))

        if len(session) >= 10:

            session_time = self._get_session_time(session)
            time_lower, _ = self._get_conf(session_time)

            if prev_track_time >= time_lower:
                return self._recommend_same_genre(prev_track_id, general_genre, profile)

        if prev_track_time >= self.THRESHOLD:
            return self._recommend_same_genre(prev_track_id, general_genre, profile)

        # return self._recommend_different_genre(track.genre, profile)
        return personalized_track

    def _get_track(self, track_id: int):
        """Fetch and deserialize a track by its ID."""
        data = self.tracks_redis.get(track_id)
        if data is None:
            raise ValueError(f"Track not found: {track_id}")
        return self.catalog.from_bytes(data)

    def _get_user_profile(self, user_id: int) -> List[Dict[str, Any]]:
        """Fetch and deserialize user listening profile."""
        data = self.users_ugd.get(user_id)
        return self.catalog.from_bytes(data)

    @staticmethod
    def _select_general_genre(
            genre,
            profile,
    ):
        """
        If the track has multiple genres, attempt to find a related genre from user profile;
        otherwise return the original genre.
        """
        if len(genre) <= 1:
            return genre

        user_genres = [item['genre'] for item in profile]
        mixed = [g for g in user_genres if genre[0] in g and set(g) != set(genre)]
        return random.choice(mixed) if mixed else genre

    @staticmethod
    def _recommend_same_genre(
            prev_track_id,
            genre,
            profile,
    ):
        """
        Recommend a different track within the same genre if possible,
        otherwise fallback to any track from the first profile entry.
        """
        tracks_in_genre = [
            t
            for entry in profile
            if set(entry['genre']) == set(genre)
            for t in entry['tracks']
        ]
        candidates = [t for t in tracks_in_genre if t != prev_track_id]

        if candidates:
            return random.choice(candidates)

        return random.choice(profile[0]['tracks'])

    @staticmethod
    def _recommend_different_genre(
            original_genre,
            profile
    ):
        """
        Recommend a track from a different genre grouping than the original.
        """
        genre_map = {}
        for entry in profile:
            key = tuple(entry['genre'])
            genre_map.setdefault(key, []).extend(entry['tracks'])

        # Exclude the original genre
        alternatives = [tracks for g, tracks in genre_map.items() if list(g) != original_genre]
        if alternatives:
            return random.choice(random.choice(alternatives))

        # Fallback to any track
        return random.choice(profile[0]['tracks'])

    def _get_session_info(self, user_id):
        session_key = f"session_tracks:{user_id}"
        raw = self.session_tracks.lrange(session_key, 0, -1)
        return [item.decode('utf-8') for item in raw]

    @staticmethod
    def _get_session_time(data):
        return [json.loads(item)['time'] for item in data]

    @staticmethod
    def _get_session_tracks(data):
        return [json.loads(item)['track_id'] for item in data]

    @staticmethod
    def _get_session_genres(data):
        return [json.loads(item)["genre"] for item in data]

    @staticmethod
    def _get_session_pop(data):
        return [
            d["pop"]
            for d in map(json.loads, data)
            if "pop" in d
        ]

    @staticmethod
    def _get_conf(data):
        data = np.array(data)
        n = len(data)
        mean = data.mean()

        sem = stats.sem(data)

        confidence = 0.95
        alpha = 1 - confidence
        df = n - 1
        t_crit = stats.t.ppf(1 - alpha / 2, df)

        h = sem * t_crit
        ci_lower = mean - h
        ci_upper = mean + h

        return ci_lower, ci_upper
