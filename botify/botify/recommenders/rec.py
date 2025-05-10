import random
from .recommender import Recommender
from botify.recommenders.random import Random


class rec(Recommender):
    def __init__(
        self,
        recommendations_dssm_redis,
        user_artist_memory_redis,
        user_track_memory_redis,
        catalog,
        tracks_redis,
        artists_redis,
        fallbacks,
        max_artist_num=3,
        max_track_num=2,
        time_threshold_big=0.5,
        time_threshold_small=0.05,
        dssm_reset_index=15,
        max_memory_entries=100,
    ):
        self.recommendations_dssm = recommendations_dssm_redis
        self.fallbacks = fallbacks
        self.catalog = catalog
        self.tracks_redis = tracks_redis
        self.artists_redis = artists_redis

        self.user_track_memory = user_track_memory_redis
        self.user_artist_memory = user_artist_memory_redis

        self.max_artist_num = max_artist_num
        self.max_track_num = max_track_num

        self.time_threshold_big = time_threshold_big
        self.time_threshold_small = time_threshold_small
        self.dssm_reset_index = dssm_reset_index

        self.max_memory_entries = max_memory_entries

    def __get_next_idx_dssm(self, prev_idx: int, dssm_lst: list, prev_track_time: float) -> int:
        """
        determine next index in DSSM recommendations with reset logic
        """
        if prev_idx >= self.dssm_reset_index:
            if prev_track_time < self.time_threshold_small:
                return 0
            elif prev_track_time < self.time_threshold_big:
                return (prev_idx + random.randint(1, 5)) % len(dssm_lst)
        return (prev_idx + 1) % len(dssm_lst)

    def __get_flags(self, user: int, next_track: int, prev_listened_track: int) -> tuple[bool, bool]:
        """
        check if track/artist should be blocked based on recommendation history
        """
        # get track history
        track_mem = self._get_user_memory(user, self.user_track_memory)
        track_flag = (
            track_mem.count(next_track) >= self.max_track_num 
            or next_track == prev_listened_track
        )

        # get artist history
        artist_mem = self._get_user_memory(user, self.user_artist_memory)
        track_data = self.catalog.from_bytes(self.tracks_redis.get(next_track))
        artist_flag = artist_mem.count(track_data.artist) >= self.max_artist_num

        return track_flag, artist_flag

    def _get_user_memory(self, user: int, redis_client) -> list:
        """
        helper to safely retrieve and deserialize user memory
        """
        memory_bytes = redis_client.get(user)
        return list(self.catalog.from_bytes(memory_bytes)) if memory_bytes else []

    def _update_user_memory(self, user: int, item: int, redis_client, max_entries: int) -> None:
        """
        update user memory with new item while enforcing size limits
        """
        memory = self._get_user_memory(user, redis_client)
        memory.append(item)
        # truncate to prevent unlimited growth
        if len(memory) > max_entries:
            memory = memory[-max_entries:]
        redis_client.set(user, self.catalog.to_bytes(memory))

    def __add_track_and_artist(self, user: int, track: int) -> None:
        """
        update both track and artist recommendation histories
        """
        self._update_user_memory(user, track, self.user_track_memory, self.max_memory_entries)
        track_data = self.catalog.from_bytes(self.tracks_redis.get(track))
        self._update_user_memory(
            user, 
            track_data.artist, 
            self.user_artist_memory, 
            self.max_memory_entries
        )

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        # try DSSM recommendations first
        dssm_recs = self.recommendations_dssm.get(user)
        if dssm_recs is not None:
            dssm_lst = list(self.catalog.from_bytes(dssm_recs))
            if dssm_lst:  # handle empty DSSM list
                start_idx = 0
                if prev_track in dssm_lst:
                    prev_idx = dssm_lst.index(prev_track)
                    start_idx = self.__get_next_idx_dssm(prev_idx, dssm_lst, prev_track_time)

                # search for valid track in DSSM recommendations
                for i in range(len(dssm_lst)):
                    current_idx = (start_idx + i) % len(dssm_lst)
                    candidate = dssm_lst[current_idx]
                    track_flag, artist_flag = self.__get_flags(user, candidate, prev_track)

                    if not track_flag and not artist_flag:
                        self.__add_track_and_artist(user, candidate)
                        return candidate

        # fallback to secondary recommenders
        for fallback in self.fallbacks:
            candidate = fallback.recommend_next(user, prev_track, prev_track_time)
            track_flag, artist_flag = self.__get_flags(user, candidate, prev_track)
            if not track_flag and not artist_flag:
                self.__add_track_and_artist(user, candidate)
                return candidate

        # final fallback to random with safety checks
        random_rec = Random(self.tracks_redis)
        for _ in range(50):
            candidate = random_rec.recommend_next(user, prev_track, prev_track_time)
            track_flag, artist_flag = self.__get_flags(user, candidate, prev_track)
            if not track_flag and not artist_flag:
                break

        self.__add_track_and_artist(user, candidate)
        return candidate