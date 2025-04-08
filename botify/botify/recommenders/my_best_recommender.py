import random
from .recommender import Recommender
from botify.recommenders.random import Random


class MyBestRecommender(Recommender):
    def __init__(self, recommendations_dssm_redis, user_artist_memory_redis, user_track_memory_redis,
                 catalog, tracks_redis, artists_redis, fallbacks, max_artist_num=3, max_track_num=2,
                 time_threshold_big=0.5, time_threshold_small=0.05):
        self.recommendations_dssm = recommendations_dssm_redis
        self.fallbacks = fallbacks
        self.catalog = catalog
        self.tracks_redis = tracks_redis
        self.artists_redis = artists_redis

        # Память о пользователях
        self.user2track = user_artist_memory_redis
        self.user2artist = user_track_memory_redis

        # Ограничения на число треков и артистов, которые можно рекомендовать одному пользователю
        self.max_artist_num = max_artist_num
        self.max_track_num = max_track_num

        # Отсечки по времени прослушивания для рекомендаций от dssm
        self.time_threshold_big = time_threshold_big
        self.time_threshold_small = time_threshold_small

    def __get_next_idx_dssm(self, prev_idx: int, dssm_lst: list, prev_track_time: float) -> int:
        next_idx = (prev_idx + 1) % len(dssm_lst)
        if prev_idx >= 15 and prev_track_time < self.time_threshold_small:
            next_idx = 0
        elif prev_idx >= 15 and prev_track_time < self.time_threshold_big:
            next_idx = (prev_idx + random.randint(1, 5)) % len(dssm_lst)
        return next_idx

    def __get_flags(self, user: int, next_track: int, prev_track: int) -> tuple:
        track_mem_bytes = self.user2track.get(user)
        artist_mem_bytes = self.user2artist.get(user)

        track_mem = list(self.catalog.from_bytes(track_mem_bytes)) if track_mem_bytes else []
        artist_mem = list(self.catalog.from_bytes(artist_mem_bytes)) if artist_mem_bytes else []

        next_track_data = self.catalog.from_bytes(self.tracks_redis.get(next_track))
        next_track_artist = next_track_data.artist

        track_flag = (sum(1 for track in track_mem if track == next_track) >= self.max_track_num or
                      next_track == prev_track)
        artist_flag = sum(1 for artist in artist_mem if artist == next_track_artist) >= self.max_artist_num

        # Если любой из флагов True, то ищем другую рекомендацию
        return track_flag, artist_flag

    def __add_track_and_artist(self, user: int, track: int) -> None:
        track_mem_bytes = self.user2track.get(user)
        artist_mem_bytes = self.user2artist.get(user)

        track_mem = list(self.catalog.from_bytes(track_mem_bytes)) if track_mem_bytes else []
        artist_mem = list(self.catalog.from_bytes(artist_mem_bytes)) if artist_mem_bytes else []

        track_data = self.catalog.from_bytes(self.tracks_redis.get(track))
        track_artist = track_data.artist

        track_mem.append(track)
        artist_mem.append(track_artist)

        self.user2track.set(user, self.catalog.to_bytes(track_mem))
        self.user2artist.set(user, self.catalog.to_bytes(artist_mem))

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        recommendations_dssm = self.recommendations_dssm.get(user)

        # DSSM
        if recommendations_dssm is not None:
            dssm_lst = list(self.catalog.from_bytes(recommendations_dssm))

            if prev_track not in dssm_lst:
                next_idx = 0
            else:
                prev_idx = dssm_lst.index(prev_track)
                next_idx = self.__get_next_idx_dssm(prev_idx, dssm_lst, prev_track_time)

            next_track = dssm_lst[next_idx]
            track_flag, artist_flag = self.__get_flags(user, next_track, prev_track)
            start_idx = next_idx

            while (track_flag or artist_flag) and next_idx != start_idx:
                next_idx = (next_idx + 1) % len(dssm_lst)
                next_track = dssm_lst[next_idx]
                track_flag, artist_flag = self.__get_flags(user, next_track, prev_track)

            if not track_flag and not artist_flag:
                final_track = next_track
                self.__add_track_and_artist(user, final_track)
                return final_track

        # FALLBACKS
        for fallback_recommender in self.fallbacks:
            next_track = fallback_recommender.recommend_next(user, prev_track, prev_track_time)
            track_flag, artist_flag = self.__get_flags(user, next_track, prev_track)

            if not track_flag and not artist_flag:
                final_track = next_track
                self.__add_track_and_artist(user, final_track)
                return final_track

        # RANDOM
        random_recommender = Random(self.tracks_redis)
        next_track = random_recommender.recommend_next(user, prev_track,
                                                       prev_track_time)
        track_flag, artist_flag = self.__get_flags(user, next_track, prev_track)
        num_iter = 0

        while (track_flag or artist_flag) and num_iter < 50:
            next_track = random_recommender.recommend_next(user, prev_track, prev_track_time)
            track_flag, artist_flag = self.__get_flags(user, next_track, prev_track)
            num_iter += 1

        if (not track_flag and not artist_flag) or num_iter >= 50:
            final_track = next_track
            self.__add_track_and_artist(user, final_track)
            return final_track