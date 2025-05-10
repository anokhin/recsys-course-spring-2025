import random

from .random import Random
from .recommender import Recommender

# TODO Семинар 1, Шаг 3 - Реализуем новый рекомендер, в качестве рекомендаций возвращающий
#  случайный трек того же исполнителя, что и предыдущий.
class StickyArtist(Recommender):
    def __init__(self, tracks_redis, artists_redis, catalog):
        self.fallback = Random(tracks_redis)
        self.artists_redis = artists_redis
        self.catalog = catalog
        self.tracks_redis = tracks_redis

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        # TODO Семинар 1, Шаг 3.1 - Спрашиваем в redis информацию по предыдущему треку, достаем имя исполнителя.
        track_data = self.tracks_redis.get(prev_track)
        if track_data is not None:
            track = self.catalog.from_bytes(track_data)
        else:
            raise ValueError(f"Track not found: {prev_track}")

        artist_data = self.artists_redis.get(track.artist)

        # TODO Семинар 1, Шаг 3.2 - Спрашиваем в redis все треки этого исполнителя.
        if artist_data is not None:
            artist_tracks = self.catalog.from_bytes(artist_data)
        else:
            raise ValueError(f"Artist not found: {track.artist}")

        # TODO Семинар 1, Шаг 3.3 - Рекомендуем случайный из найденных треков.
        index = random.randint(0, len(artist_tracks) - 1)
        return artist_tracks[index]
