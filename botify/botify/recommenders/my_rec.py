import random
from .recommender import Recommender


class MyRecommender(Recommender):
    """
    Если трек понравился — рекомендуем другого трека того же исполнителя, избегая повторов.
    Если трек был скипнут — пробуем другого исполнителя.
    Если подходящих вариантов нет — выдаем случайный трек.
    """

    def __init__(self, tracks_redis, artists_redis, catalog, fallback, memory_size=5, skip_threshold=0.3):
        self.tracks_redis = tracks_redis
        self.artists_redis = artists_redis
        self.catalog = catalog
        self.fallback = fallback
        self.memory = {}  # user -> [track_id,...]
        self.memory_size = memory_size
        self.skip_threshold = skip_threshold

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        # Получаем трек
        track_data = self.tracks_redis.get(prev_track)
        if not track_data:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        track = self.catalog.from_bytes(track_data)

        # Обновляем память
        history = self.memory.setdefault(user, [])
        history.append(prev_track)
        if len(history) > self.memory_size:
            history.pop(0)

        # Если пользователь слушал трек мало — это скип, ищем другого исполнителя
        if prev_track_time < self.skip_threshold:
            return self._recommend_different_artist(track.artist, history)

        # Иначе даем другого трека того же исполнителя, если можно
        return self._recommend_same_artist(track.artist, prev_track, history)

    def _recommend_same_artist(self, artist, prev_track, history):
        artist_data = self.artists_redis.get(artist)
        if not artist_data:
            return self.fallback.recommend_next(None, None, 0.0)

        artist_tracks = self.catalog.from_bytes(artist_data)
        candidates = [t for t in artist_tracks if t != prev_track and t not in history]

        if candidates:
            return random.choice(candidates)

        return self.fallback.recommend_next(None, None, 0.0)

    def _recommend_different_artist(self, prev_artist, history):
        # Пробуем найти похожих исполнителей или персональные рекомендации
        all_artists = self.artists_redis.keys()
        other_artists = [a for a in all_artists if a != prev_artist]

        random.shuffle(other_artists)
        for artist in other_artists:
            data = self.artists_redis.get(artist)
            if not data:
                continue
            tracks = self.catalog.from_bytes(data)
            candidates = [t for t in tracks if t not in history]
            if candidates:
                return random.choice(candidates)

        return self.fallback.recommend_next(None, None, 0.0)