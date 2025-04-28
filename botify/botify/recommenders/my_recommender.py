from datetime import datetime
import random
from botify.recommenders.recommender import Recommender
import torch
import torch.nn.functional as F


COOLDOWN_GOOD_TRACK = 5  # Кол-во треков, через которое можно вновь порекомендовать понравившийся трек
COOLDOWN_BAD_TRACK = 10  # Кол-во треков, через которое можно вновь порекомендовать плохой трек
TIME_BAD_TRACK = 0.3  # Доля трека, по которой определяем нравится ли пользователю трек


class MyRecommender(Recommender):
    def __init__(self, recommendations_redis, catalog, fallback, embedding_map, redis_history, users_fav_redis, redis_stat_genre, redis_tracks, redis_stat_artist):
        self.recommendations_redis = recommendations_redis  # Redis для рекомендованных треков из НС
        self.fallback = fallback
        self.catalog = catalog
        self.redis_history = redis_history  # Redis для хранения истории пользователей
        self.embedding_map = embedding_map  # Словарь тензоров треков
        self.users_fav_redis = users_fav_redis  # Redis для хранения любимых треков пользователя
        self.redis_stat_genre = redis_stat_genre  # Redis для хранения статистики жанров по пользователям
        self.redis_tracks = redis_tracks  # Все треки
        self.redis_stat_artist = redis_stat_artist  # Статистика по авторам


    def _get_banned_tracks(self, history_tracks):
        j = 0
        banned = set()
        for i in range(len(history_tracks) - 1, max(len(history_tracks) - COOLDOWN_BAD_TRACK, -1), - 1):
            if j <= COOLDOWN_GOOD_TRACK:
                banned.add(history_tracks[i][0])
            if history_tracks[i][-1] <= TIME_BAD_TRACK:
                banned.add(history_tracks[i][0])
            j += 1
        return banned

    def check_banned_genre(self, track, stats_genre, stats_artist):
        # Получение списка жанров у трека
        lst_g = list(self.catalog.from_bytes(self.redis_tracks.get(track)).genre)

        # Для каждого жанра находим вероятность, что он окажется плохим. Далее находим среднее значение по всем жанрам
        p_g = sum([(stats_genre[i][0] / stats_genre[i][1] if i in stats_genre else 0) for i in lst_g]) / len(lst_g)

        # Получение артиста
        art = self.get_artist(track)
        # Вычисляем вероятность, что артист не понравится
        p_a = (stats_artist[art][0] / stats_artist[art][1]) if art in stats_artist else 0

        # Вычисляем вероятность, что трек не понравится
        p_g = 0.7*p_g + 0.3*p_a

        p = random.random()
        return p > p_g


    def get_artist(self, track):
        return self.catalog.from_bytes(self.redis_tracks.get(track)).artist

    def get_genre(self, track):
        return set(self.catalog.from_bytes(self.redis_tracks.get(track)).genre)

    def calc_similarity(self, track: int, track_list: list[int], calc_min: bool) -> int:
        prev_emb = self.embedding_map[track]
        emb_list = [self.embedding_map[rec] for rec in track_list]
        candidate_emb_tensor = torch.stack(emb_list)
        # Эмбеддинг предыдущего трека для батчинга: (1, размерность)
        prev_track_emb_unsqueezed = prev_emb.unsqueeze(0)

        # Сходства: тензор формы (кол-во_кандидатов,)
        similarities = F.cosine_similarity(prev_track_emb_unsqueezed, candidate_emb_tensor, dim=1)
        if calc_min:
            best_candidate_index = torch.argmin(similarities).item()
        else:
            best_candidate_index = torch.argmax(similarities).item()
        best_candidate_id = track_list[best_candidate_index]
        best_score = similarities[best_candidate_index].item()

        return best_candidate_id

    def update_statistics(self, user, prev_track, stats_genre, stat_artist, num):
        for i in self.get_genre(prev_track):
            stats_genre[i] = (stats_genre[i][0] + num, stats_genre[i][1] + 1) if i in stats_genre else (num, 1)
        self.redis_stat_genre.set(user, self.catalog.to_bytes(stats_genre))

        art = self.get_artist(prev_track)
        stat_artist[art] = (stat_artist[art][0] + num, stat_artist[art][1] + 1) if art in stat_artist else (num, 1)
        self.redis_stat_artist.set(user, self.catalog.to_bytes(stat_artist))

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        # Получение рекомендаций
        recommendations = self.recommendations_redis.get(user)

        # Получение истории пользователя
        history_tracks = self.redis_history.get(user)
        history_tracks = list(self.catalog.from_bytes(history_tracks) if history_tracks else [])
        history_tracks.append((prev_track, prev_track_time))

        # Получение недавно прослушанных треков, которые нужно исключить из рекомендаций
        banned_tracks = self._get_banned_tracks(history_tracks)

        # Получение статистики прослушивания по жанрам {'genre': (кол-во непонравившихся треков, кол-во прослушанных треков)}
        stats_genre = self.redis_stat_genre.get(user)
        stats_genre = self.catalog.from_bytes(stats_genre) if stats_genre else dict()

        # Получение статистики прослушивания по артистам {'artist': (кол-во непонравившихся треков, кол-во прослушанных треков)}
        stat_artist = self.redis_stat_artist.get(user)
        stat_artist = self.catalog.from_bytes(stat_artist) if stat_artist else dict()

        # Обновляем историю пользователя
        self.redis_history.set(user, self.catalog.to_bytes(history_tracks))

        if recommendations is not None:
            recommendations_list = self.catalog.from_bytes(recommendations)
        else:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        # По времени прослушивания определяем, понравилось ли пользователю
        if prev_track_time <= TIME_BAD_TRACK: # Пользователю не понравился трек

            # Получение списка любимых треков пользователя с учётом недавно прослушанных треков
            # Исключаем треки, которые имеют плохую статистику прослушивания по жанру и автору
            fav_list = self.catalog.from_bytes(self.users_fav_redis.get(user))
            fav_list = [x for x in fav_list if x not in banned_tracks and self.check_banned_genre(x, stats_genre, stat_artist)]

            # Обновляем статистику прослушивания
            self.update_statistics(user, prev_track, stats_genre, stat_artist, 1)

            if len(fav_list) == 0:
                return self.fallback.recommend_next(user, prev_track, prev_track_time)

            # Возвращаем трек, который наиболее далёкий от текущего по косинусному расстоянию
            return self.calc_similarity(prev_track, fav_list, True)


        # Если пользователю понравился трек:

        # Из списка рекомендованных треков убираем недавно прослушанные, а также учитываем статистику по жанрам и авторам
        recommendations_list = [x for x in recommendations_list if x not in banned_tracks and self.check_banned_genre(x, stats_genre, stat_artist)]
        if len(recommendations_list) == 0:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        # Обновляем статистику прослушивания
        self.update_statistics(user, prev_track, stats_genre, stat_artist, 0)

        # Возвращаем трек, который наиболее близкий к текущему по косинусному расстоянию
        return self.calc_similarity(prev_track, recommendations_list, False)

