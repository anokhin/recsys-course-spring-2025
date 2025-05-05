import random
from typing import Optional
from .recommender import Recommender

class SessionRecommenderDionis(Recommender):
    def __init__(
        self,
        history_redis,
        catalog,
        recommendations_redis: dict,
        fallback: Recommender,
        indexed_sample_size: int = 15,
        use_lfm: bool = True,
        use_lgcf_m: bool = True,
        use_dssm: bool = True,
        thresholds: dict = None,
        back_counts: dict = None,
        max_history_length: int = 100,
    ):
        self.redis = history_redis
        self.catalog = catalog
        self.recs = recommendations_redis
        self.fallback = fallback

        self.indexed_sample_size = indexed_sample_size
        self.use_lfm = use_lfm
        self.use_lgcf_m = use_lgcf_m
        self.use_dssm = use_dssm
        self.thresholds = thresholds or {
            'lgcf': 0.35, 'lfm': 0.30, 'lgcf_m': 0.35, 'dssm': 0.30
        }
        self.back_counts = back_counts or {
            'lgcf': 4, 'lfm': 2, 'lgcf_m': 4, 'dssm': 2
        }
        self.max_history_length = max_history_length

    def _history_key(self, user: int):
        return f"user:{user}:hist:tracks"

    def _record_track(self, user: int, track: int):
        key = self._history_key(user)
        pipe = self.redis.pipeline()
        pipe.rpush(key, track)
        pipe.ltrim(key, -self.max_history_length, -1)
        pipe.execute()

    def _get_seen(self, user: int):
        key = self._history_key(user)
        raw = self.redis.lrange(key, 0, -1)
        return set(int(x) for x in raw)

    def _state_keys(self, user: int):
        return f"user:{user}:state:method", f"user:{user}:state:failcount"

    def _get_state(self, user: int):
        km, kf = self._state_keys(user)
        method = self.redis.get(km)
        fc = self.redis.get(kf)
        if method is None or fc is None:
            return 'lgcf', 0
        return method.decode(), int(fc)

    def _set_state(self, user: int, method: str, fc: int):
        km, kf = self._state_keys(user)
        pipe = self.redis.pipeline()
        pipe.set(km, method)
        pipe.set(kf, fc)
        pipe.execute()

    def _increment_or_reset(self, prev_time: float, method: str, current_fc: int):
        if prev_time < self.thresholds[method]:
            return current_fc + 1
        return 0

    def _pick_indexed(self, user: int, method: str, seen: set) -> Optional[int]:
        conn = self.recs.get(method)
        if not conn:
            return None
        raw = conn.get(user)
        if not raw:
            return None
        all_recs = self.catalog.from_bytes(raw)
        unrecs = [t for t in all_recs if t not in seen]
        if not unrecs:
            return None
        return random.choice(unrecs[: self.indexed_sample_size])

    def recommend_next(self, user: int, prev_track: int, prev_time: float) -> int:
        # Записываем трек в историю
        self._record_track(user, prev_track)
        seen = self._get_seen(user)

        # Текущее состояние
        method, failcount = self._get_state(user)

        # Обновляем счётчик неуспехов
        new_fc = self._increment_or_reset(prev_time, method, failcount)

        # Переключаемся
        if method == 'lgcf' and new_fc >= self.back_counts['lgcf'] and self.use_lfm:
            method, new_fc = 'lfm', 0
        elif method == 'lfm' and new_fc >= self.back_counts['lfm'] and self.use_lgcf_m:
            method, new_fc = 'lgcf_m', 0
        elif method == 'lgcf_m' and new_fc >= self.back_counts['lgcf_m'] and self.use_dssm:
            method, new_fc = 'dssm', 0
        elif method == 'dssm' and new_fc >= self.back_counts['dssm']:
            method, new_fc = 'lgcf', 0

        # Сохраняем состояние
        self._set_state(user, method, new_fc)

        # Пытаемся выдать рекомендации посредством метода
        if method in ('lgcf', 'lfm', 'dssm'):
            rec = self._pick_indexed(user, method, seen)
        else:  # возвращаемся к LGCF с пары (случайная рекомендация из всех)
            rec = self.fallback.recommend_next(user, prev_track, prev_time)

        if rec is None:
            return self.fallback.recommend_next(user, prev_track, prev_time)
        return rec
