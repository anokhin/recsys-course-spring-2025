from __future__ import annotations

from typing import Optional, List
import random
import time

import numpy as np

from .recommender import Recommender
from .random import Random
from .toppop import TopPop
from .indexed import Indexed

OFF_KEY = "off:dssm:{}"
TRACK_MEM_KEY = "mem:tracks:{}"
ART_MEM_KEY = "mem:artists:{}"
LIKED_KEY = "liked:{}"


class _RedisListReader:
    def __init__(self, redis_conn, catalog):
        self.r = redis_conn
        self.catalog = catalog

    def list_for(self, key: int | str) -> List[int]:
        data = self.r.get(key)
        if data is None:
            return []
        return list(self.catalog.from_bytes(data))


class HybridHeuristic(Recommender):

    def __init__(
        self,
        *,
        tracks_redis,
        artists_redis,
        catalog,
        rec_dssm,
        rec_contextual,
        rec_diverse,
        rec_ub,
        top_tracks_path: str,
        epsilon: float = 0.02,
    ) -> None:
        self.catalog = catalog
        self.tracks_r = tracks_redis
        self.artists_r = artists_redis

        # readers
        self.dssm = _RedisListReader(rec_dssm, catalog)
        self.ctx = _RedisListReader(rec_contextual, catalog)
        self.div = _RedisListReader(rec_diverse, catalog)
        self.ub = _RedisListReader(rec_ub, catalog)

        self.random = Random(tracks_redis)
        self.top_pop = TopPop(TopPop.load_from_json(top_tracks_path), self.random)
        self.indexed = Indexed(rec_dssm.connection, catalog, Random(tracks_redis.connection))

        self.epsilon = epsilon
        self.ttl_sec_off = 24 * 3600
        self.ttl_sec_mem = 3 * 3600

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        r = self.tracks_r

        if random.random() < self.epsilon:
            return self.top_pop.recommend_next(user, prev_track, prev_track_time)

        def allowed(tid: int) -> bool:
            if tid is None:
                return False
            pipe = r.pipeline()
            pipe.sismember(TRACK_MEM_KEY.format(user), tid)
            artist_id = self._artist_of(tid)
            if artist_id is None:
                return False
            pipe.hget(ART_MEM_KEY.format(user), artist_id)
            dup, cnt = pipe.execute()
            if dup:
                return False
            if int(cnt or 0) >= 2:
                return False

            pipe = r.pipeline()
            pipe.sadd(TRACK_MEM_KEY.format(user), tid)
            pipe.hincrby(ART_MEM_KEY.format(user), artist_id, 1)
            pipe.expire(TRACK_MEM_KEY.format(user), self.ttl_sec_mem)
            pipe.expire(ART_MEM_KEY.format(user), self.ttl_sec_mem)
            pipe.execute()
            return True

        dssm_list = self.dssm.list_for(user)
        if dssm_list:
            off_key = OFF_KEY.format(user)
            off = r.incr(off_key) - 1
            r.expire(off_key, self.ttl_sec_off)
            cand = dssm_list[off % len(dssm_list)]
            if allowed(cand):
                return cand

        if prev_track_time >= 0.8:
            cand = self._random_from_artist(prev_track)
            if allowed(cand):
                return cand

        if prev_track_time <= 0.3:
            ctx_candidates = self.ctx.list_for(prev_track)[:3]
            cand = ctx_candidates[0] if ctx_candidates else None
            if allowed(cand):
                return cand

        cand = self._first_allowed(self.div.list_for(prev_track), allowed)
        if cand is not None:
            return cand

        cand = self._first_allowed(self.ub.list_for(user), allowed)
        if cand is not None:
            return cand

        return self.indexed.recommend_next(user, prev_track, prev_track_time)

    def _artist_of(self, track_id: int) -> Optional[int]:
        raw = self.tracks_r.get(track_id)
        if raw is None:
            return None
        track = self.catalog.from_bytes(raw)
        return getattr(track, "artist", None)

    def _random_from_artist(self, track_id: int) -> Optional[int]:
        artist_id = self._artist_of(track_id)
        if artist_id is None:
            return None
        data = self.artists_r.get(artist_id)
        if data is None:
            return None
        tracks = list(self.catalog.from_bytes(data))
        if not tracks:
            return None
        return int(np.random.choice(tracks, 1)[0])

    @staticmethod
    def _first_allowed(seq: List[int], allowed_fn) -> Optional[int]:
        for tid in seq:
            if allowed_fn(tid):
                return tid
        return None


    def register_last(self, user: int, track: int, play_time: float, redis_conn) -> None:
        off_key = OFF_KEY.format(user)
        if play_time >= 0.99:
            redis_conn.zincrby(LIKED_KEY.format(user), 1.0, track)
            redis_conn.expire(LIKED_KEY.format(user), 24 * 3600)
        elif play_time <= 0.3:
            redis_conn.delete(off_key)
        else:
            redis_conn.expire(off_key, 24 * 3600)
