import itertools
import json
import pickle
from dataclasses import dataclass, field
from typing import List
from collections import defaultdict


@dataclass
class Track:
    track: int
    artist: str
    title: str
    album: str
    pop: float
    genre: List[int] = field(default=lambda: [])
    recommendations: List[int] = field(default=lambda: [])


@dataclass
class UGD:
    user: int
    tracks: List[int] = field(default=lambda: [])
    genre: List[int] = field(default=lambda: [])


class Catalog:
    """
    A helper class used to load track data upon server startup
    and store the data to redis.
    """

    def __init__(self, app):
        self.app = app
        self.tracks = []
        self.rtr = []
        self.top_tracks = []

    def load(self, catalog_path):
        self.app.logger.info(f"Loading tracks from {catalog_path}")
        with open(catalog_path) as catalog_file:
            for j, line in enumerate(catalog_file):
                data = json.loads(line)
                self.tracks.append(
                    Track(
                        data["track"],
                        data["artist"],
                        data["title"],
                        data["album"],
                        data["pop"],
                        data.get("genre", []),
                        data.get("recommendations", []),
                    )
                )
        self.app.logger.info(f"Loaded {j + 1} tracks")
        return self

    def load_users(self, catalog_path):
        self.app.logger.info(f"Loading users UGD from {catalog_path}")
        with open(catalog_path) as catalog_file:
            for j, line in enumerate(catalog_file):
                data = json.loads(line)
                self.rtr.append(
                    UGD(
                        data["user"],
                        data.get("tracks", []),
                        data.get("genre", []),
                    )
                )
        self.app.logger.info(f"Loaded {j + 1} RTR")
        return self

    def upload_tracks(self, redis_tracks):
        self.app.logger.info(f"Uploading tracks to redis")
        for track in self.tracks:
            redis_tracks.set(track.track, self.to_bytes(track))

    def upload_ugd(self, redis_conn, batch=50_000):
        self.app.logger.info("Uploading UGD (one blob per user)")

        redis_conn.config_set("save", "")
        redis_conn.config_set("appendonly", "no")

        grouped = defaultdict(list)
        for rec in self.rtr:
            grouped[rec.user].append({"tracks": rec.tracks, "genre": rec.genre})

        pipe, n = redis_conn.pipeline(), 0
        for user, payload in grouped.items():
            blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            pipe.set(user, blob)
            n += 1
            if n % batch == 0:
                pipe.execute()
                pipe = redis_conn.pipeline()

        pipe.execute()

        redis_conn.config_set("save", "900 1 300 10 60 10000")
        redis_conn.config_set("appendonly", "yes")

        self.app.logger.info(
            f"Uploaded {len(grouped)} users ({len(self.rtr)} rows) "
            f"in batches of {batch}"
        )

    def upload_artists(self, redis):
        self.app.logger.info(f"Uploading artists to redis")
        uploaded = 0
        sorted_tracks = sorted(self.tracks, key=lambda track: track.artist)
        for artist, tracks in itertools.groupby(
                sorted_tracks, key=lambda track: track.artist
        ):
            redis.set(artist, self.to_bytes([track.track for track in tracks]))
            uploaded += 1

        self.app.logger.info(f"Uploaded {uploaded} artists")

    def upload_recommendations(
            self, redis, redis_config_key, key_object="user", key_recommendations="tracks"
    ):
        recommendations_file_path = self.app.config[redis_config_key]
        self.app.logger.info(
            f"Uploading recommendations from {recommendations_file_path} to redis"
        )
        j = 0
        with open(recommendations_file_path) as rf:
            for line in rf:
                recommendations = json.loads(line)
                redis.set(
                    recommendations[key_object],
                    self.to_bytes(recommendations[key_recommendations]),
                )
                j += 1
        self.app.logger.info(
            f"Uploaded recommendations from {recommendations_file_path} for {j} {key_object}"
        )

    def to_bytes(self, instance):
        return pickle.dumps(instance)

    def from_bytes(self, bts):
        return pickle.loads(bts)
