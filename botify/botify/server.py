import json
import logging
import random
import time
from dataclasses import asdict
from datetime import datetime

from flask import Flask
from flask_redis import Redis
from flask_restful import Resource, Api, abort, reqparse
from gevent.pywsgi import WSGIServer

from botify.data import DataLogger, Datum
from botify.experiment import Experiments, Treatment
from botify.recommenders.random import Random
from botify.recommenders.sticky_artist import StickyArtist
from botify.recommenders.mvp import MVP
from botify.recommenders.toppop import TopPop
from botify.recommenders.indexed import Indexed
from botify.track import Catalog

root = logging.getLogger()
root.setLevel("INFO")

app = Flask(__name__)
app.config.from_file("config.json", load=json.load)
api = Api(app)

tracks_redis = Redis(app, config_prefix="REDIS_TRACKS")
artists_redis = Redis(app, config_prefix="REDIS_ARTIST")

data_logger = DataLogger(app)

catalog = Catalog(app).load(app.config["TRACKS_CATALOG"])
catalog.upload_tracks(tracks_redis.connection)
catalog.upload_artists(artists_redis.connection)

users_ugd = Redis(app, config_prefix="REDIS_USERS_UGD")
catalog.load_users(app.config["REDIS_USERS_UGD_PATH"])
catalog.upload_ugd(users_ugd.connection)

session_tracks = Redis(app, config_prefix="REDIS_SESSION_TRACKS")
session_tracks.flushdb()

top_tracks = TopPop.load_from_json("./data/top_tracks.json")

lfm_recs = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_LFM_RECS")
catalog.upload_recommendations(lfm_recs.connection, "RECOMMENDATIONS_LFM_RECS_FILE_PATH")

dpp_recs = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_DPP_RECS")
catalog.upload_recommendations(dpp_recs.connection, "RECOMMENDATIONS_DPP_RECS_FILE_PATH")

parser = reqparse.RequestParser()
parser.add_argument("track", type=int, location="json", required=True)
parser.add_argument("time", type=float, location="json", required=True)

recommendations_top_shuffle = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_TOP_SHUFFLE")
catalog.upload_recommendations(
    recommendations_top_shuffle.connection, "RECOMMENDATIONS_TOP_SHUFFLE_FILE_PATH")


class Hello(Resource):
    def get(self):
        return {
            "status": "alive",
            "message": "welcome to botify, the best toy music recommender",
        }


class Track(Resource):
    def get(self, track: int):
        data = tracks_redis.connection.get(track)
        if data is not None:
            return asdict(catalog.from_bytes(data))
        else:
            abort(404, description="Track not found")


class NextTrack(Resource):
    def post(self, user: int):
        start = time.time()

        args = parser.parse_args()

        fallback = Random(tracks_redis.connection)
        treatment = Experiments.MVP.assign(user)
        #treatment = Experiments.TOP_SHUFFLE.assign(user)

        if treatment == Treatment.T1:
            #recommender = Indexed(recommendations_top_shuffle.connection, catalog, fallback)
            recommender = MVP(tracks_redis.connection, artists_redis.connection, catalog, users_ugd.connection,
                              session_tracks.connection, lfm_recs.connection, dpp_recs.connection)
        else:
            recommender = StickyArtist(tracks_redis.connection, artists_redis.connection, catalog)

        recommendation = recommender.recommend_next(user, args.track, args.time)

        data_logger.log(
            "next",
            Datum(
                int(datetime.now().timestamp() * 1000),
                user,
                args.track,
                args.time,
                time.time() - start,
                recommendation,
            ),
        )
        return {"user": user, "track": recommendation}


class LastTrack(Resource):
    def post(self, user: int):
        start = time.time()
        args = parser.parse_args()
        data_logger.log(
            "last",
            Datum(
                int(datetime.now().timestamp() * 1000),
                user,
                args.track,
                args.time,
                time.time() - start,
            ),
        )
        return {"user": user}


api.add_resource(Hello, "/")
api.add_resource(Track, "/track/<int:track>")
api.add_resource(NextTrack, "/next/<int:user>")
api.add_resource(LastTrack, "/last/<int:user>")

app.logger.info(f"Botify service started")

if __name__ == "__main__":
    http_server = WSGIServer(("", 5001), app)
    http_server.serve_forever()
