import json
import logging
import random
import time
from dataclasses import asdict
from datetime import datetime

from flask import Flask
from flask_redis import Redis
import redis
from flask_restful import Resource, Api, abort, reqparse
from gevent.pywsgi import WSGIServer

from botify.data import DataLogger, Datum
from botify.experiment import Experiments, Treatment
from botify.recommenders.random import Random
from botify.recommenders.sticky_artist import StickyArtist
from botify.recommenders.toppop import TopPop
from botify.recommenders.indexed import Indexed
from botify.recommenders.lfm import LightFMRecommender
from botify.track import Catalog

from recommenders.sequential import Sequential

root = logging.getLogger()
root.setLevel("INFO")

app = Flask(__name__)
app.config.from_file("config.json", load=json.load)
api = Api(app)

tracks_redis = Redis(app, "REDIS_TRACKS")
artists_redis = Redis(app, "REDIS_ARTIST")

# recommendations_svd = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_DEBIAS_SVD")
# recommendations_svd_ips = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_DEBIAS_SVD_IPS")
# recommendations_lfm = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_LFM")
# recommendations_dssm = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_DSSM")

data_logger = DataLogger(app)

catalog = Catalog(app).load(app.config["TRACKS_CATALOG"])

catalog.upload_tracks(tracks_redis.connection)
catalog.upload_artists(artists_redis.connection)
# catalog.upload_recommendations(
#     recommendations_svd.connection, "RECOMMENDATIONS_DEBIAS_SVD_FILE_PATH"
# )
# catalog.upload_recommendations(
#     recommendations_svd_ips.connection, "RECOMMENDATIONS_DEBIAS_SVD_IPS_FILE_PATH"
# )
# catalog.upload_recommendations(
#     recommendations_lfm.connection, "RECOMMENDATIONS_LFM_FILE_PATH"
# )
# catalog.upload_recommendations(
#     recommendations_dssm.connection, "RECOMMENDATIONS_DSSM_FILE_PATH"
# )

top_tracks = TopPop.load_from_json("./data/top_tracks.json")

parser = reqparse.RequestParser()
parser.add_argument("track", type=int, location="json", required=True)
parser.add_argument("time", type=float, location="json", required=True)


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
    fallback4 = StickyArtist(tracks_redis.connection, artists_redis.connection, catalog)
    recommenders_by_treatment = {
        Treatment.T1: LightFMRecommender(catalog, tracks_redis.connection, fallback4, topn=100, p=0.1),
        Treatment.C: StickyArtist(tracks_redis.connection, artists_redis.connection, catalog),
    }
        
    def post(self, user: int):
        start = time.time()

        args = parser.parse_args()

        treatment = Experiments.HW2.assign(user)
        recommendation = NextTrack.recommenders_by_treatment[treatment].recommend_next(user, args.track, args.time)

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

app.logger.info(f"Botify service stared")

if __name__ == "__main__":
    http_server = WSGIServer(("", 5001), app)
    http_server.serve_forever()
