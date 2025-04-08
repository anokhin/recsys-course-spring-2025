import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
import random

from flask import Flask
from flask_redis import Redis
from flask_restful import Resource, Api, abort, reqparse
from gevent.pywsgi import WSGIServer

from botify.recommenders.contextual import Contextual
from botify.data import DataLogger, Datum
from botify.experiment import Experiments, Treatment
from botify.recommenders.random import Random
from botify.recommenders.sticky_artist import StickyArtist
from botify.recommenders.toppop import TopPop
from botify.recommenders.indexed import Indexed
from botify.recommenders.my_best_recommender import MyBestRecommender

from botify.track import Catalog


root = logging.getLogger()
root.setLevel("INFO")

app = Flask(__name__)
app.config.from_file("config.json", load=json.load)
api = Api(app)

tracks_redis = Redis(app, config_prefix="REDIS_TRACKS")
artists_redis = Redis(app, config_prefix="REDIS_ARTIST")

# recommendations_ub = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_UB")
# recommendations_lfm = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_LFM")
# recommendations_ncf = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_NCF")
# recommendations_contextual = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_CONTEXTUAL")
recommendations_dssm = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_DSSM")

artist_memory = Redis(app, config_prefix="REDIS_ARTIST_MEMORY")
track_memory = Redis(app, config_prefix="REDIS_TRACK_MEMORY")

data_logger = DataLogger(app)

catalog = Catalog(app).load(app.config["TRACKS_CATALOG"])
catalog.upload_tracks(tracks_redis.connection)

catalog.upload_artists(artists_redis.connection)
# catalog.upload_recommendations(recommendations_ub.connection, "RECOMMENDATIONS_UB_FILE_PATH")
# catalog.upload_recommendations(recommendations_lfm.connection, "RECOMMENDATIONS_LFM_FILE_PATH")
# catalog.upload_recommendations(recommendations_ncf.connection, "RECOMMENDATIONS_NCF_FILE_PATH")
# catalog.upload_recommendations(recommendations_contextual.connection, "RECOMMENDATIONS_CONTEXTUAL_FILE_PATH",
#                                key_object="track", key_recommendations="recommendations")
catalog.upload_recommendations(recommendations_dssm.connection, "RECOMMENDATIONS_DSSM_FILE_PATH")

top_tracks = TopPop.load_from_json(r"./data/top_tracks.json")

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
    def post(self, user: int):
        start = time.time()
        args = parser.parse_args()

        fallback = Random(tracks_redis.connection)

        fallback_sa = StickyArtist(tracks_redis.connection, artists_redis.connection, catalog)
        fallback_dssm= Indexed(recommendations_dssm.connection, catalog, fallback)
        fallback_toppop = TopPop(top_tracks, fallback)

        fallbacks_group = [fallback_sa, fallback_dssm, fallback_toppop]

        treatment = Experiments.MY_BEST_RECOMMENDER_EXPERIMENT.assign(user)

        if treatment == Treatment.T1:
            recommender = MyBestRecommender(recommendations_dssm_redis=recommendations_dssm.connection,
                                            user_artist_memory_redis=artist_memory.connection,
                                            user_track_memory_redis=track_memory.connection,
                                            catalog=catalog,
                                            tracks_redis=tracks_redis.connection,
                                            artists_redis=artists_redis.connection,
                                            fallbacks=fallbacks_group,
                                            max_track_num=2,
                                            max_artist_num=4,
                                            time_threshold_big=0.5,
                                            time_threshold_small=0.05)
        else:
            recommender = Indexed(recommendations_dssm.connection, catalog, fallback)

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

        track_memory.set(user, catalog.to_bytes([]))
        artist_memory.set(user, catalog.to_bytes([]))

        return {"user": user}


api.add_resource(Hello, "/")
api.add_resource(Track, "/track/<int:track>")
api.add_resource(NextTrack, "/next/<int:user>")
api.add_resource(LastTrack, "/last/<int:user>")

app.logger.info(f"Botify service stared")

if __name__ == "__main__":
    http_server = WSGIServer(("", 5001), app)
    http_server.serve_forever()
