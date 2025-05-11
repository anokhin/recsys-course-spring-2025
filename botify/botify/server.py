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
from botify.recommenders.toppop import TopPop
from botify.recommenders.indexed import Indexed
from botify.recommenders.hybrid_heuristic import HybridHeuristic
from botify.track import Catalog

from recommenders.sequential import Sequential

root = logging.getLogger()
root.setLevel("INFO")

app = Flask(__name__)
app.config.from_file("config.json", load=json.load)
api = Api(app)

tracks_redis = Redis(app, config_prefix="REDIS_TRACKS")
artists_redis = Redis(app, config_prefix="REDIS_ARTIST")
recommendations_ub = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_UB")
recommendations_dssm = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_DSSM")
recommendations_contextual = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_CONTEXTUAL")
recommendations_div = Redis(app, config_prefix="REDIS_TRACKS_WITH_DIVERSE_RECS")


recommendations_svd = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_DEBIAS_SVD")
recommendations_svd_ips = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_DEBIAS_SVD_IPS")

data_logger = DataLogger(app)

catalog = Catalog(app).load(app.config["TRACKS_CATALOG"])
catalog.upload_tracks(tracks_redis.connection)
catalog.upload_artists(artists_redis.connection)
catalog.upload_recommendations(
    recommendations_ub.connection, "RECOMMENDATIONS_UB_FILE_PATH"
)
catalog.upload_recommendations(
    recommendations_dssm.connection, "RECOMMENDATIONS_DSSM_FILE_PATH"
)
catalog.upload_recommendations(
    recommendations_contextual, "RECOMMENDATIONS_CONTEXTUAL_FILE_PATH",
    key_object='track', key_recommendations='recommendations'
)
catalog.upload_recommendations(
    recommendations_div, "TRACKS_WITH_DIVERSE_RECS_CATALOG_FILE_PATH",
    key_object='track', key_recommendations='recommendations'
)
catalog.upload_recommendations(
    recommendations_svd.connection, "RECOMMENDATIONS_DEBIAS_SVD_FILE_PATH"
)
catalog.upload_recommendations(
    recommendations_svd_ips.connection, "RECOMMENDATIONS_DEBIAS_SVD_IPS_FILE_PATH"
)

top_tracks = TopPop.load_from_json(app.config["TOP_TRACKS"])

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

        treatment = Experiments.HW2.assign(user)

        if treatment == Treatment.T1:
            recommender = HybridHeuristic(
                tracks_redis=tracks_redis.connection,
                artists_redis=artists_redis.connection,
                catalog=catalog,
                rec_dssm=recommendations_dssm,
                rec_contextual=recommendations_contextual,
                rec_diverse=recommendations_div,
                rec_ub=recommendations_ub,
                top_tracks_path=app.config["TOP_TRACKS"],
                epsilon=0.04,
            )
        else:
            recommender = Indexed(recommendations_dssm.connection, catalog, Random(tracks_redis.connection))

        recommendation = recommender.recommend_next(user, args.track, args.time)
        if recommendation is None:
            app.logger.warning("Recommender returned None, use Random fallback")
            recommendation = Indexed(recommendations_dssm.connection, catalog, Random(tracks_redis.connection))

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
        if hasattr(self, "recommender") and isinstance(self.recommender, HybridHeuristic):
            self.recommender.register_last(
                user, args.track, args.time, tracks_redis.connection
            )

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
