import json
import logging
import time
from dataclasses import asdict
from datetime import datetime

import torch
from flask import Flask
from flask_redis import Redis
from flask_restful import Resource, Api, abort, reqparse
from gevent.pywsgi import WSGIServer

from botify.data import DataLogger, Datum
from botify.experiment import Experiments, Treatment
from botify.recommenders.my_recommender import MyRecommender
from botify.recommenders.random import Random
from botify.recommenders.sticky_artist import StickyArtist
from botify.recommenders.toppop import TopPop
from botify.track import Catalog

root = logging.getLogger()
root.setLevel("INFO")

app = Flask(__name__)
app.config.from_file("config.json", load=json.load)
api = Api(app)

tracks_redis = Redis(app, config_prefix="REDIS_TRACKS")
artists_redis = Redis(app, config_prefix="REDIS_ARTIST")


my_recommendations = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_MY")
recommendations_dssm = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_DSSM")
history_redis = Redis(app, config_prefix="REDIS_HISTORY")
fav_redis = Redis(app, config_prefix="REDIS_FAV")
statistic_genre_redis = Redis(app, config_prefix="REDIS_STATISTIC_GENRE")
statistic_artist_redis = Redis(app, config_prefix="REDIS_STATISTIC_ARTIST")

data_logger = DataLogger(app)

catalog = Catalog(app).load(app.config["TRACKS_CATALOG"])
catalog.upload_tracks(tracks_redis.connection)
catalog.upload_artists(artists_redis.connection)
catalog.upload_recommendations(
    my_recommendations.connection, "RECOMMENDATIONS_MY_FILE_PATH"
)
catalog.upload_recommendations(
    recommendations_dssm.connection, "RECOMMENDATIONS_DSSM_FILE_PATH"
)

catalog.upload_recommendations(
    fav_redis.connection, 'FAV_PATH', key_recommendations='fav_tracks'
)

top_tracks = TopPop.load_from_json("./data/top_tracks.json")

parser = reqparse.RequestParser()
parser.add_argument("track", type=int, location="json", required=True)
parser.add_argument("time", type=float, location="json", required=True)


track_embeddings = torch.load('./data/learned_track_embeddings_interaction_model_v3_2_bert_bert_base_multilingual.pth', map_location=torch.device('cpu'), weights_only=False)


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
        treatment = Experiments.PERSONALIZED.assign(user)

        if treatment == Treatment.T1:
            recommender = MyRecommender(my_recommendations.connection, catalog, fallback, track_embeddings,
                                        history_redis.connection, fav_redis.connection, statistic_genre_redis.connection,
                                        tracks_redis.connection, statistic_artist_redis.connection)
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

app.logger.info(f"Botify service stared")

if __name__ == "__main__":
    http_server = WSGIServer(("", 5001), app)
    http_server.serve_forever()
