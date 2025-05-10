import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Any, List, Optional

from flask import Flask, current_app
from flask_redis import FlaskRedis
from flask_restful import Api, Resource, abort, reqparse
from gevent.pywsgi import WSGIServer
from pydantic import BaseModel, ValidationError

from botify.data import DataLogger, Datum
from botify.experiment import Experiments, Treatment
from botify.recommenders.contextual import Contextual
from botify.recommenders.indexed import Indexed
from botify.recommenders.random import Random
from botify.recommenders.sticky_artist import StickyArtist
from botify.recommenders.toppop import TopPop
from botify.track import Catalog
from botify.recommenders.rec import rec


class AppConfig(BaseModel):
    REDIS_TRACKS: Dict[str, Any]
    REDIS_ARTIST: Dict[str, Any]
    REDIS_RECOMMENDATIONS_DSSM: Dict[str, Any]
    REDIS_ARTIST_MEMORY: Dict[str, Any]
    REDIS_TRACK_MEMORY: Dict[str, Any]
    TRACKS_CATALOG: str
    RECOMMENDATIONS_DSSM_FILE_PATH: str
    TOP_TRACKS_PATH: str = "./data/top_tracks.json"
    LOGGING_CONFIG: Dict[str, Any] = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }

redis_store = FlaskRedis(decode_responses=False)

def configure_logging(config: AppConfig):
    logging.basicConfig(**config.LOGGING_CONFIG)
    logging.captureWarnings(True)

def create_app(config_path: str = "config.json") -> Flask:
    app = Flask(__name__)
    
    try:
        with open(config_path) as f:
            raw_config = json.load(f)
        config = AppConfig(**raw_config)
        app.config.update(config.dict())
    except (FileNotFoundError, ValidationError) as e:
        logging.error(f"Configuration error: {e}")
        raise

    try:
        configure_logging(config)
        redis_store.init_app(app)
        api = Api(app)
        
        catalog = Catalog(app).load(config.TRACKS_CATALOG)
        catalog.upload_all(
            tracks_redis=redis_store,
            artists_redis=redis_store,
            recommendations_redis=redis_store,
            recommendations_path=config.RECOMMENDATIONS_DSSM_FILE_PATH
        )
        
        api.add_resource(HealthCheck, "/")
        api.add_resource(TrackResource, "/track/<int:track_id>")
        api.add_resource(NextTrackResource, "/next/<int:user_id>")
        api.add_resource(LastTrackResource, "/last/<int:user_id>")
        
        return app
    except Exception as e:
        logging.error(f"App initialization failed: {e}")
        raise

class HealthCheck(Resource):
    def get(self) -> Dict[str, str]:
        return {"status": "ok", "service": "botify-api"}

class TrackResource(Resource):
    def get(self, track_id: int) -> Dict[str, Any]:
        try:
            track_data = redis_store.get(f"track:{track_id}")
            if not track_data:
                abort(404, message="Track not found")
            return asdict(current_app.catalog.from_bytes(track_data))
        except Exception as e:
            logging.error(f"Track lookup error: {e}")
            abort(500, message="Internal server error")

class NextTrackResource(Resource):
    _parser = reqparse.RequestParser()
    _parser.add_argument("track", type=int, required=True, location="json")
    _parser.add_argument("time", type=float, required=True, location="json")
    
    def _init_recommenders(self) -> Dict[Treatment, reqparse.Recommender]:
        base_fallback = Random(redis_store)
        return {
            Treatment.T1: rec(
                recommendations_redis=redis_store,
                catalog=current_app.catalog,
                fallbacks=[
                    StickyArtist(redis_store, current_app.catalog),
                    Indexed(redis_store, current_app.catalog, base_fallback),
                    TopPop.load_from_json(current_app.config["TOP_TRACKS_PATH"])
                ],
                track_memory_redis=redis_store,
                artist_memory_redis=redis_store
            ),
            Treatment.DEFAULT: Indexed(
                redis_store, 
                current_app.catalog, 
                base_fallback
            )
        }
    
    def post(self, user_id: int) -> Dict[str, int]:
        start_time = time.time()
        try:
            args = self._parser.parse_args()
            recommenders = self._init_recommenders()
            
            treatment = Experiments.rec.assign(user_id)
            recommender = recommenders.get(treatment, recommenders[Treatment.DEFAULT])
            
            recommendation = recommender.recommend_next(
                user_id, args["track"], args["time"]
            )
            
            current_app.data_logger.log(
                "next",
                Datum(
                    timestamp=int(datetime.now().timestamp() * 1000),
                    user=user_id,
                    track=args["track"],
                    time=args["time"],
                    response_time=time.time() - start_time,
                    recommendation=recommendation
                )
            )
            
            return {"user": user_id, "track": recommendation}
        except ValidationError as e:
            abort(400, message=str(e))
        except Exception as e:
            logging.error(f"Recommendation error: {e}")
            abort(500, message="Recommendation failed")

class LastTrackResource(Resource):
    def post(self, user_id: int) -> Dict[str, int]:
        try:
            current_app.data_logger.log_cleanup(user_id)
            redis_store.delete(f"user:{user_id}:tracks")
            redis_store.delete(f"user:{user_id}:artists")
            return {"user": user_id}
        except Exception as e:
            logging.error(f"Cleanup error: {e}")
            abort(500, message="Cleanup failed")

if __name__ == "__main__":
    try:
        app = create_app()
        server = WSGIServer(("0.0.0.0", 5001), app)
        logging.info("Botify API service started")
        server.serve_forever()
    except Exception as e:
        logging.critical(f"Failed to start service: {e}")
        raise