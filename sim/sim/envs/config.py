from dataclasses import dataclass

import marshmallow_dataclass


@dataclass()
class TrackCatalogConfig:
    track_meta_path: str
    track_embeddings_path: str


@dataclass
class UserCatalogConfig:
    user_catalog_path: str
    default_interest_neighbours: int = 10
    default_consume_bias: float = 5.0
    default_consume_sharpness: float = 1.0
    default_session_budget: int = 5
    default_artist_discount_gamma: float = 0.8


@dataclass()
class RemoteRecommenderConfig:
    host: str
    port: int


@dataclass
class RecEnvConfig:
    track_catalog_config: TrackCatalogConfig
    user_catalog_config: UserCatalogConfig
    remote_recommender_config: RemoteRecommenderConfig


RecEnvConfigSchema = marshmallow_dataclass.class_schema(RecEnvConfig)
