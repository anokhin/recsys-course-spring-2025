import random
from gensim.models import KeyedVectors
from .recommender import Recommender


class Artist2Vec(Recommender):
    """
    Recommend a new track by finding artists similar to the last-listened artist
    in the latent embedding space (artist2vec), then sampling a track.
    """
    # load only the keyed vectors for similarity queries
    kv = KeyedVectors.load("artist2vec.kv", mmap='r')

    def __init__(self, tracks_redis, artists_redis, catalog, topn=10):
        self.tracks_redis = tracks_redis
        self.artists_redis = artists_redis
        self.catalog = catalog
        self.topn = topn

    def recommend_next(self, user, prev_track, prev_time):
        # Decode last track to artist
        track_data = self.tracks_redis.get(prev_track)
        if track_data is None:
            raise ValueError(f"Track not found: {prev_track}")
        track = self.catalog.from_bytes(track_data)
        art = str(track.artist)

        # If artist not in vocab, fallback to random track of same artist
        if art not in Artist2Vec.kv:
            artist_data = self.artists_redis.get(track.artist)
            if artist_data is None:
                raise ValueError(f"Artist not found: {track.artist}")
            artist_tracks = self.catalog.from_bytes(artist_data)
            return random.choice(artist_tracks)

        # Find nearest neighbor artists weighted by similarity
        neighbors = Artist2Vec.kv.most_similar(art, topn=self.topn)
        artists, sims = zip(*neighbors)
        # pick is the artist key (string) in the keyed vectors
        pick = random.choices(artists, weights=sims, k=1)[0]

        # Sample a random track from the chosen artist
        artist_data = self.artists_redis.get(pick)
        if artist_data is None:
            raise ValueError(f"Artist not found: {pick}")
        artist_tracks = self.catalog.from_bytes(artist_data)
        return random.choice(artist_tracks)