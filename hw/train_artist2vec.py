#!/usr/bin/env python3
"""
train_artist2vec.py

Train a simple skip-gram Word2Vec model over user listening histories.
Input: JSON data files containing user listening sessions and artist metadata.
Output: gensim Word2Vec model file.
"""
import argparse
import glob
import os
import time
import json

import pandas as pd
import sys
try:
    from gensim.models import Word2Vec
except ImportError:
    print("Error: gensim library not installed. Please run `pip install gensim`.", file=sys.stderr)
    sys.exit(1)
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Artist2Vec (skip-gram Word2Vec) on raw listening data."
    )
    parser.add_argument(
        "--input-dir", type=str, default="hw/data/raw",
        help="Directory containing JSON data files with user listening logs"
    )
    parser.add_argument(
        "--output-model", type=str, default="artist2vec.model",
        help="Path to save the trained Word2Vec model"
    )
    parser.add_argument(
        "--dim", type=int, default=100,
        help="Dimensionality of artist embeddings"
    )
    parser.add_argument(
        "--window", type=int, default=10,
        help="Context window size for Word2Vec"
    )
    parser.add_argument(
        "--min-count", type=int, default=5,
        help="Minimum occurrence count of an artist to be included"
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel worker threads"
    )
    return parser.parse_args()


def load_track_to_artist_mapping(tracks_file):
    """Load track to artist mapping from tracks.json file."""
    print(f"Loading track to artist mapping from {tracks_file}...")
    track_to_artist = {}
    
    # Process the file line by line to handle large files
    with open(tracks_file, 'r') as f:
        for line in tqdm(f):
            try:
                track_data = json.loads(line.strip())
                if 'track' in track_data and 'artist' in track_data:
                    track_to_artist[track_data['track']] = track_data['artist']
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded artist information for {len(track_to_artist)} tracks")
    return track_to_artist


def main():
    args = parse_args()
    data_path = args.input_dir
    
    # First, load track to artist mapping
    tracks_file = os.path.join(data_path, "tracks.json")
    if not os.path.exists(tracks_file):
        print(f"Error: Tracks file not found at {tracks_file}", file=sys.stderr)
        sys.exit(1)
    
    track_to_artist = load_track_to_artist_mapping(tracks_file)
    
    # Load listening data from all data.json files
    print(f"Reading listening data from {data_path}...")
    data_files = glob.glob(os.path.join(data_path, "*/data.json*"))
    
    if not data_files:
        print(f"Error: No data files found in {data_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(data_files)} data files")
    
    # Load and process data in chunks
    all_user_sequences = {}
    line_count = 0
    
    for file_path in tqdm(data_files):
        with open(file_path, 'r') as f:
            for line in f:
                line_count += 1
                try:
                    event = json.loads(line.strip())
                    
                    # Skip if required fields are missing
                    if not all(k in event for k in ['user', 'track', 'timestamp']):
                        continue

                    if event['time'] < 0.8:
                        continue

                    user_id = event['user']
                    track_id = event['track']
                    
                    # Skip if we don't have artist info for this track
                    if track_id not in track_to_artist:
                        continue
                    
                    # Get artist ID for this track
                    artist = track_to_artist[track_id]
                    
                    # Add to user's sequence
                    if user_id not in all_user_sequences:
                        all_user_sequences[user_id] = {'artists': [], 'timestamps': []}
                    
                    all_user_sequences[user_id]['artists'].append(artist)
                    all_user_sequences[user_id]['timestamps'].append(event['timestamp'])
                except json.JSONDecodeError:
                    continue
    
    print(f"Processed {line_count} listening events for {len(all_user_sequences)} users")
    
    # Sort artist sequences by timestamp
    print("Sorting artist sequences by timestamp...")
    sequences = []
    for user_id, data in all_user_sequences.items():
        # Sort by timestamp
        sorted_artists = [artist for _, artist in sorted(zip(data['timestamps'], data['artists']))]
        sequences.append([str(artist) for artist in sorted_artists])
    
    print(f"Prepared {len(sequences)} user sequences for training")
    
    # Train Word2Vec model
    print("Training Word2Vec model...")
    start = time.time()
    model = Word2Vec(
        sentences=sequences,
        vector_size=args.dim,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        epochs=args.epochs,
        sg=1  # skip-gram
    )
    duration = time.time() - start
    print(f"Training completed in {duration:.2f} seconds")

    # Save full model
    print(f"Saving model to '{args.output_model}'...")
    model.save(args.output_model)

    # Save only the keyed vectors for use in Botify experiments
    kv_output = os.path.splitext(args.output_model)[0] + ".kv"
    print(f"Saving keyed vectors to '{kv_output}'...")
    model.wv.save(kv_output)

    # Print model stats
    vocab_size = len(model.wv.key_to_index)
    print(f"Model saved with vocabulary of {vocab_size} artists")
    print("Done.")


if __name__ == "__main__":  # noqa
    main()