import random
import numpy as np
import pickle
import os
# библиотека для работы с графами
import networkx as nx
from collections import defaultdict, Counter
# базовый класс для рекомендателей
from .recommender import Recommender
from ..track import Catalog

class GraphBasedRecommender(Recommender):
    def __init__(self, tracks_redis, artists_redis, catalog, model_path='./data/graph_model.pkl'):
        # хранение информации о треках и артистах
        self.tracks_redis = tracks_redis
        self.artists_redis = artists_redis
        # catalog - инструмент для преобразования данных из Redis в объекты
        self.catalog: Catalog = catalog
        # model_path - путь для сохранения/загрузки модели
        self.model_path = model_path
        
        # граф для представления того, как пользователи переходят от одного трека к другому и между артистами
        # узлы - треки, ребра - переходы между ними
        self.track_transition_graph = None
        self.artist_transition_graph = None
        
        # предпочтения пользователей по трекам
        # {user_id: {track_id: rating, ...}, ...}
        self.user_track_preferences = defaultdict(Counter)
        
        # предпочтения пользователей по артистам
        # {user_id: {artist_id: rating, ...}, ...}
        self.user_artist_preferences = defaultdict(Counter)
        
        # результаты PageRank для треков
        # {track_id: pagerank, ...}
        # чем выше pagerank, тем больше вероятность, что пользователь прослушает этот трек
        self.track_pageranks = {}
        
        # история треков для каждого пользователя
        # {user_id: [track_id1, track_id2, ...], ...}
        # чтобы избегать повторов
        self.user_track_history = defaultdict(list)
        
        # время прослушивания для каждого трека
        # {track_id: [time1, time2, ...], ...}
        self.track_listen_times = defaultdict(list)
        
        # длительность каждого трека (seconds)
        # {track_id: duration, ...}
        self.track_durations = {}
        
        # загружаем существующую модель или создаем новую
        self._load_or_create_model()
        
    def _load_or_create_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.track_transition_graph = model_data.get('track_graph', nx.DiGraph())
                    self.artist_transition_graph = model_data.get('artist_graph', nx.DiGraph())
                    self.track_pageranks = model_data.get('track_pageranks', {})
                    self.track_listen_times = model_data.get('track_listen_times', defaultdict(list))
                    self.track_durations = model_data.get('track_durations', {})
                    
                    self.user_track_history = model_data.get('user_track_history', defaultdict(list))
                    for user, history in self.user_track_history.items():
                        # для экономии памяти
                        self.user_track_history[user] = history[-50:]
            except Exception as e:
                print(f"Error while loading model: {e}")
                self._create_new_model()
        else:
            self._create_new_model()
    
    def _create_new_model(self):
        self.track_transition_graph = nx.DiGraph()
        self.artist_transition_graph = nx.DiGraph()
        self.track_pageranks = {}
        self.track_listen_times = defaultdict(list)
        self.track_durations = {}
        self.user_track_history = defaultdict(list)
    
    def _update_transition_graphs(self, user, prev_track, next_track, prev_track_time):
        # обновить графы переходов между треками и между артистами
        
        prev_track_data = self.tracks_redis.get(prev_track)
        next_track_data = self.tracks_redis.get(next_track)
        
        if prev_track_data is not None and next_track_data is not None:
            prev_track_obj = self.catalog.from_bytes(prev_track_data)
            next_track_obj = self.catalog.from_bytes(next_track_data)
            
            if prev_track not in self.track_durations:
                self.track_durations[prev_track] = prev_track_obj.duration
            if next_track not in self.track_durations:
                self.track_durations[next_track] = next_track_obj.duration

            # если 60% времени прослушивания, то хороший сигнал
            # диапазон от 0.1 до 1.5
            listen_quality = min(prev_track_time / max(prev_track_obj.duration * 0.6, 0.1), 1.5)
            
            self.track_listen_times[prev_track].append(prev_track_time)
            if len(self.track_listen_times[prev_track]) > 100:
                self.track_listen_times[prev_track] = self.track_listen_times[prev_track][-100:]
            
            if listen_quality >= 0.8:
                transition_weight = 1.0 + listen_quality
                
                # граф переходов треков
                if self.track_transition_graph.has_edge(prev_track, next_track):
                    # если ребро существует, увеличить вес
                    weight = self.track_transition_graph[prev_track][next_track].get('weight', 1)
                    self.track_transition_graph[prev_track][next_track]['weight'] = weight + transition_weight
                else:
                    # иначе добавить новое ребро
                    self.track_transition_graph.add_edge(prev_track, next_track, weight=transition_weight)
                
                # граф переходов артистов
                if self.artist_transition_graph.has_edge(prev_track_obj.artist, next_track_obj.artist):
                    weight = self.artist_transition_graph[prev_track_obj.artist][next_track_obj.artist].get('weight', 1)
                    self.artist_transition_graph[prev_track_obj.artist][next_track_obj.artist]['weight'] = weight + transition_weight
                else:
                    self.artist_transition_graph.add_edge(prev_track_obj.artist, next_track_obj.artist, weight=transition_weight)
            
            self.user_track_preferences[user][prev_track] += listen_quality
            self.user_artist_preferences[user][prev_track_obj.artist] += listen_quality
            
            self.user_track_history[user].append(prev_track)
            if len(self.user_track_history[user]) > 50:
                self.user_track_history[user] = self.user_track_history[user][-50:]
            
            # пересчитываем PageRank только в 1% случаев
            # то есть модель остается относительно актуальной
            # но ресурсы уменьшаются в 100 раз
            if random.random() < 0.01:
                self._update_pageranks()
                self._save_model()
    
    def _get_average_listen_time(self, track_id):
        # среднее время прослушивания трека
        if track_id in self.track_listen_times and self.track_listen_times[track_id]:
            return sum(self.track_listen_times[track_id]) / len(self.track_listen_times[track_id])
        return 0
    
    def _get_listen_time_ratio(self, track_id):
        # отношение среднего времени прослушивания к длительности трека
        avg_time = self._get_average_listen_time(track_id)
        # fallback: 180 sec если длительность None
        duration = self.track_durations.get(track_id, 180)
        if duration > 0:
            return min(avg_time / duration, 1.0)
        return 0
    
    def _update_pageranks(self):
        # обновить  PageRank для треков с учетом времени прослушивания
        if len(self.track_transition_graph) > 5:
            for u, v, d in self.track_transition_graph.edges(data=True):
                listen_ratio_u = self._get_listen_time_ratio(u)
                d['weight'] = d.get('weight', 1.0) * (1.0 + listen_ratio_u)
            
            try:
                # обновить PageRank
                self.track_pageranks = nx.pagerank(self.track_transition_graph, weight='weight', max_iter=100)
            except Exception:
                # fallback: skip
                pass
    
    def _compute_pagerank(self, user, prev_track):
        # PageRank для данного пользователя и трека
        if len(self.track_transition_graph) < 5:
            return self.track_pageranks
        
        # у пользователя есть история предпочтений
        if user in self.user_track_preferences and len(self.user_track_preferences[user]) > 0:
            personalization = {k: v for k, v in self.user_track_preferences[user].items()}
            
            # усилить влияние текущего трека
            if prev_track in self.track_transition_graph:
                personalization[prev_track] = personalization.get(prev_track, 0) + 2.0
            
            # последние прослушанные треки с убывающими весами
            if user in self.user_track_history:
                for i, track in enumerate(reversed(self.user_track_history[user][-5:])):
                    recency_weight = 1.0 / (i + 1)
                    if track in self.track_transition_graph:
                        personalization[track] = personalization.get(track, 0) + recency_weight
            
            total = sum(personalization.values())
            if total > 0:
                personalization = {k: v/total for k, v in personalization.items()}
                
            try:
                return nx.pagerank(self.track_transition_graph, weight='weight', 
                                  personalization=personalization, max_iter=100)
            except Exception:
                # fallback: обычный PageRank
                return self.track_pageranks
        
        # fallback: обычный PageRank (если нет истории предпочтений)
        return self.track_pageranks
    
    def _get_artist_tracks(self, artist_id):
        artist_data = self.artists_redis.get(artist_id)
        if artist_data is not None:
            return self.catalog.from_bytes(artist_data)
        return []
    
    def _filter_recently_played(self, user, tracks, n=3):
        if user not in self.user_track_history or len(self.user_track_history[user]) < n:
            return tracks
        
        recent_tracks = set(self.user_track_history[user][-n:])
        
        return [t for t in tracks if t not in recent_tracks]
    
    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        # основная логика рекомендаций
        try:
            # fallback к random если нет prev_track
            if prev_track is None:
                all_tracks = [int(t) for t in self.tracks_redis.keys()]
                if all_tracks:
                    return int(random.choice(all_tracks))
                return 1
                
            # обновляем данные о текущем треке
            if prev_track in self.tracks_redis.keys():
                self.track_listen_times[prev_track].append(prev_track_time)
                if len(self.track_listen_times[prev_track]) > 100:
                    self.track_listen_times[prev_track] = self.track_listen_times[prev_track][-100:]
                
                self.user_track_history[user].append(prev_track)
                if len(self.user_track_history[user]) > 50:
                    self.user_track_history[user] = self.user_track_history[user][-50:]
            
            # обрабатываю случай, когда пользователь слушал трек <15% времени
            # в таком случае предложить другой трек того же артиста
            track_data = None
            try:
                track_data = self.tracks_redis.get(prev_track)
                if track_data is not None:
                    track = self.catalog.from_bytes(track_data)
                    if track.duration > 0 and prev_track_time / track.duration < 0.15:
                        artist_tracks = self._get_artist_tracks(track.artist)
                        if artist_tracks:
                            filtered_tracks = self._filter_recently_played(user, artist_tracks)
                            if not filtered_tracks:
                                filtered_tracks = artist_tracks
                            
                            # треки с хорошим соотношением времени прослушивания (>50%)
                            good_tracks = []
                            for t in filtered_tracks:
                                if t == prev_track:
                                    continue
                                listen_ratio = self._get_listen_time_ratio(t)
                                if listen_ratio >= 0.5:
                                    good_tracks.append((t, listen_ratio))
                            
                            if good_tracks:
                                weights = [ratio for _, ratio in good_tracks]
                                tracks = [t for t, _ in good_tracks]
                                if sum(weights) > 0:
                                    weights = [w/sum(weights) for w in weights]
                                    return int(np.random.choice(tracks, p=weights))
                                return int(random.choice(tracks))
            except Exception as e:
                print(f"Error while processing next recommendation for low listening time: {e}")
            
            # основная логика рекомендаций::::

            # если текущий трек не в графе::
            if prev_track not in self.track_transition_graph or len(list(self.track_transition_graph.successors(prev_track))) == 0:
                # тогда используем рекомендации на основе артиста
                if track_data is not None:
                    track = self.catalog.from_bytes(track_data)
                    artist_tracks = self._get_artist_tracks(track.artist)
                    if artist_tracks:
                        filtered_tracks = self._filter_recently_played(user, artist_tracks)
                        if not filtered_tracks:
                            filtered_tracks = artist_tracks
                        
                        if prev_track in filtered_tracks:
                            filtered_tracks.remove(prev_track)
                        
                        if filtered_tracks:
                            # треки с хорошим соотношением времени прослушивания (>50%)
                            # [(track_id, weight), ...]
                            tracks_with_weights = []
                            for t in filtered_tracks:
                                avg_time = self._get_average_listen_time(t)
                                tracks_with_weights.append((t, 1.0 + avg_time))
                            
                            if tracks_with_weights:
                                tracks = [t for t, _ in tracks_with_weights]
                                weights = [w for _, w in tracks_with_weights]
                                
                                if sum(weights) > 0:
                                    weights = [w/sum(weights) for w in weights]
                                    # нормализуем веса, random choice
                                    return int(np.random.choice(tracks, p=weights))
                                return int(random.choice(tracks))
                
                # если не можем использовать рекомендации на основе артиста, то используем треки с высоким PageRank
                if self.track_pageranks:
                    top_tracks = sorted(self.track_pageranks.items(), key=lambda x: x[1], reverse=True)[:20]
                    
                    # недавно проигранные
                    filtered_top_tracks = [(t, r) for t, r in top_tracks if t not in self.user_track_history.get(user, [])]
                    
                    if not filtered_top_tracks:
                        filtered_top_tracks = top_tracks
                    
                    # [(track_id, weight), ...]
                    tracks_with_weights = []
                    for t, pagerank in filtered_top_tracks:
                        avg_listen_time = self._get_average_listen_time(t)


                        weight = pagerank * (1.0 + avg_listen_time)


                        tracks_with_weights.append((t, weight))
                    
                    if tracks_with_weights:
                        tracks = [t for t, _ in tracks_with_weights]
                        weights = [w for _, w in tracks_with_weights]
                        
                        if sum(weights) > 0:
                            weights = [w/sum(weights) for w in weights]
                            return int(np.random.choice(tracks, p=weights))
                        return int(random.choice(tracks))
                
                # fallback: random choice
                all_tracks = [int(t) for t in self.tracks_redis.keys()]
                all_tracks = [t for t in all_tracks if t != prev_track]
                if all_tracks:
                    return int(random.choice(all_tracks))
                return int(prev_track)
            
            # если текущий трек в графе::

            # персонализированный PageRank для пользователя
            personalized_ranks = self._compute_pagerank(user, prev_track)
            # треки, которые следуют за текущим и недавно проигрывались
            neighbors = list(self.track_transition_graph.successors(prev_track))
            filtered_neighbors = self._filter_recently_played(user, neighbors)
            if not filtered_neighbors:
                filtered_neighbors = neighbors
            
            
            candidates = []
            for neighbor in filtered_neighbors:
                # от текущего трека к кандидату
                transition_weight = self.track_transition_graph[prev_track][neighbor].get('weight', 1)
                
                page_rank = personalized_ranks.get(neighbor, 0)
                avg_listen_time = self._get_average_listen_time(neighbor)
                
                combined_weight = (0.3 * transition_weight + 0.3 * page_rank + 0.4 * avg_listen_time)
                
                # !!  чтобы рекомендации не были слишком предсказуемыми
                diversity_factor = random.uniform(0.8, 1.2)
                final_weight = combined_weight * diversity_factor
                
                candidates.append((neighbor, final_weight))
            
            
            if candidates:
                # нормализуем веса для random choice
                tracks = [t for t, _ in candidates]
                weights = [w for _, w in candidates]
                
                if sum(weights) > 0:
                    weights = [w / sum(weights) for w in weights]
                    next_track = np.random.choice(tracks, p=weights)
                else:
                    next_track = random.choice(tracks)
                
                self._update_transition_graphs(user, prev_track, next_track, prev_track_time)
                
                return int(next_track)
            
            # иначе рекомендуем трек того же артиста
            if track_data is not None:
                track = self.catalog.from_bytes(track_data)
                artist_tracks = self._get_artist_tracks(track.artist)
                if artist_tracks:
                    filtered_tracks = [t for t in artist_tracks if t != prev_track]
                    if filtered_tracks:
                        return int(random.choice(filtered_tracks))
            
            # fallback: текущий трек todo
            return int(prev_track)
        
        except Exception as e:
            # fallback: текущий трек
            print(f"Error in recommender: {e}")
            try:
                return int(prev_track)
            except Exception:
                # fallback: случайный трек
                all_tracks = [int(t) for t in self.tracks_redis.keys()]
                if all_tracks:
                    return int(random.choice(all_tracks))
                return 1
    
    def _save_model(self):
        try:
            model_data = {
                'track_graph': self.track_transition_graph,
                'artist_graph': self.artist_transition_graph,
                'track_pageranks': self.track_pageranks,
                'track_listen_times': self.track_listen_times,
                'track_durations': self.track_durations,
                'user_track_history': self.user_track_history
            }
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            print(f"Error when saveng model: {e}")
