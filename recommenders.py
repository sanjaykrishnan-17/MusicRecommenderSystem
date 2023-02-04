import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz

class content_based_recommender:    
    def __init__(self, matrix):
        self.matrix_similar = matrix

    def _print_message(self, song, recom_song):
        rec_items = len(recom_song)
        
        print(f'The {rec_items} recommended songs for {song} are:')
        for i in range(rec_items):
            print(f"Number {i+1}:")
            print(f"{recom_song[i][1]} by {recom_song[i][2]} with {round(recom_song[i][0], 3)} similarity score") 
            print("--------------------")
        
    def recommend(self, recommendation):
        # Get song to find recommendations for
        song = recommendation['song']
        # Get number of songs to recommend
        number_songs = recommendation['number_songs']
        # Get the number of songs most similars from matrix similarities
        recom_song = self.matrix_similar[song][:number_songs]
        # print each item
        self._print_message(song=song, recom_song=recom_song)

class collaborative_based_recommender:
    class Recommender:
        def __init__(self, metric, algorithm, k, data, decode_id_song):
            self.metric = metric
            self.algorithm = algorithm
            self.k = k
            self.data = data
            self.decode_id_song = decode_id_song
            self.data = data
            self.model = self._recommender().fit(data)
        
        def make_recommendation(self, new_song, n_recommendations):
            recommended = self._recommend(new_song=new_song, n_recommendations=n_recommendations)
            print("... Done")
            return recommended 
        
        def _recommender(self):
            return NearestNeighbors(metric=self.metric, algorithm=self.algorithm, n_neighbors=self.k, n_jobs=-1)
        
        def _recommend(self, new_song, n_recommendations):
            # Get the id of the recommended songs
            recommendations = []
            recommendation_ids = self._get_recommendations(new_song=new_song, n_recommendations=n_recommendations)
            # return the name of the song using a mapping dictionary
            recommendations_map = self._map_indeces_to_song_title(recommendation_ids)
            # Translate this recommendations into the ranking of song titles recommended
            for i, (idx, dist) in enumerate(recommendation_ids):
                recommendations.append(recommendations_map[idx])
            return recommendations
                    
        def _get_recommendations(self, new_song, n_recommendations):
            # Get the id of the song according to the text
            recom_song_id = self._fuzzy_matching(song=new_song)
            # Start the recommendation process
            print(f"Starting the recommendation process for {new_song} ...")
            # Return the n neighbors for the song id
            distances, indices = self.model.kneighbors(self.data[recom_song_id], n_neighbors=n_recommendations+1)
            return sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
        
        def _map_indeces_to_song_title(self, recommendation_ids):
            # get reverse mapper
            return {song_id: song_title for song_title, song_id in self.decode_id_song.items()}
        
        def _fuzzy_matching(self, song):
            match_tuple = []
            # get match
            for title, idx in self.decode_id_song.items():
                ratio = fuzz.ratio(title.lower(), song.lower())
                if ratio >= 60:
                    match_tuple.append((title, idx, ratio))
            # sort
            match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
            if not match_tuple:
                print(f"The recommendation system could not find a match for {song}")
                return
            return match_tuple[0][1]
    