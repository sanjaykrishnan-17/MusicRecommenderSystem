from recommenders import *
import warnings
warnings.filterwarnings("ignore")

user_data = pd.read_csv(r'C:\Users\PC\Documents\DMMiniProject\datasets\user_data.csv')
song_data = pd.read_csv(r'C:\Users\PC\Documents\DMMiniProject\datasets\song_data.csv')
song_data.drop_duplicates(['song_id'], inplace=True)
content_songs = pd.read_csv(r"C:\Users\PC\Documents\DMMiniProject\datasets\lyricsfreak.csv")

print("Enter your name : ", end = "")
user_name = input()
print("\nHello " + user_name, end='\n')

print("Press 1 for Content-based Recommendations", end = '\n')
print("Press 2 for Collaborative-based Recommendations", end = '\n')
print("Press 3 for Popularity-based Recommendations", end = '\n')
recommendation_type = int(input("Enter your choice : "))

if(recommendation_type == 1):
  print("Enter the song number : ", end = "")
  song_n = int(input())
  print("Enter the number of recommendations needed : ", end = "")
  n = int(input())
  content_songs = content_songs.sample(n=5000).drop('link', axis=1).reset_index(drop=True)
  content_songs['text'] = content_songs['text'].str.replace(r'\n', '')
  tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
  lyrics = tfidf.fit_transform(content_songs['text'])
  cosine = cosine_similarity(lyrics)
  similarities = {}
  for i in range(len(cosine)):
      similar_indices = cosine[i].argsort()[:-50:-1]
      similarities[content_songs['song'].iloc[i]] = [(cosine[i][x], content_songs['song'][x], content_songs['artist'][x]) for x in similar_indices][1:]
  recommedations = content_based_recommender(similarities)
  recommendation = {
    "song": content_songs['song'].iloc[song_n],
    "number_songs": n 
  }
  recommedations.recommend(recommendation)


elif(recommendation_type == 2):
  print("Enter the song number : ", end = "")
  song_name = input()
  col = collaborative_based_recommender()
  collab_songs = pd.merge(user_data, song_data, on="song_id", how="left")
  song_user = collab_songs.groupby('user_id')['song_id'].count()
  users_morethan_16 = song_user[song_user > 16].index.to_list()
  songid_morethan_16 = collab_songs[collab_songs['user_id'].isin(users_morethan_16)].reset_index(drop=True)
  songs_features = songid_morethan_16.pivot(index='song_id', columns='user_id', values='listen_count').fillna(0)
  new_songs_features = csr_matrix(songs_features.values)
  uniq = collab_songs.drop_duplicates(subset=['song_id']).reset_index(drop=True)[['song_id', 'title']]
  mapped_songs = {
    song: i for i, song in enumerate(list(uniq.set_index('song_id').loc[songs_features.index].title))
  }
  model = col.Recommender(metric='cosine', algorithm='brute', k=20, data=new_songs_features, decode_id_song=mapped_songs)
  recos = model.make_recommendation(new_song=song_name, n_recommendations=10)
  print(f'The recommendations for "{song_name}" are:')
  for i in recos:    
      print(f">> {i}", end = "\n")

elif(recommendation_type == 3):
  print("popularitybased(user_name)")


else:
  print("\nEnter a Valid Number")