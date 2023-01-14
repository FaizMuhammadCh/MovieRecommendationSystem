import numpy as np
import pandas as pd
from scipy import stats

'''
This function gets top k similar users with their corresponding similarities using the Pearson's Coefficient
'''


def get_similarUsers(u_index, ui_matrix, topk_users=30):  # get top k similar users
    sim = []
    total_users = ui_matrix.shape[0]
    x = np.array(ui_matrix.iloc[u_index, :])  # converting our main user's ratings into a numpy array
    for i in range(total_users):
        if i != u_index:
            y = np.array(ui_matrix.iloc[i, :])  # converting similar user's ratings into a numpy array
            xy_stacked = np.vstack([x, y])  # Stacking two users
            xy = xy_stacked[:,
                 ~np.any(np.isnan(xy_stacked), axis=0)]  # only getting the vector where both arrays have values

            if len(xy[0]) > 3 and len(xy[1]) > 3:
                r, p = stats.mstats.pearsonr(xy[0], xy[1])
                if np.isnan(r):  # Checking if pearson coefficient is Nan
                    r = -2
                sim.append((i, r))  # appending (index, value) tuple in the similarity list
    sim.sort(key=lambda o: o[1])  # sorting the tuples based on similarity
    return list(reversed(sim[-topk_users:]))  # getting the top 30 most similar users to our user


'''
This function returns the movie indices along with their score for a particular user identified by u_index
'''


def pred(ui_matrix, u_index, sim_matrix, topk_recommendations=20):
    item_pred = []
    x = np.array(ui_matrix.iloc[u_index, :])  # converting our main user's ratings into a numpy array
    x_mean = np.nanmean(x)
    sim_sum = np.sum([pair[1] for pair in sim_matrix])  # summing the similarity of top 30 similar users
    count = 0
    counter = 0
    for item in range(len(x)):
        if np.isnan(ui_matrix.iloc[u_index, item]):  # if the item hasn't been rated by our main user
            simsum_centering = []
            counter += 1
            for index, similarity in sim_matrix:  # for users in 30 most similar users
                if not np.isnan(ui_matrix.iloc[index, item]):  # if this user has rated the item
                    count += 1
                    y = np.array(ui_matrix.iloc[index, :])  # convert the similar user to a numpy array
                    y_mean = np.nanmean(y)  # get the mean rating of this user
                    rating = ui_matrix.iloc[index, item]  # get the rating of the item
                    rating_centered = rating - y_mean  # sub mean_rating of the similar user from this item's rating
                    simsum_centering.append(similarity * rating_centered)  # mul w this user's similarity & add to list
            if (np.sum(simsum_centering) / sim_sum) > 0:
                pred = x_mean + (np.sum(simsum_centering) / sim_sum)
                item_pred.append((item, pred))  # add the predicted score for the item along with its index
    item_pred.sort(key=lambda u: u[1])  # sort the items based on their score
    return list(reversed(item_pred[-topk_recommendations:]))  # return the top k items


# There are some movies that are non-rated by any user
# This function gets rid of those movies
def prune_movies_df(movies, ratings):
    movie_ids = np.array(movies['movieId'])
    # print(len(movie_ids))  # Uncomment for debugging purposes
    rated_movie_ids = set(list(np.array(ratings['movieId'])))
    # print(len(rated_movie_ids))  # Uncomment for debugging purposes
    uncommon_movies = []
    # Getting movies that are in movies.csv but not in ratings.csv
    for i in movie_ids:
        if i not in rated_movie_ids:
            uncommon_movies.append(i)
    # print(len(uncommon_movies))
    ind = []
    for movie_id in uncommon_movies:
        movies.drop(movies.loc[movies['movieId'] == movie_id].index,
                    inplace=True)  # removing those movies from the dataframe
    return movies_df


# main
# make changes according to your path
root = 'ml-latest-small'

# read movies.csv and ratings.csv from the directory
movies_df = pd.read_csv(root + '/movies.csv', sep=',')
ratings_df = pd.read_csv(root + '/ratings.csv', sep=',')
movies_df = prune_movies_df(movies_df, ratings_df)  # pruning the movies from movies_df that weren't rated by anyone

# ratings_df is reshaped according to the example in the slides for simplicity
ratings_df_reshaped = ratings_df.pivot(index='userId', columns='movieId', values='rating')
user_item_matrix = ratings_df_reshaped  # We are going to use the matrix that has NaN values
# uncomment for debugging purposes
print(f'movies_df: \n')
print(movies_df)
print(f'ratings_df: \n')
print(ratings_df)
print(f'ratings_df_reshaped: \n')
print(ratings_df_reshaped)

userId = 1  # subject to change | userId starts from 1
sim = get_similarUsers(u_index=userId - 1, ui_matrix=user_item_matrix,
                       topk_users=30)  # topk_users argument takes 30 top similar users default
predictions = pred(ui_matrix=user_item_matrix, u_index=userId - 1, sim_matrix=sim, topk_recommendations=20)

# Top 10 Most Similar Users
print("Top 10 Most Similar Users for user: ", userId)
for index, score in sim[:10]:
    print(index + 1, ", Score: ", score)  # + 1 because users start from 1

# Top 20 Movie Recommendations for User
print("Top 20 Recommended Movies for user: ", userId)
print('\n')
for index, value in predictions:
    movie_id = movies_df.get('movieId')[index]  # get the Movie ID corresponding the index
    movie_title = movies_df.get('title')[index]  # get the Movie Title corresponding the index
    movie_genre = movies_df.get('genres')[index]  # get the Movie Genres corresponding the index
    print(movie_id, ", Movie: ", movie_title, ", Genre: ", movie_genre, ", Score: ", value)

