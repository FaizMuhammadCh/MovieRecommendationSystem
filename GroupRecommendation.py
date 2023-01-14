# User-based Collaborative Filtering Group Recommendations
# Faiz Muhammad

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


def pred(u_index, ui_matrix, sim_matrix, topk_recommendations=20):
    item_pred = []
    x = np.array(ui_matrix.iloc[u_index, :])  # converting our main user's ratings into a numpy array
    x_mean = np.nanmean(x)
    sim_sum = np.sum([pair[1] for pair in sim_matrix])  # summing the similarity of top 30 similar users
    count = 0
    counter = 0
    user_rating = []  # this list will contain all the user's ratings (predicted or original) in the same order
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
            else:
                pred = 1
                item_pred.append((item, pred))  # else give the lowest possible rating to the item
            user_rating.append(pred)  # append that item's rating in the user_rating list
        else:  # if the item was already been rated by the user
            user_rating.append(ui_matrix.iloc[u_index, item])  # add the rating in the user_rating list
    item_pred.sort(key=lambda u: u[1])  # sort the items based on their score
    return list(reversed(item_pred[-topk_recommendations:])), user_rating  #

'''
There are some movies that are non-rated by any user. This function gets rid of those movies
'''


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

'''
This function displays the item's information by using the index value of the item in list.
'''


def display_topk(movies, agg_list):
    for index, value in agg_list:  # agg_list refers to the group's top 20 recommended movie list
        movie_title = movies.get('title')[index]  # get the Movie Title corresponding the index
        print(movie_title)


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

''' 
Let's make a group of three users: 1, 473 and 511
These values are not chosen randomly, These users are very similar to each other. Say a group of friends who have the 
same taste in movies. 
'''
group_predictions = []
group_ratings = []  # rating of all the original + predicted ratings of all the users in the group
group = [1, 473, 511]
for member in group:
    member_index = member - 1  # Since the first user starts from index=0 in the dataframe
    sim = get_similarUsers(u_index=member_index, ui_matrix=user_item_matrix, topk_users=30)
    predictions, u_rating = pred(u_index=member_index, ui_matrix=user_item_matrix, sim_matrix=sim, topk_recommendations=20)
    group_predictions.append(predictions)  # Appending the 20 recommended movies for a user in the group list
    group_ratings.append(u_rating)


print("Top 20 Predictions for User 1")
display_topk(movies_df, group_predictions[0])
print("\nTop 20 Predictions for User 2")
display_topk(movies_df, group_predictions[1])
print("\nTop 20 Predictions for User 3")
display_topk(movies_df, group_predictions[2])

'''
Average Aggregation and Least Misery Aggregation:
for each item in all three lists of the group, we are going to calculate the group rating using the following formula

for Average Aggregation: r*(g,i) = AVGu∈g {r*(u,i)}
for Least Misery Aggregation: r*(g,i) = MINu∈g {r*(u,i)}

where r*(u,i) is either the predicted rating of u for i, or r(u,i) if this rating is present in the data set

'''
r_mean_gi = []  # list of average aggregated group item indices and their respective scores
r_min_gi = []  # list of least misery aggregated group item indices and their respective scores
index_visted = []  # list of visited indices/items
for i in range(len(group_predictions)):
    for index, value in group_predictions[i]:
        if index not in index_visted:  # check to see if the score has already been aggregated for this item
            index_visted.append(index)  # if the item repeats, don't calculate the average again
            r_u1i = group_ratings[0][index]  # rating for item by user 1
            r_u2i = group_ratings[1][index]  # rating for item by user 2
            r_u3i = group_ratings[2][index]  # rating for item by user 3
            r_mean = np.mean([r_u1i, r_u2i, r_u3i])  # mean of all three scores for a particula item
            r_min = min(r_u1i, r_u2i, r_u3i)  # min of all three scores for a particula item
            r_mean_gi.append((index, r_mean))  # add the tuple of item index and its average score
            r_min_gi.append((index, r_min))  # add the tuple of item index and its minimum score
r_mean_gi.sort(key=lambda w: w[1])  # sort the items based on their score
r_min_gi.sort(key=lambda w: w[1])  # sort the items based on their score
top20_average_agg_list = list(reversed(r_mean_gi[-20:]))  # reverse the list to get the highest score on top
top20_leastmisery_agg_list = list(reversed(r_min_gi[-20:]))  # reverse the list to get the highest score on top

print("\nTop 20 Movies for the group using Average Aggregation Method:")
display_topk(movies_df, top20_average_agg_list)

print("\nTop 20 Movies for the group using Least Misery Aggregation Method:")
display_topk(movies_df, top20_leastmisery_agg_list)
