"""
Author: Isabella Fisch
Date: April 14, 2025
Class: DS 4420
Assignment: Final Project
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def item_collab_filter(data, target, similarity, k):
    ''' 
    Performs item-item collaborative filtering on a given dataset to recommend new items to a target user using 
    predicted rating based on similar items.
    
    Args:
        data (DataFrame): ratings data to use
        target (str): target user to predict new items for
        similarity (str): similarity metric to use to compute similarity scores (cosine similarity or L2 norm)
        k (int): number of similar users to use to make predictions
        
    Returns:
        recommendations (dict): predicted ratings of new items for the target user
    '''
    # check if the target user is in the dataset
    if target not in data.index:
        print(f'Error: Target user {target} not found in the dataset.')
        return None
    
    # check if the target user has any missing ratings
    if not data.loc[target].isna().any():
        print(f'Error: Target user {target} has rated all of the professors.')
        return None
    
    # center the ratings for each item
    centered = data.apply(lambda x: x - np.mean(x), axis=0)
    
    # get list of items with missing ratings for the target user
    missing_ratings = data.loc[target].isnull()
    missing_ratings_names = missing_ratings[missing_ratings].index.tolist()
    
    # calculate similarity scores between the target item and the other items
    all_similarity_scores = {}
    
    # iterate through each missing item rating for the target user
    for item in missing_ratings_names:
        similarity_scores = []

        # iterate through each item in the database and compare with the current item
        for column in centered.columns:
            if column != item:
                
                # only keep ratings from users who have rated both of the items
                both_ratings = centered[centered[[item, column]].notna().all(axis=1)].index
                item_keep = centered[item].loc[both_ratings]
                column_keep = centered[column].loc[both_ratings]

                # calculate the cosine similarity
                if similarity == 'cosine':
                    score = cosine_similarity(
                        np.array(item_keep).reshape(1, -1), 
                        np.array(column_keep).reshape(1, -1)
                    )
                    similarity_scores.append((column, score))

                # calculate the L2 norm
                elif similarity == 'L2':
                    score = np.linalg.norm(item_keep - column_keep, ord=2) * -1
                    similarity_scores.append((column, score))

                # print error if given a different similarity metric
                else:
                    print(f'Error: Can only compute similarity using cosine similarity or L2 norm.')
            
        all_similarity_scores[item] = similarity_scores

    # scale the similarity scores with min-max scaling
    chatgpt_sim_scores = {}
    for item, groups in all_similarity_scores.items():
        scores = [group[1] for group in groups]
        min_score = min(scores)
        max_score = max(scores)

        scaled_similarity = []
        for group in groups:
            scaled_score = (group[1] - min_score) / (max_score - min_score)
            scaled_similarity.append((group[0], scaled_score))
            
        # keep the k most similar users to the target user
        sorted_scores = sorted(scaled_similarity, key=lambda x: x[1], reverse=True)
        similar_items = sorted_scores[:k]
        all_similarity_scores[item] = similar_items

        if item == 'ChatGPT':
            chatgpt_sim_scores = sorted_scores
    
    # use the most similar items to predict missing ratings for the target user
    imputed_original = data.apply(lambda x: x.fillna(np.mean(x)), axis=0)
    
    predicted_ratings = {}
    for item, similarity in all_similarity_scores.items():
        weighted_ratings = 0
            
        for score in similarity:
            weighted_ratings += np.mean(imputed_original[score[0]].loc[target]) * score[1]
            
        prediction = weighted_ratings / sum([score[1] for score in similarity])
        predicted_ratings[item] = prediction
    
    # return dictionary 
    return [predicted_ratings, chatgpt_sim_scores]


def main():

    # read in data
    stud_apps = pd.read_csv('student_app_usage.csv')
    stud_apps.drop(stud_apps.columns[0], axis=1, inplace=True)

    # create copy of dataframe to check predictions later
    stud_apps_truth = stud_apps.copy()

    # impute NA randomly to create test set
    for col in stud_apps.columns:
        stud_apps.loc[stud_apps.sample(frac=0.05).index, col] = np.nan

    # get list of rows with null values
    missing_rates = stud_apps[stud_apps.isnull().any(axis=1)].index.tolist()

    # predict missing values
    metric = 'L2'
    user_preds = {}
    for row in missing_rates:
        prediction, sim_scores = item_collab_filter(stud_apps, row, metric, 4)
        user_preds[row] = prediction
        
        # save ChatGPT similarity scores
        if len(sim_scores) > 0:
            chatgpt_sim_scores = sim_scores

    # input predictions to dataframe
    for user, preds in user_preds.items():
        for app, pred in preds.items():
            if metric == 'cosine':
                stud_apps.loc[user, app] = pred[0]
            else:
                stud_apps.loc[user, app] = pred

    # subtract dataframes from each other to see difference between predicted and actual usage minutes
    rates_diff = stud_apps_truth - stud_apps

    # convert difference dataframe to list to calculate average 
    rates_diff_lst = rates_diff.values.flatten().tolist()

    # only keep predicted values
    rates_diff_preds = [abs(x) for x in rates_diff_lst if x != 0.0]
    
    # calculate average error
    avg_pred_error = sum(rates_diff_preds) / len(rates_diff_preds)
    print(avg_pred_error)

    # look at most similar apps to ChatGPT
    print(chatgpt_sim_scores)


main()
