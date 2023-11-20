import pandas as pd
import numpy as np

#receive files as args
import sys
ratings_file = sys.argv[1]
content_file = sys.argv[2]
targets_file = sys.argv[3]


#read ratings.csv
df_ratings = pd.read_json(ratings_file, lines=True)
df_ratings.drop(columns =["Timestamp"], inplace = True)

#separate and vectorize unique users and items
unique_users = df_ratings['UserId'].unique()
unique_items = df_ratings['ItemId'].unique()

ratings = df_ratings['Rating'].values

user_to_index = {user: i for i, user in enumerate(unique_users)}
item_to_index = {item: i for i, item in enumerate(unique_items)}

user_indices = df_ratings['UserId'].map(user_to_index).values
item_indices = df_ratings['ItemId'].map(item_to_index).values 

#we will use the same matrix factorization used in RC1, which is Funk SVD with bias terms
#hyperparameters
learning_rate = 0.007 #Learning rate
num_epochs = 20        #Number of epochs
num_factors = 30       #Number of latent factors
regularization = 0.2   #Regularization parameter for user and item biases

#number of unique users and items
num_users = len(unique_users)
num_items = len(unique_items)

#initialize user and item matrices with a uniform distribution based on the xavier initialization
np.random.seed(12)
user_matrix = np.random.uniform(-np.sqrt(6 / (num_users + num_factors)), np.sqrt(6 / (num_users + num_factors)), size=(num_users, num_factors))
item_matrix = np.random.uniform(-np.sqrt(6 / (num_items + num_factors)), np.sqrt(6 / (num_items + num_factors)), size=(num_items, num_factors))

#dataset ratings mean
ratings_mean = np.mean(ratings)

#initialize user and item biases as an array of zeros
user_bias = np.zeros(num_users)
item_bias = np.zeros(num_items)

#batch size 
batch_size = 256

#number of batches
batch_num = int(len(ratings) / batch_size) 

#training loop, for each epoch, shuffle the ratings
for epoch in range(num_epochs):
    
    #shuffle the ratings, to avoid overfitting
    shuffled = np.random.choice(len(ratings), len(ratings), replace=False)

    #for each batch, update the user and item matrices and biases
    for batch in range(batch_num):

        #get the batch indices, by taking the batch size and multiplying it by the batch number
        batch_indices = shuffled[batch * batch_size: (batch + 1) * batch_size]
        
        #get the user and item indices and ratings for the batch by taking the batch indices
        user_batch = user_indices[batch_indices]
        item_batch = item_indices[batch_indices]
        rating_batch = ratings[batch_indices]

        #get the user and item matrices for the batch
        user_batch_matrix = user_matrix[user_batch, :]
        item_batch_matrix = item_matrix[item_batch, :]
        
        #get the dot product of the user and item matrices, to get the predicted ratings
        product = np.sum(user_batch_matrix * item_batch_matrix, axis=1)

        #get the predicted ratings by adding the user and item biases and the ratings mean to the dot product
        rating_hat = product + user_bias[user_batch] + item_bias[item_batch] + ratings_mean

        #calculate the error by subtracting the predicted ratings from the actual ratings
        error = rating_batch - rating_hat

        #calculate the gradients for the user and item biases, by multiplying the learning rate by the error, adding the regularization terms and the biases
        user_bias[user_batch] += learning_rate * (error - regularization * user_bias[user_batch])
        item_bias[item_batch] += learning_rate * (error - regularization * item_bias[item_batch])

        #update user and item latent feature matrices for each batch
        user_matrix[user_batch, :] += learning_rate * (error[:, np.newaxis] * user_matrix[user_batch, :] - regularization * user_matrix[user_batch, :])
        item_matrix[item_batch, :] += learning_rate * (error[:, np.newaxis] * item_matrix[item_batch, :] - regularization * item_matrix[item_batch, :])


#read targets and content files
df_targets = pd.read_csv(targets_file)
df_content = pd.read_json(content_file, lines=True)

df_content['imdbRating'] = df_content['imdbRating'].replace('N/A', np.nan)
df_content['imdbRating'] = df_content['imdbRating'].astype(float)
df_content['imdbRating'].fillna(df_content['imdbRating'].mean(), inplace=True)

#converting imdbRating to dictionary
imdb_rating_dict = pd.Series(df_content.imdbRating.values, index=df_content.ItemId).to_dict()

#processing imdbVotes
df_content['imdbVotes'] = df_content['imdbVotes'].replace('N/A', np.nan)
df_content['imdbVotes'] = df_content['imdbVotes'].str.replace(',', '').astype(float)
df_content['imdbVotes'].fillna(df_content['imdbVotes'].mean(), inplace=True)

#converting imdbVotes to dictionary
imdb_votes_dict = pd.Series(df_content.imdbVotes.values, index=df_content.ItemId).to_dict()

df_targets = pd.merge(df_targets, pd.DataFrame(list(imdb_rating_dict.items()), columns=['ItemId', 'imdbRating']), on='ItemId')
df_targets = pd.merge(df_targets, pd.DataFrame(list(imdb_votes_dict.items()), columns=['ItemId', 'imdbVotes']), on='ItemId')


#normalizing the ratings to a 10 point scale
# Função para normalizar os ratings para uma escala de 0 a 10
def normalize_rating(value):
    if '/' in value:
        numerator, denominator = value.split('/')
        return float(numerator) / float(denominator) * 10
    elif '%' in value:
        return float(value.strip('%')) / 10
    else:
        return None

#function to calculate the average rating of a row
def calculate_average_ratings(ratings):
    normalized_ratings = []
    for rating in ratings:
        normalized_value = normalize_rating(rating['Value'])
        if normalized_value is not None:
            normalized_ratings.append(normalized_value)

    if normalized_ratings:
        return sum(normalized_ratings) / len(normalized_ratings)
    else:
        return None

#apply the function to the ratings column
ratingsEnsemble = df_content[['ItemId', 'Ratings']].copy()
ratingsEnsemble['AverageRating'] = ratingsEnsemble['Ratings'].apply(calculate_average_ratings)

# replace null values with the mean of the non-null values
ratingsEnsemble['AverageRating'].fillna(ratingsEnsemble['AverageRating'].mean(), inplace=True)
ratingsEnsemble.drop(columns =["Ratings"], inplace = True)
df_targets = pd.merge(df_targets, ratingsEnsemble, on='ItemId')

#calculating predicted ratings
#create new column fro the rating predictions
df_targets['Rating'] = 0


#for each pair (user,item) in the targets.csv file, predict the rating and print it
for _, row in df_targets.iterrows():

    #get the user and item indices
    user_index = user_to_index[row['UserId']]

    if  row['ItemId'] in item_to_index:
        item_index = item_to_index[row['ItemId']]
    
        svd_rating = (np.dot(user_matrix[user_index, :], item_matrix[item_index, :].T) + user_bias[user_index] + item_bias[item_index] + ratings_mean) 

        predicted_rating = svd_rating * row['imdbRating'] * row['imdbVotes'] * row['AverageRating'] 
              
    
    else:
        svd_rating = ratings_mean
        predicted_rating = svd_rating * row['imdbRating'] * row['imdbVotes'] * row['AverageRating'] 
   
    df_targets.at[_, 'Rating'] = predicted_rating

#by UserId, sort the predicted ratings in descending order
df_targets.sort_values(by=['UserId', 'Rating'], ascending=[True, False], inplace=True)

#format and print the output
df_targets.drop(columns =["Rating", 'imdbRating', 'imdbVotes', 'AverageRating'], inplace = True)

#print the output that would be this csv: df_targets.to_csv('submission.csv', index=False)
print('UserId,ItemId')
for _, row in df_targets.iterrows():
    print(f'{row["UserId"]},{row["ItemId"]}')