
import pandas as pd
import numpy as np

#receive files as args
import sys
ratings_file = sys.argv[1]
targets_file = sys.argv[2]

#read ratings.csv   
df = pd.read_csv(ratings_file)

#separate UserId:ItemId into two columns
df[['UserId','ItemId']] = df['UserId:ItemId'].str.split(':',expand=True)

#drop the original column 
df.drop(columns =["UserId:ItemId"], inplace = True)

#unique users and items,
unique_users = df['UserId'].unique()
unique_items = df['ItemId'].unique()

#map users and items to indices
user_to_index = {user: i for i, user in enumerate(unique_users)}
item_to_index = {item: i for i, item in enumerate(unique_items)}

#vectorize unique users and items
user_indices = df['UserId'].map(user_to_index).values
item_indices = df['ItemId'].map(item_to_index).values   

#save ratings in a vector
ratings = (df['Rating'].values)


#hyperparameters
learning_rate = 0.0095 #Learning rate
num_epochs = 19        #Number of epochs
num_factors = 15       #Number of latent factors
lambda_l2 = 0.095      #Regularization parameter for L2
lambda_l1 = 0.01       #Regularization parameter for L1


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
batch_size = 400 

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
        user_bias[user_batch] += learning_rate * (error - lambda_l2 * user_bias[user_batch] - lambda_l1 * np.sign(user_bias[user_batch]))
        item_bias[item_batch] += learning_rate * (error - lambda_l2 * item_bias[item_batch] - lambda_l1 * np.sign(item_bias[item_batch]))

        #calculate the gradients for the user and item matrices, by multiplying the learning rate by the error, adding the regularization terms and the matrices
        user_matrix[user_batch, :] += learning_rate * (error[:, np.newaxis] * item_matrix[item_batch, :] - lambda_l2 * user_matrix[user_batch, :] - lambda_l1 * np.sign(user_matrix[user_batch, :]))
        item_matrix[item_batch, :] += learning_rate * (error[:, np.newaxis] * user_matrix[user_batch, :] - lambda_l2 * item_matrix[item_batch, :] - lambda_l1 * np.sign(item_matrix[item_batch, :]))


    mse = np.mean(error ** 2)

    #used for testing and improving the results
    #print(f'Epoch {epoch + 1}/{num_epochs} | MSE: {mse:.5f}')




#read targets.csv
df_targets = pd.read_csv(targets_file)

#separate UserId:ItemId into two columns
df_targets[['User_id','Item_id']] = df_targets['UserId:ItemId'].str.split(':',expand=True)

#drop the original column
df_targets.drop(columns =["UserId:ItemId"], inplace = True)

#function to round the predicted rating to the nearest integer if the decimal part is greater than or equal to 0.95
def custom_round(value):
    decimal_part = value - int(value)
    if decimal_part >= 0.95:
        return int(value) + 1
    else:
        return value


print('UserId:ItemId,Rating')


#for each pair (user,item) in the targets.csv file, predict the rating and print it
for _, row in df_targets.iterrows():

    #get the user and item indices
    user_index = user_to_index[row['User_id']]
    item_index = item_to_index[row['Item_id']]
    
    #predict the rating by taking the dot product of the user and item vectors and adding the user and item biases and the ratings mean
    predicted_rating = np.dot(user_matrix[user_index, :], item_matrix[item_index, :].T) + user_bias[user_index] + item_bias[item_index] + ratings_mean
    
    #if the predicted rating is greater than 5, set it to 5
    if predicted_rating > 5:
        predicted_rating = 5

    #if the predicted rating is less than 1, set it to 1
    if predicted_rating < 1:
        predicted_rating = 1

    print(f'{row["User_id"]}:{row["Item_id"]},{custom_round(predicted_rating)}')
    

    

    
