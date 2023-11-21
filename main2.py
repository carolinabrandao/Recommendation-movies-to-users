import pandas as pd
import numpy as np

#receive files as args
import sys


class DataLoader:
    def __init__(self, ratings_file, content_file, targets_file):
        self.ratings_file = ratings_file
        self.content_file = content_file
        self.targets_file = targets_file

    def load_data(self):
        # Reading files
        self.df_ratings = pd.read_json(self.ratings_file, lines=True)
        self.df_content = pd.read_json(self.content_file, lines=True)
        self.df_targets = pd.read_csv(self.targets_file)

        # Preprocessing
        self.df_ratings.drop(columns=["Timestamp"], inplace=True)
        self.df_content['imdbRating'] = self.df_content['imdbRating'].replace('N/A', np.nan).astype(float)
        self.df_content['imdbRating'].fillna(self.df_content['imdbRating'].mean(), inplace=True)

        self.df_content['imdbVotes'] = self.df_content['imdbVotes'].replace('N/A', np.nan)
        self.df_content['imdbVotes'] = self.df_content['imdbVotes'].str.replace(',', '').astype(float)
        self.df_content['imdbVotes'].fillna(self.df_content['imdbVotes'].mean(), inplace=True)

        return self.df_ratings, self.df_content, self.df_targets


class RecommenderModel:
    def __init__(self, num_users, num_items, num_factors=30, learning_rate=0.007, regularization=0.2, num_epochs=20):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.num_epochs = num_epochs
        self.ratings_mean = None

        # Initialize user and item matrices
        np.random.seed(12)  # Seed for reproducibility
        self.user_matrix = np.random.uniform(-np.sqrt(6 / (num_users + num_factors)), np.sqrt(6 / (num_users + num_factors)), size=(num_users, num_factors))
        self.item_matrix = np.random.uniform(-np.sqrt(6 / (num_items + num_factors)), np.sqrt(6 / (num_items + num_factors)), size=(num_items, num_factors))

        # Initialize biases
        self.user_bias = np.zeros(num_users)
        self.item_bias = np.zeros(num_items)

    def train_model(self, user_indices, item_indices, ratings):
        self.ratings_mean = np.mean(ratings)
        batch_size = 128
        batch_num = int(len(ratings) / batch_size)

        for epoch in range(self.num_epochs):
            shuffled_indices = np.random.permutation(len(ratings))

            for batch in range(batch_num):
                batch_indices = shuffled_indices[batch * batch_size: (batch + 1) * batch_size]
                user_batch = user_indices[batch_indices]
                item_batch = item_indices[batch_indices]
                rating_batch = ratings[batch_indices]

                product = np.sum(self.user_matrix[user_batch, :] * self.item_matrix[item_batch, :], axis=1)
                rating_hat = product + self.user_bias[user_batch] + self.item_bias[item_batch] + self.ratings_mean

                error = rating_batch - rating_hat

                # Update biases
                self.user_bias[user_batch] += self.learning_rate * (error - self.regularization * self.user_bias[user_batch])
                self.item_bias[item_batch] += self.learning_rate * (error - self.regularization * self.item_bias[item_batch])

                # Update matrices
                for i in range(self.num_factors):
                    self.user_matrix[user_batch, i] += self.learning_rate * (error * self.item_matrix[item_batch, i] - self.regularization * self.user_matrix[user_batch, i])
                    self.item_matrix[item_batch, i] += self.learning_rate * (error * self.user_matrix[user_batch, i] - self.regularization * self.item_matrix[item_batch, i])


class PredictionProcessor:
    def __init__(self, df_content, df_targets, recommender_model):
        self.df_content = df_content
        self.df_targets = df_targets
        self.recommender_model = recommender_model

        # Convert imdbRating and imdbVotes to dictionaries for easy lookup
        self.imdb_rating_dict = pd.Series(self.df_content.imdbRating.values, index=self.df_content.ItemId).to_dict()
        self.imdb_votes_dict = pd.Series(self.df_content.imdbVotes.values, index=self.df_content.ItemId).to_dict()

    def normalize_rating(self, value):
        if '/' in value:
            numerator, denominator = value.split('/')
            return float(numerator) / float(denominator) * 10
        elif '%' in value:
            return float(value.strip('%')) / 10
        else:
            return None

    def calculate_average_ratings(self, ratings):
        normalized_ratings = [self.normalize_rating(rating['Value']) for rating in ratings if self.normalize_rating(rating['Value']) is not None]
        return sum(normalized_ratings) / len(normalized_ratings) if normalized_ratings else None

    def process_predictions(self):
        # Normalize and calculate average ratings
        ratingsEnsemble = self.df_content[['ItemId', 'Ratings']].copy()
        ratingsEnsemble['AverageRating'] = ratingsEnsemble['Ratings'].apply(self.calculate_average_ratings)
        ratingsEnsemble['AverageRating'].fillna(ratingsEnsemble['AverageRating'].mean(), inplace=True)
        ratingsEnsemble.drop(columns=["Ratings"], inplace=True)

        self.df_targets = pd.merge(self.df_targets, ratingsEnsemble, on='ItemId')
        self.df_targets = pd.merge(self.df_targets, pd.DataFrame(list(self.imdb_rating_dict.items()), columns=['ItemId', 'imdbRating']), on='ItemId')
        self.df_targets = pd.merge(self.df_targets, pd.DataFrame(list(self.imdb_votes_dict.items()), columns=['ItemId', 'imdbVotes']), on='ItemId')

        # Predict ratings
        self.df_targets['Rating'] = 0

        for _, row in self.df_targets.iterrows():
            user_index = self.recommender_model.user_to_index.get(row['UserId'], None)
            item_index = self.recommender_model.item_to_index.get(row['ItemId'], None)

            if user_index is not None and item_index is not None:
                svd_rating = np.dot(self.recommender_model.user_matrix[user_index, :], self.recommender_model.item_matrix[item_index, :].T) + self.recommender_model.user_bias[user_index] + self.recommender_model.item_bias[item_index] + self.recommender_model.ratings_mean
                predicted_rating = svd_rating * row['imdbRating'] * row['imdbVotes'] * row['AverageRating']
            else:
                predicted_rating = self.recommender_model.ratings_mean * row['imdbRating'] * row['imdbVotes'] * row['AverageRating']

            self.df_targets.at[_, 'Rating'] = predicted_rating

        # Sort predictions
        self.df_targets.sort_values(by=['UserId', 'Rating'], ascending=[True, False], inplace=True)

        # Prepare for output
        self.df_targets.drop(columns=["Rating", 'imdbRating', 'imdbVotes', 'AverageRating'], inplace=True)


class OutputGenerator:
    def __init__(self, df_targets):
        self.df_targets = df_targets

    def generate_output(self):
        # Format and print the output as a CSV format
        print('UserId,ItemId')
        for _, row in self.df_targets.iterrows():
            print(f'{row["UserId"]},{row["ItemId"]}')


def main(ratings_file, content_file, targets_file):
    # Initialize and load data
    data_loader = DataLoader(ratings_file, content_file, targets_file)
    df_ratings, df_content, df_targets = data_loader.load_data()

    # Preprocess data and get necessary details for model training
    unique_users = df_ratings['UserId'].unique()
    unique_items = df_ratings['ItemId'].unique()
    user_to_index = {user: i for i, user in enumerate(unique_users)}
    item_to_index = {item: i for i, item in enumerate(unique_items)}
    user_indices = df_ratings['UserId'].map(user_to_index).values
    item_indices = df_ratings['ItemId'].map(item_to_index).values
    ratings = df_ratings['Rating'].values

    # Initialize and train the recommender model
    recommender_model = RecommenderModel(len(unique_users), len(unique_items))
    recommender_model.train_model(user_indices, item_indices, ratings)

    # Attach user and item indices mappings to the model for prediction use
    recommender_model.user_to_index = user_to_index
    recommender_model.item_to_index = item_to_index

    # Process predictions
    prediction_processor = PredictionProcessor(df_content, df_targets, recommender_model)
    prediction_processor.process_predictions()

    # Generate output
    output_generator = OutputGenerator(prediction_processor.df_targets)
    output_generator.generate_output()

if __name__ == "__main__":
    # Get the file paths from command line arguments
    ratings_file = sys.argv[1]
    content_file = sys.argv[2]
    targets_file = sys.argv[3]

    # Call the main function
    main(ratings_file, content_file, targets_file)