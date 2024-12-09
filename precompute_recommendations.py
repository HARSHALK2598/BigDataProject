import pandas as pd
import numpy as np
import torch
from torch import nn
import sqlite3

class NCF(nn.Module):
    def __init__(self, num_users, num_items):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)

    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        vector = torch.cat([user_embedded, item_embedded], dim=-1)
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))
        pred = nn.Sigmoid()(self.output(vector))
        return pred

def load_model(file_path, num_users, num_items):
    model = NCF(num_users, num_items)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model

def precompute_recommendations():
    conn = sqlite3.connect("movie_recommendation.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='recommendations';")
    if cursor.fetchone() is None:
        print("Precomputing recommendations...")
        ratings = pd.read_sql("SELECT * FROM ratings", conn)
        num_users = ratings['UserID'].max() + 1
        num_items = ratings['MovieID'].max() + 1
        model = load_model("mrs-v4.pkl", num_users, num_items)

        recommendations = []
        for user_id in range(1, num_users):
            interacted_items = ratings[ratings["UserID"] == user_id]["MovieID"].tolist()
            not_interacted_items = set(range(1, num_items)) - set(interacted_items)
            test_items = list(np.random.choice(list(not_interacted_items), 99))
            if interacted_items:
                test_items.append(interacted_items[0])

            user_tensor = torch.tensor([user_id] * 100)
            item_tensor = torch.tensor(test_items)
            predicted_labels = model(user_tensor, item_tensor).detach().numpy().squeeze()
            top10_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][:10]]

            for rank, item in enumerate(top10_items, start=1):
                recommendations.append((user_id, item, rank))

        recommendations_df = pd.DataFrame(recommendations, columns=["UserID", "MovieID", "Rank"])
        recommendations_df.to_sql("recommendations", conn, if_exists="replace", index=False)
    else:
        print("Recommendations already exist in the database. Skipping precomputation.")
    conn.close()

if __name__ == "__main__":
    precompute_recommendations()

