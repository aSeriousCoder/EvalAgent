import json
import pickle
import random
import os
import numpy as np


class DataloaderFrame:
    """
    This class is used to 
    1. load data in iteration manner, providing a unified interface for different datasets.
    2. evaluate the output and save the results.
    """
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.users = json.load(open(os.path.join(dataset_name, "processed", "sampled_users.json")))
        self.news = pickle.load(open(os.path.join(dataset_name, "processed", "news.pkl"), "rb"))
        self.current_user_idx = 0
        self.users = self.users
    
    def next_user(self):
        if self.current_user_idx >= len(self.users):
            return None, None
        else:
            user = self.users[self.current_user_idx]
            self.current_user_idx += 1
            return self.current_user_idx - 1, {
                "history_news": [self.news[news_id] for news_id in user["history_news"]],
                "tests": [
                    {
                        "test_news": [self.news[news_id] for news_id in test["news_ids"]],
                        "labels": test["labels"]
                    } for test in user["tests"]
                ]
            }
            

    def has_next(self):
        return self.current_user_idx < len(self.users)

    def progress(self):
        return f"{self.current_user_idx + 1} / {len(self.users)}"
