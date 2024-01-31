import csv
import random
from collections import namedtuple


class Data:
    """
    Data class for loading data from local files.
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.items = {}
        self.users = {}
        self.advertisers = {}
        self.load_items(config["item_path"])
        self.load_users(config["user_path"])
        self.load_advertisers(config["advertiser_path"])

    def load_items(self, file_path):
        with open(file_path, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header line
            for row in reader:
                (
                    id,
                    type,
                    content,
                    product,
                    price
                ) = row
                self.items[int(id)] = {
                    "type": type,
                    "content": content,
                    "product": product,
                    "price": price
                }

    def get_user_num(self):
        """
        Return the number of users.
        """
        return len(self.users.keys())

    def load_users(self, file_path):
        with open(file_path, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header line
            for row in reader:
                (
                    user_id,
                    name,
                    gender,
                    age,
                    cash,
                    personality,
                    interest,
                    feature,
                    occupation,
                ) = row
                self.users[int(user_id)] = {
                    "name": name,
                    "gender": gender,
                    "age": int(age),
                    "cash": int(cash),
                    "personality": personality,
                    "interest": interest,
                    "feature": feature,
                    "occupation": occupation,
                }
                if self.get_user_num() == self.config["user_agent_num"]:
                    break

    def get_adv_num(self):
        """
        Return the number of advertisers.
        """
        return len(self.advertisers.keys())

    def load_advertisers(self, file_path):
        with open(file_path, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                adv_id, name, funds, type, feature = row
                self.advertisers[int(adv_id)] = {
                    "name": name,
                    "funds": int(funds),
                    "type": type,
                    "feature": feature
                }

                if self.get_adv_num() == self.config["adv_agent_num"]:
                    break