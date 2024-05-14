# Usage
# m = MongoDBUtils(URI, 'brfss_project', 'dataset')
# m.csv_to_mongodb('path/to/BRFSS.csv')
# Now you can check your database and it should have the BRFSS loaded in 'dataset' collection

from pymongo import MongoClient
import pandas as pd

class MongoDBUtils:
    def __init__(self, mongodb_uri, db_name, collection_name):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def csv_to_mongodb(self, csv_file):
        df = pd.read_csv(csv_file)
        data = df.to_dict(orient='records')
        self.collection.insert_many(data)

    def mongodb_to_csv(self, csv_file):
        cursor = self.collection.find({})
        data = list(cursor)
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

    def close_connection(self):
        self.client.close()
