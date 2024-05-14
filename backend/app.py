from flask import Flask, jsonify, g, request
import pymongo
import pandas as pd
from utils.train_model import train_model_scikit, train_model_cross
import pickle
from bson.json_util import dumps
from flask_cors import CORS
import numpy as np
import json

app = Flask(__name__)
CORS(app)

mongodb_uri = 'mongodb://localhost:27017/'
db_name = 'brfss_project'

def get_db():
    if 'db' not in g:
        # Connect to MongoDB and store the connection in the application context (g)
        client = pymongo.MongoClient(mongodb_uri)
        g.db = client[db_name]
    return g.db

@app.before_request
def before_request():
    # Establish a single MongoDB connection before each request
    g.db = get_db()

@app.route('/model/train')
def train():
    # 1. Read the parameters and df
    db = g.db
    sort_query = [("timestamp", pymongo.DESCENDING)]
    configurations = db['configurations'].find_one({}, sort=sort_query)
    df = pd.DataFrame(list(db['10k_dataset'].find()))
    df = df.drop(columns=['_id'])
    # 2. Do the machine training
    train_model_scikit(
        df, 
        train_percent=configurations['train_percent'], 
        eval_percent=configurations['eval_percent'], 
        test_percent=configurations['test_percent'],        
        n_estimators=configurations['n_estimators'],        
        min_samples_split=configurations['min_samples_split'],        
        max_features=configurations['max_features'],        
        )
    # train_model_cross(
    #     df,         
    #     n_estimators=configurations['n_estimators'],        
    #     min_samples_split=configurations['min_samples_split'],        
    #     max_features=configurations['max_features'],        
    #     )
    # 3. Send the final response when (2) is finished
    return jsonify({"response": 90})

@app.route('/model/predict', methods=['POST'])
def predict():
    # 1. Read the arguments
    data = request.get_json(force=True)
    feature_values = [data[feature] for feature in ['PHYSHLTH', 'EXERANY2', 'BPHIGH6', 'CHOLMED3', 'DIABETE4', 'HAVARTH5', 
                                                    'LMTJOIN3', 'EDUCA', 'EMPLOY1', 'DIFFWALK', 'ALCDAY5', '_RFHLTH', 
                                                    '_PHYS14D', '_MENT14D', '_TOTINDA', '_RFHYPE6', '_MICHD', '_DRDXAR3', 
                                                    '_LMTACT3', '_LMTWRK3', '_AGEG5YR', '_AGE80', '_AGE_G', 'WTKG3', 
                                                    '_BMI5', '_BMI5CAT', '_EDUCAG']]
    
    # 2. Load the model from current folder (called model.pkl) and use it for predicting
    model = pickle.load(open('model.pkl', 'rb'))
    prediction = model.predict(np.array(feature_values).reshape(1, -1))
    
    # 3. Return the result to the UI
    return jsonify({'prediction': int(prediction[0])})

@app.route('/model/hyperparameters', methods=['POST'])
def hyperparameters():
    db = g.db
    # 1. Read the parameters from request
    parameters = request.get_json()

    # 2. Update the current hyperparameters in the configurations collection
    config = db['configurations']
    config.update_one({}, {'$set': parameters})

    # 3. Fetch the updated configuration
    updated_config = config.find_one()

    # 4. Return the updated configurations
    return jsonify(json.loads(json.dumps(updated_config, default=str))), 200

@app.route('/dataset/items', methods=['GET'])
def get_dataset():
    db = g.db
    # Read the 'page' and 'pagesize' query parameters
    page = int(request.args.get('page', 1))
    pagesize = int(request.args.get('pagesize', 10))

    # Calculate the number of documents to skip
    skips = pagesize * (page - 1)

    # Fetch the data from the collection with pagination
    cursor = db['dataset'].find().skip(skips).limit(pagesize)

    # Create a list from the cursor
    list_cur = list(cursor)

    # Convert the list to JSON and return the response
    return dumps(list_cur), 200



@app.route('/model/info')
def model_info():
    db = g.db
    # Load the model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Get the model information
    model_info = {}
    model_info['model_name'] = type(model).__name__
    model_info['hyperparameters'] = model.get_params()

    # Depending on the model, some trained parameters might not exist, we should use try except to handle these cases.
    try:
        model_info['coef'] = model.coef_.tolist() if hasattr(model, 'coef_') else "Not available"
    except AttributeError:
        model_info['coef'] = "Not available"

    try:
        model_info['intercept'] = model.intercept_.tolist() if hasattr(model, 'intercept_') else "Not available"
    except AttributeError:
        model_info['intercept'] = "Not available"
    
    try:
        model_info['feature_importances'] = model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else "Not available"
    except AttributeError:
        model_info['feature_importances'] = "Not available"

    # Return the model information as a JSON response
    total_samples = db['dataset'].count_documents({})

    # assuming 'configurations' collection has a document with 'train_percent', 'eval_percent', 'test_percent' fields
    conf = db['configurations'].find_one({}, {'_id': 0, 'train_percent': 1, 'eval_percent': 1, 'test_percent': 1})

    if conf is None or 'train_percent' not in conf or 'eval_percent' not in conf or 'test_percent' not in conf:
        return jsonify({"error": "train_percent, eval_percent, test_percent not found in configurations"}), 404

    # calculate the count for each type of samples
    train_samples = int(total_samples * conf['train_percent'])
    evaluation_samples = int(total_samples * conf['eval_percent'])
    test_samples = int(total_samples * conf['test_percent'])

    model_info['train_samples'] = train_samples
    model_info['evaluation_samples'] = evaluation_samples
    model_info['test_samples'] = test_samples

    return jsonify(model_info)


if __name__ == "__main__":
    app.run(debug=True)
