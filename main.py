import os
import urllib.request
from app import app
from flask import Flask, request, redirect, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
cors = CORS(app)
import pandas as pd
import json
import squarify
import csv
import uuid
from flask import Flask, jsonify, Response
import requests
import io
from io import StringIO
import re
import  string
from typing import List, Tuple
import plotly.graph_objects as go
from textblob import TextBlob
from LDA import get_lda
import pandas as pd
import squarify
import matplotlib.pyplot as plt
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from datetime import datetime
import  nltk, spacy, string
from textblob import TextBlob
from numpy import outer
import uuid
import en_core_web_sm
nlp = en_core_web_sm.load()
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import plotly.express as px
# from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import tempfile



ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/file-upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['file']

    if file.filename == '':
        resp = jsonify({'message': 'No file selected for uploading'})
        resp.status_code = 400
        return resp

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        foldername = os.path.splitext(filename)[0]
        folderpath = os.path.join(app.config['UPLOAD_FOLDER'], foldername)

        i = 1
        while os.path.exists(folderpath):
            foldername = os.path.splitext(filename)[0] + f"({i})"
            folderpath = os.path.join(app.config['UPLOAD_FOLDER'], foldername)
            i += 1

        os.makedirs(folderpath)
        file.save(os.path.join(folderpath, filename))

        new_filename = str(uuid.uuid4()) + os.path.splitext(filename)[-1]

        return jsonify({"new_filename": new_filename})
    else:
        resp = jsonify({'message': 'Allowed file types are csv'})
        resp.status_code = 400
        return resp


@app.route('/projectstudio_table/<path:file_path>', methods=['GET'])
def get_table(file_path):
            # data = request.get_json()
            # file_path = data['file_path']
            df = pd.read_csv(file_path)
            ab= df.head(15)
            json_index = ab.to_json(orient ='records')
            parsed = json.loads(json_index)
            json.dumps(parsed, indent=4)
            return parsed

# @app.route("/describe", methods=['GET'])
# def describe():
#     dir_path = ("C:\\Users\\207065\\Desktop\\workspace\\backend\\Files")

#     # Get a list of all the subdirectories in the directory
#     subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

#     # Get the most recently added subdirectory
#     latest_subdir = max(subdirs, key=os.path.getctime)

#     # Get a list of all the files in the most recently added subdirectory
#     files = [os.path.join(latest_subdir, f) for f in os.listdir(latest_subdir) if os.path.isfile(os.path.join(latest_subdir, f))]

#     # Get the path of the most recently added file
#     latest_file = max(files, key=os.path.getctime)
#     df = pd.read_csv(latest_file)
#     ab= df.describe(include='all')
#     json_index = ab.to_json(orient ='index')
#     parsed = json.loads(json_index)
#     json.dumps(parsed, indent=4)  
#     return parsed 

@app.route("/describe", methods=['GET'])
def describe():
    dir_path = ("C:\\Users\\207065\\Desktop\\workspace\\backend\\Files")

    # Get a list of all the subdirectories in the directory
    subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

    # Get the most recently added subdirectory
    latest_subdir = max(subdirs, key=os.path.getctime)

    # Get a list of all the files in the most recently added subdirectory
    files = [os.path.join(latest_subdir, f) for f in os.listdir(latest_subdir) if os.path.isfile(os.path.join(latest_subdir, f))]

    # Get the path of the most recently added file
    latest_file = max(files, key=os.path.getctime)
    df = pd.read_csv(latest_file)
    ab= df.describe(include='all')
    json_index = ab.to_json(orient ='index')
    parsed = json.loads(json_index)

    # Add statistics names as a new column
    for col, values in parsed.items():
        values['Statistics'] = col

    json.dumps(parsed, indent=4)  
    return parsed



@app.route("/describ/<path:file_path>", methods=['GET'])
def describ(file_path):
   
    df = pd.read_csv(file_path)
    ab= df.describe(include='all')
    json_index = ab.to_json(orient ='index')
    parsed = json.loads(json_index)
    json.dumps(parsed, indent=4)  
    return parsed 

@app.route('/pivot_select/<path:file_path>', methods=['GET'])
def get_pivot(file_path):
            # data = request.get_json()
            # file_path = data['file_path']
            df = pd.read_csv(file_path)
            data = []
            with open(file_path, newline='',encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append(row)
            return jsonify(data=data)

@app.route('/gt_tree')
def get_tree():
    dir_path = ("C:\\Users\\207065\\Desktop\\workspace\\backend\\Files")

    # Get a list of all the subdirectories in the directory
    subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

    # Get the most recently added subdirectory
    latest_subdir = max(subdirs, key=os.path.getctime)

    # Get a list of all the files in the most recently added subdirectory
    files = [os.path.join(latest_subdir, f) for f in os.listdir(latest_subdir) if os.path.isfile(os.path.join(latest_subdir, f))]

    # Get the path of the most recently added file
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, newline='',  encoding='ISO-8859-1') as csvfile:
        reader = csv.DictReader(csvfile)
        data = [row for row in reader]
    return jsonify(data)
   


@app.route('/getdata/<path:file_path>', methods=['GET'])
def get_data(file_path):
			df = pd.read_csv(file_path)
			ab= df.head(15)
			json_index = ab.to_json(orient ='records')
			parsed = json.loads(json_index)
			json.dumps(parsed, indent=4)
			return parsed
 
    
@app.route('/getdtypes', methods=['GET'])
def get_type():
    dir_path = ("C:\\Users\\207065\\Desktop\\workspace\\backend\\Files")

    # Get a list of all the subdirectories in the directory
    subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

    # Get the most recently added subdirectory
    latest_subdir = max(subdirs, key=os.path.getctime)

    # Get a list of all the files in the most recently added subdirectory
    files = [os.path.join(latest_subdir, f) for f in os.listdir(latest_subdir) if os.path.isfile(os.path.join(latest_subdir, f))]

    # Get the path of the most recently added file
    latest_file = max(files, key=os.path.getctime)
    df = pd.read_csv(latest_file, low_memory=False)
    data_types=[] 
    for column in df.columns:
        data_types.append((column, str(df[column].dtype)))  
    return jsonify(data_types)



@app.route("/api/my-endpoint", methods=["POST"])
def my_endpoint():
    data = request.json
    my_dict = {
        "SOURCE": data["select"],
        "DESTINATION": data["de"]
    }
    # print(my_dict)
    # return jsonify(my_dict)

    dir_path = "C:\\Users\\207065\\Desktop\\workspace\\backend\\Files"
    subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    latest_subdir = max(subdirs, key=os.path.getctime)
    files = [os.path.join(latest_subdir, f) for f in os.listdir(latest_subdir) if os.path.isfile(os.path.join(latest_subdir, f))]
    latest_file = max(files, key=os.path.getctime)

    df = pd.read_csv(latest_file)

    Dict = dict(zip(data["select"], data["de"]))
    itemkeys=[]
    items=[]
    for key in Dict:
        itemkeys.append(key)
        items.append(Dict[key])
    
    dfTemp = df[itemkeys] 
    # print([dfTemp])   
    dfTemp.rename(columns =Dict, inplace=True)
    # print(dfTemp)
    json_index = dfTemp.to_json(orient ='index')
    # dfTemp = dfTemp.drop('Unnamed: 0', axis=1)
    parsed = json.loads(json_index)
    json.dumps(parsed, indent=4) 
    # resp = print("Creating csv report")
    unique_id = str(uuid.uuid4())
    current_datetime=datetime.now().strftime("%m-%d-%Y %H-%M")
    str_current_time= str(current_datetime)
    file_name = (latest_subdir + "\\Mapped_" + str_current_time + ".csv")
    dfTemp.to_csv(file_name, index=False)
    return parsed 

@app.route('/getmappeddata', methods=["GET"])
def get_mappeddata():
        dir_path = "C:\\Users\\207065\\Desktop\\workspace\\backend\\Files"
        subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        latest_subdir = max(subdirs, key=os.path.getctime)
        files = [os.path.join(latest_subdir, f) for f in os.listdir(latest_subdir) if os.path.isfile(os.path.join(latest_subdir, f))]
        latest_file = max(files, key=os.path.getctime)
        
        # get the most recent CSV file in the folder
        csv_file_path = max([os.path.join(latest_subdir, f) for f in os.listdir(latest_subdir) if f.endswith('.csv')], key=os.path.getctime)
        df = pd.read_csv(csv_file_path)
        a = df.head(10)
        json_index = a.to_json(orient='index')
        parsed = json.loads(json_index)
        json.dumps(parsed, indent=4)
        # data = a.to_dict(orient='records')
        return parsed

@app.route('/getdatatypes/<path:file_path>', methods=["GET"])
def get_csv_datatypes(file_path):
	df = pd.read_csv(file_path)
	data_types=[]
	for column in df.columns: 
			data_types.append((column, str(df[column].dtype)))
	return jsonify(data_types)


@app.route('/get_column_names', methods=['GET'])
def get_column_names():
    # get the folder path from the POST request
    dir_path = "C:\\Users\\207065\\Desktop\\workspace\\backend\\Files"
    subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    folder_path = max(subdirs, key=os.path.getctime)
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    latest_file = max(files, key=os.path.getctime)

    # get the most recent CSV file in the folder
    csv_file_path = max([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')], key=os.path.getctime)
    
    # read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # get the column names from the DataFrame
    column_names = list(df.columns)

    # return the column names as a JSON response
    return {'column_names': column_names}


@app.route('/merge-csv-columns',methods=["POST"])
def merge_csv_columns():

    data = request.get_json()
    filename = data['filepath']
    column1 = data['column1']
    column2 = data['column2']

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(filename)

    # Merge the two columns into a new column
    if df[column1].dtype == object:
        df[column1 + "-" + column2] = df[column1].astype(str) + ' ' + df[column2].astype(str)
    else:
        df[column1+ "-" + column2] = df[column1].astype(str) +' ' + df[column2].astype(str)

    # Drop the original columns
    # df = df.drop([column1, column2], axis=1)

    # Write the modified DataFrame back to the original CSV file
    df.to_csv(filename, index=False)

    # Return a success message
    return ("ok")


@app.route('/get_filepath', methods=['GET'])
def get_path():
    # get the folder path from the POST request
    dir_path = "C:\\Users\\207065\\Desktop\\workspace\\backend\\Files"
    subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    latest_subdir = max(subdirs, key=os.path.getctime)
    files = [os.path.join(latest_subdir, f) for f in os.listdir(latest_subdir) if os.path.isfile(os.path.join(latest_subdir, f))]
    latest_file = max(files, key=os.path.getctime)

    # folder_path = ('C:\\Users\\207065\\Desktop\\workspace\\backend\\MappedFiles')

    # get the most recent CSV file in the folder
    csv_file_path = max([os.path.join(latest_subdir, f) for f in os.listdir(latest_subdir) if f.endswith('.csv')], key=os.path.getctime)
    return csv_file_path

def get_topic_subtopics(lda_model, vectorizer, topic_idx: int, n_words: int) -> Tuple[str, List[str]]:
    """
    Returns the top words and subtopics for a given topic index.
    """
    topic_top_words = []
    feature_names = vectorizer.get_feature_names_out()
    topic = lda_model.components_[topic_idx]
    top_words_idx = topic.argsort()[:-n_words-1:-1]
    top_words = [feature_names[i] for i in top_words_idx]
    topic_top_words.append("-".join(top_words))
    
    subtopic_top_words = []
    subtopic_probabilities = topic / topic.sum()
    subtopic_idx = subtopic_probabilities.argsort()[:-2:-1]
    subtopic_words_idx = lda_model.components_[subtopic_idx].argsort()[:-n_words-1:-1]
    subtopic_words = [feature_names[i] for i in subtopic_words_idx]
    subtopic_top_words.append("-".join(subtopic_words))
    
    return "-".join(topic_top_words), subtopic_top_words
    
# @app.route('/get_lda', methods=['POST'])
# def func():
#     data = request.get_json()
#     file_path = data['file_path']
#     max_df = float(data['max_df'])
#     min_df = int(data['min_df'])
#     n_components = int(data['n_components'])
#     random = int(data['random'])
#     column_name = data['column_name']

#     data = pd.read_csv(file_path)
#     # data = data[data[column_name].notna()]
#     # create a TF-IDF vectorizer
#     vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=1000, stop_words='english')

#     # fit and transform the data with the vectorizer
#     X = vectorizer.fit_transform(data[column_name].apply(lambda x: np.str_(x)))

#     # create an LDA model with 10 topics
#     lda = LatentDirichletAllocation(n_components=n_components, random_state=random)
#     lda.fit(X)
#     topic_probabilities = lda.transform(X)

#     # get the topics and their top two words
#     topic_top_words = []
#     feature_names = vectorizer.get_feature_names_out()
#     for topic_idx, topic in enumerate(lda.components_):
#         top_words_idx = topic.argsort()[:-3:-1]
#         top_words = [feature_names[i] for i in top_words_idx]
#         topic_top_words.append((topic_idx, "-".join(top_words)))

#     ############
#     data['topic_percent'] = np.max(topic_probabilities, axis=1) * 100
#     ##############

#     # add the topics and top words as a new column in the CSV file
#     current_datetime=datetime.now().strftime("%m-%d-%Y %H-%M")
#     str_current_time= str(current_datetime)

#     data["topic"] = lda.transform(X).argmax(axis=1)
#     data["top_words"] = [topic_top_words[i][1] for i in data["topic"]]

#     subtopic_top_words = []
#     for i, prob in enumerate(topic_probabilities):
#         if np.max(prob) * 100 < 70:
#             topic_idx = np.argmax(prob)
#             topic_word, subtopic_words = get_topic_subtopics(lda, vectorizer, topic_idx, 2)
#             subtopic_top_words.append(subtopic_words[0])
#         else:
#             subtopic_top_words.append(None)
    
#     data["subtopic_top_words"] = subtopic_top_words

#     dir_path = "C:\\Users\\207065\\Desktop\\workspace\\backend\\Files"
#     subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
#     latest_subdir = max(subdirs, key=os.path.getctime)
#     # files = [os.path.join(latest_subdir, f) for f in os.listdir(latest_subdir) if os.path.isfile(os.path.join(latest_subdir, f))]
#     # latest_file = max(files, key=os.path.getctime)
#     file_name = (latest_subdir + "\\LDA_" + str_current_time + ".csv")
#     data.to_csv(file_name, index=False)
#     # data.to_csv("output_file.csv", index=False)
            
#     return ("ok")




@app.route('/get_lda', methods=['POST'])
def func():
    data = request.get_json()
    file_path = data['file_path']
    max_df = float(data['max_df'])
    min_df = int(data['min_df'])
    n_components = int(data['n_components'])
    random = int(data['random'])
    column_name = data['column_name']

    data = pd.read_csv(file_path)
    # data = data[data[column_name].notna()]
    # create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=1000, stop_words='english')

    # fit and transform the data with the vectorizer
    X = vectorizer.fit_transform(data[column_name].apply(lambda x: np.str_(x)))

    # create an LDA model with 10 topics
    lda = LatentDirichletAllocation(n_components=n_components, random_state=random)
    lda.fit(X)
    topic_probabilities = lda.transform(X)

    # get the topics and their top two words
    topic_top_words = []
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-3:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topic_top_words.append((topic_idx, "-".join(top_words)))

    ############
    data['topic_percent'] = np.max(topic_probabilities, axis=1) * 100
    ##############

    # add the topics and top words as a new column in the CSV file
    current_datetime=datetime.now().strftime("%m-%d-%Y %H-%M")
    str_current_time= str(current_datetime)

    data["topic"] = lda.transform(X).argmax(axis=1)
    data["top_words"] = [topic_top_words[i][1] for i in data["topic"]]

    # add the subtopics and subtopic names as a new column in the CSV file
    # data["subtopic"] = ""
    # data["subtopic_name"] = ""
    # for i in range(len(data)):
    #     if data["topic_percent"][i] < 70:
    #         sub_X = X[i]
    #         sub_lda = LatentDirichletAllocation(n_components=2, random_state=random)
    #         sub_lda.fit(sub_X)
    #         sub_topic_probabilities = sub_lda.transform(sub_X)
    #         sub_topic_idx = sub_topic_probabilities.argmax()
    #         sub_topic_top_words_idx = sub_lda.components_[sub_topic_idx].argsort()[:-2:-1]
    #         sub_topic_top_words = [feature_names[i] for i in sub_topic_top_words_idx]
    #         sub_topic_name = "-".join(sub_topic_top_words)
    #         data["subtopic"][i] = sub_topic_idx
    #         data["subtopic_name"][i] = sub_topic_name

    dir_path = "C:\\Users\\207065\\Desktop\\workspace\\backend\\Files"
    subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    latest_subdir = max(subdirs, key=os.path.getctime)
    file_name = (latest_subdir + "\\LDA_" + str_current_time + ".csv")
    data.to_csv(file_name, index=False)          
    return ("ok")


@app.route('/recent_folder', methods=['GET'])
def recent_folder():
    folder_path = 'C:\\Users\\207065\\Desktop\\workspace\\backend\\Files'  # Replace with the actual path to your directory
    folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    recent_folder = max(folders, key=lambda f: os.path.getctime(os.path.join(folder_path, f)))
    return recent_folder


    
@app.route('/map_file', methods=['GET'])
def map_file():
    
    dir_path = "C:\\Users\\207065\\Desktop\\workspace\\backend\\Files"
    subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    latest_subdir = max(subdirs, key=os.path.getctime)
    files = [os.path.join(latest_subdir, f) for f in os.listdir(latest_subdir) if os.path.isfile(os.path.join(latest_subdir, f))]
    latest_file = max(files, key=os.path.getctime)
    return latest_subdir


@app.route('/rfile', methods=['GET'])
def rec_file():
    
    dir_path = "C:\\Users\\207065\\Desktop\\workspace\\backend\\Files"
    subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    latest_subdir = max(subdirs, key=os.path.getctime)
    files = [os.path.join(latest_subdir, f) for f in os.listdir(latest_subdir) if os.path.isfile(os.path.join(latest_subdir, f))]
    latest_file = max(files, key=os.path.getctime)
    return latest_file

@app.route('/project_studio_select', methods=['GET'])
def get_recently_added():
    folder_path = 'C:\\Users\\207065\\Desktop\\workspace\\backend\\Files'
    subdirectories = [os.path.join(folder_path, subdirectory) for subdirectory in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subdirectory))]
    most_recent_subdirectory = max(subdirectories, key=os.path.getmtime)
    file_names = os.listdir(most_recent_subdirectory)
    return jsonify(file_names)

# @app.route('/cleandata', methods=["POST"])
# def clean_text():    
#     data = request.get_json()
#     sent = data['file_path']
#     name = data['name']
#     removenumbersinbetween = (data['removenumbersinbetween'])

#     df = pd.read_csv(sent,low_memory=False)
#     new_df = df.copy()
#     a = new_df[name].apply(lambda x: x.lower() if type(x) == str else x)
#     a = a.to_string(index=False)
#     pattern = '[^\w\s]' # Removing punctuation
#     a = re.sub(pattern, '', a) 
#     if(removenumbersinbetween): 
#         pat = r'\w*\d\w*' # Removing words with numbers in between
#         new_df[name] = new_df[name].apply(lambda x: re.sub(pat, '', x) if type(x) == str else x)

#     # if(removenumbersinbetween): 
#     #     pat = '\w*\d\w*' # Removing words with numbers in between
#     #     new_df[name] = re.sub(pat, '', a) 
#     # # print(a)
#     my_list = a.strip().split('\n')
    
#     df1 = pd.DataFrame(my_list, columns=[name + '_cleaned'])
#     # print(df1.head(10))
#     df["Cleaned_Column"]= df1[[name + '_cleaned']]
#     dir_path = "C:\\Users\\207065\\Desktop\\workspace\\backend\\Files"
#     subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
#     latest_subdir = max(subdirs, key=os.path.getctime)
#     current_datetime=datetime.now().strftime("%m-%d-%Y %H-%M")
#     str_current_time= str(current_datetime)

#     file_name = (latest_subdir + "\\Cleaned_" + str_current_time + ".csv")
#     df.to_csv(file_name, index=False)
#     return "ok"

@app.route('/cleandata', methods=["POST"])
def clean_text():    
    # Get data from request JSON
    data = request.get_json()
    file_path = data['file_path']
    column_name = data['name']
    remove_numbers = data.get('removenumbersinbetween')
    print(remove_numbers)
    my_string = str(remove_numbers).capitalize()
    print(my_string) 
    df = pd.read_csv(file_path, low_memory=False)

    # Clean specified column
    cleaned_col = df[column_name].apply(lambda x: x.lower() if type(x) == str else x)  # convert to lowercase
    cleaned_col = cleaned_col.apply(lambda x: re.sub(r'[^\w\s]', '', x) if type(x) == str else x)  # remove punctuations
    if my_string:
        cleaned_col = cleaned_col.apply(lambda x: re.sub(r'\b\w*\d\w*\b', '', x) if type(x) == str else x)  # remove words with numbers

    # Add cleaned column to the DataFrame
    df['Cleaned_Column'] = cleaned_col

    # Save cleaned DataFrame to a new CSV file
    # dir_path = "C:\\Users\\207065\\Desktop\\workspace\\backend\\Files"
    # subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    # latest_subdir = max(subdirs, key=os.path.getctime)
    # current_datetime=datetime.now().strftime("%m-%d-%Y %H-%M")
    # str_current_time= str(current_datetime)

    # file_name = (latest_subdir + "\\Cleaned_" + str_current_time + ".csv")
    # df.to_csv(file_name, index=False)

    cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
    df.to_csv(cleaned_file_path, index=False)

    return jsonify({'message': 'Data cleaned successfully'})
    
@app.route('/lemmatize_csv', methods=['POST'])
def lemmatize_csv():
    data = request.get_json()
    file_name = data['file_path']
    column_name = data['name']

    df = pd.read_csv(file_name)

    # Perform lemmatization on specified column
    df["Lemmatized_Column"] = df[column_name].apply(lambda x: " ".join([token.lemma_ for token in nlp(x)]))
    print(df.head())
    dir_path = "C:\\Users\\207065\\Desktop\\workspace\\backend\\Files"
    subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    latest_subdir = max(subdirs, key=os.path.getctime)
    current_datetime=datetime.now().strftime("%m-%d-%Y %H-%M")
    str_current_time= str(current_datetime)

    file_name = (latest_subdir + "\\Lemmatized_" + str_current_time + ".csv")
    # df.to_csv(file_name, index=False)
    # unique_id = str(uuid.uuid4())
    # file_name = f"C:\\Users\\192301\\PROJECTS\\CLEANING\\cleanedfiles\\data_{unique_id}.csv"
    df.to_csv(file_name, index=False)
    return "ok"

@app.route('/postags', methods=['POST'])
def get_POS_tags():

    data = request.get_json()
    file_name = data['file_path']
    column_name = data['name']
    df = pd.read_csv(file_name)
    df["Final_cleaned"] = df[column_name].apply(lambda x: ' '.join(TextBlob(x).noun_phrases))
    df.to_csv(file_name, index=False)
    
    return "ok"

@app.route('/data', methods=['GET'])
def get_dat():
    dir_path = "C:\\Users\\207065\\Desktop\\workspace\\backend\\Files"
    subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    latest_subdir = max(subdirs, key=os.path.getctime)
    files = [os.path.join(latest_subdir, f) for f in os.listdir(latest_subdir) if os.path.isfile(os.path.join(latest_subdir, f))]
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, 'r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]

    return jsonify(data)

@app.route('/api/csv_data', methods=['GET'])
def csv_data():
    dir_path = "C:\\Users\\207065\\Desktop\\workspace\\backend\\Files"
    subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    latest_subdir = max(subdirs, key=os.path.getctime)
    files = [os.path.join(latest_subdir, f) for f in os.listdir(latest_subdir) if os.path.isfile(os.path.join(latest_subdir, f))]
    latest_file = max(files, key=os.path.getctime)

    df = pd.read_csv(latest_file)
    data = []
    with open(latest_file, newline='',encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data = [row for row in reader]
    return jsonify(data)


@app.route('/api/csv')
def csv_dat():
    dir_path = "C:\\Users\\207065\\Desktop\\workspace\\backend\\Files"
    subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    latest_subdir = max(subdirs, key=os.path.getctime)
    files = [os.path.join(latest_subdir, f) for f in os.listdir(latest_subdir) if os.path.isfile(os.path.join(latest_subdir, f))]
    latest_file = max(files, key=os.path.getctime)

    df = pd.read_csv(latest_file)
    data = []
    with open(latest_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append({
                'name': row['topic'],
                'value': (row['top_words']),
                'category': row['subtopic_name']
            })
    return {'data': data}
# @app.route('/plot_piechart', methods=['GET'])
# def plot_piechart():

#     column_name = "top_words"
#     dir_path = "C:\\Users\\207065\\Desktop\\workspace\\backend\\Files"
#     subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
#     latest_subdir = max(subdirs, key=os.path.getctime)
#     files = [os.path.join(latest_subdir, f) for f in os.listdir(latest_subdir) if os.path.isfile(os.path.join(latest_subdir, f))]
#     latest_file = max(files, key=os.path.getctime)
#     filename = pd.read_csv(latest_file)
#     freq_table = filename[column_name].value_counts()
#     percentages = freq_table.values / len(filename) * 100
#     plt.pie(percentages, labels=freq_table.index, autopct='%1.1f%%')
#     plt.title('Pie Chart of ' + column_name)

#     current_datetime=datetime.now().strftime("%m-%d-%Y %H-%M")
#     str_current_time= str(current_datetime)

#     name = ( "Pie_Chart_" + str_current_time + ".png")

#     plt.savefig(name)
#     plt.show
#     return jsonify({'chart': 'pie_chart.png'})


@app.route('/plot_piechart', methods=['GET'])
def plot_piechart():
    # Get the file name and column name from the POST request
    # filename = request.args.get('filename')
    # column_name = request.args.get('column_name')
    column_name = "top_words"
    dir_path = "C:\\Users\\207065\\Desktop\\workspace\\backend\\Files"
    subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    latest_subdir = max(subdirs, key=os.path.getctime)
    files = [os.path.join(latest_subdir, f) for f in os.listdir(latest_subdir) if os.path.isfile(os.path.join(latest_subdir, f))]
    latest_file = max(files, key=os.path.getctime)
    filename = pd.read_csv(latest_file)
    # Read the CSV file into a pandas dataframe
    # filename = pd.read_csv(filename)
    groupby_data = filename.groupby('top_words')['topic_percent'].sum().reset_index()

    # create a pie chart
    fig, ax = plt.subplots()
    ax.pie(groupby_data['topic_percent'], labels=groupby_data['top_words'], autopct='%1.1f%%')
    ax.axis('equal')
    ax.set_title('Top Words Distribution')

    # save the pie chart to a PNG file
    current_datetime = datetime.now().strftime("%m-%d-%Y %H-%M")
    file_name = 'pie_chart_' + current_datetime + '.png'
    fig.savefig(file_name)
    image_path = os.path.abspath(file_name)

    return Response(image_path)


@app.route('/pie-chart', methods=['POST'])
def generate_pie_chart():
    file_name = request.args.get('file_name')
    num_topics = int(request.args.get('num_topics'))
    data = request.get_json()
    filename = data['file_name']
    num_topics = int(data['num_topics'])
    df = pd.read_csv(file_name)
    top_words = []
    for i in range(num_topics):
        words = df.loc[df['topic'] == i]['top_words'].iloc[0].split(',')
        top_words.append(words[0])

    values = [len(df.loc[df['topic'] == i]) for i in range(num_topics)]
    labels = top_words
    colors = ['red', 'blue', 'green', 'orange'][:num_topics]
    plt.pie(values, labels=labels, colors=colors)

    img = BytesIO()
    plt.savefig(img, format='jpeg')
    img.seek(0)
    chart_data = base64.b64encode(img.getvalue()).decode('utf-8')
    chart = 'data:image/png;base64,' + chart_data

    # Return the pie chart data as a JSON response
    return jsonify({'chart': chart, 'labels': labels, 'values': values})

@app.route('/api/download', methods=['POST'])
def download_file():
    data = request.get_json()
    path = data.get('path')
    # Read data from path or generate it
    file_contents = open(path, 'rb').read() if path else b'column1,column2\n1,4\n2,5\n3,6\n'
    return file_contents


if (__name__) == "__main__":
    app.run()