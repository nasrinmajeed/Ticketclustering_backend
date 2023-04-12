import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from datetime import datetime

class get_lda():

    def lda(self,file_path,max_df,min_df,n_components,random,column):
    # read the data from the CSV file
        data = pd.read_csv(file_path)

        # create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=1000, stop_words='english')

        # fit and transform the data with the vectorizer
        X = vectorizer.fit_transform(data[column])

        # create an LDA model with 10 topics
        lda = LatentDirichletAllocation(n_components=n_components, random_state=random)
        lda.fit(X)

        # get the topics and their top two words
        topic_top_words = []
        feature_names = vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-3:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_top_words.append((topic_idx, ", ".join(top_words)))

        # add the topics and top words as a new column in the CSV file
        current_datetime=datetime.now().strftime("%m-%d-%Y %H-%M")
        str_current_time= str(current_datetime)

        data["topic"] = lda.transform(X).argmax(axis=1)
        data["top_words"] = [topic_top_words[i][1] for i in data["topic"]]
        data.to_csv("C:\\Users\\207065\\Desktop\\workspace\\backend\\LDAFiles\\Doc"+ str_current_time + ".csv",sep="\t", index=False)
        # data.to_csv("output_file.csv", index=False)
        return("OK")
