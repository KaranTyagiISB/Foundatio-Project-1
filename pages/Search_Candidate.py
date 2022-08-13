#from pyresparser import ResumeParser
import os
from docx import Document
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import re
import spacy
#from pyresparser import ResumeParser
import nltk
nltk.download('stopwords')
import pandas as pd
import numpy as np
import sqlite3,csv
from sqlite3 import Error
import docx2txt
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2

pd.set_option('display.max_colwidth', None)


import streamlit as st

st.set_page_config(
    page_title="Search",
    page_icon="👋",
)

st.subheader("Job Description Match")

st.text("")
st.text("")

# conn = sqlite3.connect("C:/Users/ktyagi/Desktop/ISB/FP1/Project 3/resume_database.db")
# c= conn.cursor()
#
# c.execute('Select * from Resume_Table')
# data = c.fetchall()
#
# title = [i[0] for i in c.description]
# df_head = pd.DataFrame(title)
# df_head  = df_head.T
#
# all_data = []
# for row in data:
#     all_data.append(row)
# df = pd.DataFrame(all_data)
#
# df_all = df_head.append(df)
# df_all = df_all.rename(columns=df_all.iloc[0]).drop(df_all.index[0])


conn = psycopg2.connect(database="bestfynd",
                        host="ec2-3-6-116-139.ap-south-1.compute.amazonaws.com",
                        user="postgres",
                        password="mynewpassword",
                        port="5432")

c = conn.cursor()
c.execute("SELECT * FROM resume_table")


data = c.fetchall()

title = [i[0] for i in c.description]
df_head = pd.DataFrame(title)
df_head  = df_head.T

all_data = []
for row in data:
    all_data.append(row)
df = pd.DataFrame(all_data)

df_all = df_head.append(df)
df_all = df_all.rename(columns=df_all.iloc[0]).drop(df_all.index[0])

docx_file = st.file_uploader("Upload JD", type=["pdf","docx","txt"])

Match_Threshold = st.text_input('Please enter the Threshold (0-100)', '70')
st.write('Similarity score Threshold is ', Match_Threshold)
Match_Threshold = int(Match_Threshold)

if st.button("Process"):
    if docx_file is not None:

        if docx_file.type == "text/plain":
            # Read as string (decode bytes to string)
            raw_text = str(docx_file.read(),"utf-8")
            # st.text(raw_text)

        elif docx_file.type == "application/pdf":
            try:
                with pdfplumber.open(docx_file) as pdf:
                    pages = pdf.pages[0]
                    st.write(pages.extract_text())
            except:
                st.write("None")


        else:
            raw_text = docx2txt.process(docx_file)


    def match(resume) :

        text = [resume, raw_text]

        idf = TfidfVectorizer()
        count_matrix = idf.fit_transform(text)
        match_score = round(cosine_similarity(count_matrix)[0,1]*100,2)

        return match_score

    stopwords = nltk.corpus.stopwords.words('english')

    def top_words(resume) :

        text = [resume, raw_text]
        idf = TfidfVectorizer()
        count_matrix = idf.fit_transform(text)

        col_names = idf.get_feature_names()

        DTM = pd.DataFrame(count_matrix.todense(),columns = col_names).T.reset_index()
        DTM.columns = ["Words","Resume","JD"]
        clean_DTM = DTM[~DTM.Words.isin(stopwords)]

        df_top = clean_DTM[clean_DTM['Words'].isin(text[0].lower().split())].sort_values(by = ['Resume','JD'], ascending =[False,False])
        df_top = df_top[df_top.JD > 0].reset_index(drop = True)
        top_words = list(df_top.Words[0:5])

        return(top_words)

    df_all["Match_Score"] = df_all["Clean_Resume"].apply(lambda x : match(x))
    df_all["Top_Match_Words"] = df_all["Clean_Resume"].apply(lambda x : top_words(x))
    df_all = df_all.sort_values("Match_Score",ascending= False)
    df_all.drop_duplicates(subset = 'Skills', keep = 'first', inplace = True)
    print(df_all)


    # Match_thres = 70
    df_all["Resume_Passed"] = np.where(df_all["Match_Score"]>= Match_Threshold,1,0)
    df_filter = df_all.loc[df_all['Resume_Passed']==1]

    df_filter = df_filter[['Category','Skills','Phone_Number','Email_id','Qualification','City','Top_Match_Words','Match_Score']]
    df_filter['Match_Score'] = df_filter['Match_Score'].round(2)
    st.write(df_filter)
    count = df_filter.shape[0]
    st.write("Number of Suitable Candidates are: ",count)

    count = df_filter.shape[0]
