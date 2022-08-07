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
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk import sent_tokenize
nltk.download('punkt')
nltk.download('popular')
import pandas as pd
import numpy as np
import sqlite3,csv
from sqlite3 import Error
import docx2txt
import pdfplumber


pd.set_option('display.max_colwidth', None)


import streamlit as st

st.set_page_config(
    page_title="Add Resume",
    page_icon="👋",
)

st.subheader("Add New Resume To BestFynd Database ")

st.text("")
st.text("")



#elif choice == "DocumentFiles":
# st.subheader("Document Files")
docx_file = st.file_uploader("Upload Document", type=["pdf","docx","txt"])

if st.button("Process"):
    if docx_file is not None:

        # file_details = {"filename":docx_file.name, "filetype":docx_file.type,
        #                 "filesize":docx_file.size}
        # st.write(file_details)

        if docx_file.type == "text/plain":
            # Read as string (decode bytes to string)
            raw_text = str(docx_file.read(),"utf-8")
            st.text(raw_text)
        elif docx_file.type == "application/pdf":
            try:
                with pdfplumber.open(docx_file) as pdf:
                    pages = pdf.pages[0]
                    st.write(pages.extract_text())
            except:
                st.write("None")

        else:
            raw_text = docx2txt.process(docx_file)
            st.write(raw_text)



        df = pd.DataFrame(columns=['Category','Resume','Clean_Resume','Skills','Phone_Number','Email_id'])
        df['Resume'] = [raw_text]
        Resume = str(df['Resume'][0])


        def clean_text(text) :
            clean_text = re.sub("\\n"," ",text)
            clean_text = re.sub("\\r"," ",clean_text)
            clean_text = re.sub("â ¢"," ",clean_text)
            clean_text = re.sub("â"," ",clean_text)
            clean_text = re.sub("â"," ",clean_text)
            clean_text = re.sub("â¢"," ",clean_text)
            clean_text = re.sub("ï¶"," ",clean_text)
            clean_text = re.sub("  "," ",clean_text)
            clean_text = clean_text.lower()

            return(clean_text)
        df['Clean_Resume']=df['Resume'].apply(lambda x: clean_text(x))

        Clean_Resume = str(df["Clean_Resume"][0])


        # For extracting skills
        df_skill = pd.read_fwf('list_of_skills.txt',header=None)
        df_skill = df_skill[0].str.replace(',','')
        df_skill = pd.DataFrame(df_skill)
        SKILLS_DB = df_skill[0].tolist()
        
        
        
        def extract_skills(text):
            #stop_words = set(nltk.corpus.stopwords.words('english'))
            stop_words = nltk.corpus.stopwords.words('english')
            word_tokens = nltk.tokenize.word_tokenize(text)
            # remove the stop words
            filtered_tokens = [w for w in word_tokens if w not in stop_words]
            # remove the punctuation
            filtered_tokens = [w for w in word_tokens if w.isalpha()]
            # generate bigrams and trigrams (such as artificial intelligence)
            bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens,1, 2)))
            # we create a set to keep the results in.
            found_skills = set()
            # we search for each token in our skills database
            for token in filtered_tokens:
                if token.lower() in SKILLS_DB:
                    found_skills.add(token)
        #     we search for each bigram and trigram in our skills database
            for ngram in bigrams_trigrams:
                if ngram.lower() in SKILLS_DB:
                    found_skills.add(ngram)

            return found_skills
        #df['Skills']=df['Clean_Resume'].apply(lambda x: extract_skills(x))
        #Skills = str(df['Skills'][0])

        # For extracting phone number from resume
        def find_phone_number(text):
        #     ph_no = re.findall(r"\b\d{10}\b",text)
            ph_no = re.findall(re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]'),text)
            return "".join(ph_no)
        df['Phone_Number']=df['Clean_Resume'].apply(lambda x: find_phone_number(x))
        Phone_Number = str(df['Phone_Number'][0])

        # For extracting email_id from resume
        def find_email(text):
            email = re.findall(re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+'),text)
            return "".join(email)
        df['Email_id']=df['Resume'].apply(lambda x: find_email(x))
        Email_id = str(df['Email_id'][0])

        #For insering a new resume in the database
        def run_query():
            conn = sqlite3.connect("C:/Users/ktyagi/Desktop/ISB/FP1/Project 3/resume_database.db")
            c = conn.cursor()
            # c.execute("INSERT INTO Resume_Table ('Category','Resume','Clean_Resume','Skills','Phone_Number','Email_id') VALUES (?,?,?,?,?,?)",('cat1',Resume,Clean_Resume,Skills,Phone_Number,Email_id))
            conn.commit()
            conn.close()
            st.success('Resume uploaded successfully.')
            return()

        if st.button('Update',on_click = run_query):
            run_query()
