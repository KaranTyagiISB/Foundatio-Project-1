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
nltk.download("stopwords")
nltk.download("punkt")
import sqlite3,csv
from sqlite3 import Error
import docx2txt
import pdfplumber
import psycopg2


pd.set_option('display.max_colwidth', None)


import streamlit as st

st.set_page_config(
    page_title="Add Resume",
    page_icon="👋",
)

st.subheader("Add New Resume To BestFynd Database ")

st.text("")
st.text("")


with st.form("my_form"):
    Category = st.selectbox(
    'Please select the catgory',
    ('None','Data Science','HR','Advocate','Arts','Web Designing','Mechanical Engineer','Sales','Health and fitness','Civil Engineer',
    'Java Developer','Business Analyst','SAP Developer','Automation Testing','Electrical Engineering','Operations Manager'
    'Python Developer','DevOps Engineer','Network Security Engineer','PMO','Database','Hadoop','ETL Developer','DotNet Developer'
    'Blockchain','Testing'))
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("This candidate is a : ", Category)

# st.write("Outside the form", Category)


#elif choice == "DocumentFiles":
# st.subheader("Document Files")
docx_file = st.file_uploader("Upload Document", type=["pdf","docx","txt"])

if st.button("Process"):
    if docx_file is not None:
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



        df = pd.DataFrame(columns=['Category','Resume','Clean_Resume','Qualification','City','Skills','Phone_Number','Email_id'])
        df['Resume'] = [raw_text]
        Resume = str(df['Resume'][0])


        def clean_text(text) :
            clean_text = re.sub("[\n,\r,\n\n,â¢,,\t]","",text)
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


        # For extracting Qualification
        df_qual = pd.read_fwf('list_of_degree.txt',header=None)
        df_qual = df_qual[0].str.replace(',','')
        df_qual = pd.DataFrame(df_qual)
        QUAL_DB = df_qual[0].tolist()

        def extract_qual(text):
            stop_words = set(nltk.corpus.stopwords.words('english'))
            word_tokens = nltk.tokenize.word_tokenize(text)
            filtered_tokens = [w for w in word_tokens if w not in stop_words]
            filtered_tokens = [w for w in word_tokens if w.isalpha()]
            bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens,1, 2)))
            found_qual = set()
            for token in filtered_tokens:
                if token.lower() in QUAL_DB:
                    found_qual.add(token)
            return found_qual
        df['Qualification']=df['Clean_Resume'].apply(lambda x: extract_qual(x))
        Qualification = str(df["Qualification"][0])


        # For extracting cities
        df_city = pd.read_fwf('list_of_cities.txt',header=None)
        df_city = df_city[0].str.replace(',','')
        df_city = pd.DataFrame(df_city)
        CITY_DB = df_city[0].tolist()
        # For extracting city

        def extract_city(text):
            stop_words = set(nltk.corpus.stopwords.words('english'))
            word_tokens = nltk.tokenize.word_tokenize(text)
            filtered_tokens = [w for w in word_tokens if w not in stop_words]
            filtered_tokens = [w for w in word_tokens if w.isalpha()]
            bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens,1, 2)))
            found_city = set()
            for token in filtered_tokens:
                if token.lower() in CITY_DB:
                    found_city.add(token)
            return found_city

        df['City']=df['Clean_Resume'].apply(lambda x: extract_city(x))
        City = str(df["City"][0])

        # For extracting skills
        df_skill = pd.read_fwf('list_of_skills.txt',header=None)
        df_skill = df_skill[0].str.replace(',','')
        df_skill = pd.DataFrame(df_skill)
        SKILLS_DB = df_skill[0].tolist()

        def extract_skills(text):
            stop_words = set(nltk.corpus.stopwords.words('english'))
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
        df['Skills']=df['Clean_Resume'].apply(lambda x: extract_skills(x))
        Skills = str(df['Skills'][0])

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

            conn = psycopg2.connect(database="bestfynd",
                        host="ec2-3-6-116-139.ap-south-1.compute.amazonaws.com",
                        user="postgres",
                        password="mynewpassword",
                        port="5432")
            c = conn.cursor()
            #c.execute("INSERT INTO resume_table ('Category','Resume','Clean_Resume','Qualification','City','Skills','Phone_Number','Email_id') VALUES (?,?,?,?,?,?,?,?)",(Category,Resume,Clean_Resume,Qualification,City,Skills,Phone_Number,Email_id))
            #postgres_insert_query = """ INSERT INTO resume_table (Category,Resume,Clean_Resume,Qualification,City,Skills,Phone_Number,Email_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"""
            #record_to_insert = (Category,Resume,Clean_Resume,Qualification,City,Skills,Phone_Number,Email_id)
            try:
                cursor.execute("INSERT INTO resume_table (Category,Resume,Clean_Resume,Qualification,City,Skills,Phone_Number,Email_id) VALUES (Category,Resume,Clean_Resume,Qualification,City,Skills,Phone_Number,Email_id)")
            except NameError as ER:
                cursor.execute("INSERT INTO resume_table (Category,Resume,Clean_Resume,Qualification,City,Skills,Phone_Number,Email_id) VALUES (Category,Resume,Clean_Resume,Qualification,City,Skills,Phone_Number,Email_id)")
            conn.commit()
            conn.close()
            st.success('Resume uploaded successfully.')
            return()


        if st.button('Upload Resume',on_click = run_query):
            run_query()
