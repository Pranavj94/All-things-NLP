# Text cleaning

# Importing libraries

import pandas as pd
import re
import json
import numpy as np
import random
from tqdm import tqdm
tqdm.pandas()


# Libraries for NLP tasks
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 
from nltk.corpus import wordnet 



def lower_case(text):
    return(text.lower())

def remove_linebreaks(text):
    text = re.sub(r"\n", " ", text)
    return(text)

def identify_email_id(text):
    text = re.sub(r"\S+@\S+", "email_id", text)
    return(text)

def identify_webaddress(text):
    text=re.sub(r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*", 'webaddress',text)
    return(text)

def identify_money_symbol(text):
    text=re.sub(r'£|\$', 'money ',text)
    return(text)

def identify_phone_number(text):
    text=re.sub(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})', 'phone_number',text)
    return(text)

def identify_number(text):
    text=re.sub(r'\d+(\.\d+)?', 'number',text)
    return(text)

def remove_punctuation(text):
    text=re.sub(r'[^\w\s]', ' ',text)
    return(text)

def remove_multiple_spaces(text):
    text=re.sub(r'\s+', ' ',text)
    return(text)

def remove_trailing_white_spaces(text):
    text=text.strip()
    return(text)

def get_lemmatized_text(text):
    text=' '.join(wordnet_lemmatizer.lemmatize(term) for term in text.split())
    return(text)

# Remove stop words
# def remove_stopwords(text):
#     text=' '.join(term for term in text.split() if term not in stop_words)
#     return(text)
    
    

# Slang Correction - The text contains lots of slags. Since the option to create a comprehensive slang to actual word mapping
# is not feasible. Using slangs/contractions mapping dictionary for corrections. Accuracy can be improved with a more extensive 
#mapping.

# with open('slangs_contractions.json') as json_file:
#     slangs_contractions = json.load(json_file)
    
# def remove_slangs_contractions(text):
#     text=' '.join(slangs_contractions[term] if term in slangs_contractions.keys() else term for term in text.split())
#     return(text)
    
# Ginger it library is accurate for slang correction, but the processing time is extremely high
# def remove_slangs(text):
#     parser = GingerIt()
#     out=parser.parse(text)
#     return(out['result'])


def preprocess_text(text,processing_steps):
    # text=lower_case(text)
    # text=remove_linebreaks(text)
    # text=identify_email_id(text)
    # text=identify_webaddress(text)
    # text=identify_money_symbol(text)
    # text=identify_phone_number(text)
    # text=remove_punctuation(text)
    # text=remove_multiple_spaces(text)
    # text=remove_trailing_white_spaces(text)
    # text=get_lemmatized_text(text)
    #text=remove_stopwords(text)
    #text=remove_slangs_contractions(text)

    for processes in processing_steps:
        text=processes(text)
    
    return(text)


def clean_text(input_df,column_name,inplace=False,exclude_list=[]):

    # Creating a new column if inplace = False
    if inplace == True:
        cleaned_column_name=str(column_name)
    else:
        cleaned_column_name=str(column_name)+'_cleaned'

    processing_steps=[lower_case,remove_linebreaks,identify_email_id,identify_webaddress,
    identify_money_symbol,identify_phone_number,remove_punctuation,remove_multiple_spaces,
    remove_trailing_white_spaces,get_lemmatized_text]
        
    processing_steps=[items for items in processing_steps if items not in exclude_list]

    input_df[cleaned_column_name]=input_df[column_name].progress_apply(lambda x:preprocess_text(x,processing_steps))

    return(input_df)
      
# Adding a random change
