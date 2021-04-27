import pandas as pd
import os

# path to metadata archive (downloadable from https://www.kaggle.com/Cornell-University/arxiv)
home = os.path.expanduser("~")
path_archive = os.path.join(home,'Downloads/archive.zip')

# load the archive in pandas dataframe
df = pd.read_json(path_archive,lines=True)

# The category and year we are interested in
category = 'quant-ph'
year = 2016

# The relevant columns for us
relevant_columns = ['id','title','abstract','update_date','authors_parsed']

# Select papers according to category
relevant_items = df['categories'].apply(lambda item : category in item)
df_quant = df[relevant_items][relevant_columns].copy()
df_quant.drop_duplicates(subset='id',inplace=True)

# # Select papers according to year
df_quant['update_date'] = df_quant['update_date'].apply(lambda item: item[:4])
relevant_years = df_quant['update_date'] >= str(year)
df_quant = df_quant[relevant_years]

df_quant.reset_index(drop=True,inplace=True)

df_quant = df_quant.rename(columns={'update_date':'year','authors_parsed':'author_list'})

# save cleaned dataframe 
df_quant.to_pickle('./dataset/dataset_quant_ph.pkl')