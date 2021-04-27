from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel

import re
import string

import numpy as np

from itertools import combinations

# clean the text and use lemmatizer
def clean_text_lemmatize(item,lemmatizer,stopwords):
    
    # remove latex equations
    item = re.sub('\$+.*?\$+','',item)
    
    # tokenize and remove punctuation
    item = re.findall('[a-zA-Z0-9]+',item)
    
    # lowecase everything
    item = [word.lower() for word in item]
    
    # remove english stopwords
    item = [word for word in item if word not in stopwords]
    
    # lemmatize the words
    item = [lemmatizer.lemmatize(word) for word in item]
    
    return item

# Join the author name and surname with undesrcore (and remove the punctuation)
def chained_author(author):
    
    author = ''.join(char for char in author if char not in string.punctuation)
    author = re.sub('\W','_',author)
    
    return author

# Get list of authors from single string
def author_list(item):
    
    # If string contains comma-and, replace with comma
    author_string = re.sub(', and ',', ',item)
    
    # If string contains and, replace with comma
    author_string = re.sub(' and ',', ',author_string)
    
    # Split at the commas
    author_string = author_string.split(', ')
    
    # chain together name and surname
    author_string = [chained_author(au) for au in author_string]
    
    return author_string

# map scited and unscited fields into a vector repesentation (TF-IDF or others)
def vectorize_field(token_scited,token_unscited,object_vectorizer):

    # This is needed since we did all the pre-processing outside sklearn
    tokens_to_string = lambda word_list: ' '.join(word_list)

    string_scited = [tokens_to_string(token) for token in token_scited]
    string_unscited = [tokens_to_string(token) for token in token_unscited]

    vectorizer = object_vectorizer(analyzer=str.split)

    scited_vectors = vectorizer.fit_transform(string_scited)
    unscited_vectors = vectorizer.transform(string_unscited)
    
    return scited_vectors, unscited_vectors

# Create similarity matrix between scited and unscited papers.
# - this is the product of text, author, and year similarities
def similarity_matrix(scited_papers,unscited_papers,gamma_authors,gamma_year):

    # Cosine similarity for the text of the papers
    similarity_text = 1 - pairwise_distances(scited_papers['text'],
                                             unscited_papers['text'],
                                             metric='cosine',
                                             n_jobs=-1
                                            )

    # Cosine distance for the authors of the papers
    distance_matrix_authors = pairwise_distances(scited_papers['authors'],
                                                 unscited_papers['authors'],
                                                 metric='cosine',
                                                 n_jobs=-1
                                                )

    similarity_authors = np.exp(-gamma_authors*distance_matrix_authors)

    # Radial base function for year similarity (uses euclidean distance between years)
    similarity_year = rbf_kernel(scited_papers['year'],
                                 unscited_papers['year'],
                                 gamma_year
                                )

    return similarity_text * similarity_authors * similarity_year

# Implementation of content-based filtering for scirate papers
class ContentBasedFiltering():
    
    def __init__(self,similarity_matrix,K,N):
        
        # The M1xM2 similarity matrix (M1 = # scited papers, M2 = # unscited papers)
        self.similarity_matrix = similarity_matrix
        
        # Number of nearest neighbours to use for computing the score of the paper
        self.K = K
        
        # Number of unscited papers to show in the top-N list
        self.N = N
    
    # Set the array of scores for the scited papers (needed for collective_method)
    def set_score_scited(self,scores):
        self.scores = scores
       
    # fit the model using the user-centric technique or the collective one
    def fit(self,user_centric=True):
        
        if user_centric:
            self.user_centric_model()
        else:
            self.collective_model()
    
    # Use similarties between scited and unscited papers to create top-N list
    def user_centric_model(self):
        
        # Sort each column of the similarity matrix independently.
        # Get the sorted similarities between 1 unscited paper and all scited ones. 
        unscited_sorted_similarity = np.sort(self.similarity_matrix,axis=0)

        # Take the mean similarity of unscited papers by averagin over the K most similar papers.
        unscited_mean_similarity = np.mean(unscited_sorted_similarity[-self.K:,:],axis=0)

        # Sort the mean similarities of the unscited papers
        unscited_sorted_most_similar = np.argsort(unscited_mean_similarity)

        # Get the N unscited papers that 
        self.top_N_list = np.flip(unscited_sorted_most_similar[-self.N:])
    
    # Use similarties between scited and unscited papers together
    # with (collective) scited scores to create top-N list
    def collective_model(self):
        
        if self.scores is None:
            raise('Score of the scited papers is needed for this method.')
        
        # Sort the unscited paper based on their similarity score,
        # and get the indices of the sorting
        uscited_sorted_indices = np.argsort(self.similarity_matrix,axis=0)

        # For each unscited paper, get the similarity score of the K most similar scited papers
        unscited_sorted_similarity = np.take_along_axis(self.similarity_matrix,
                                                        uscited_sorted_indices[-self.K:,:],
                                                        axis=0
                                                       )

        # For each uscited paper, get the score assigned to the K most similar scited papers
        unscited_sorted_score = self.scores[uscited_sorted_indices[-self.K:,:]]
        
        # Computed weighted score for each unscited paper
        unscited_weighted_score = np.sum(unscited_sorted_similarity*unscited_sorted_score,axis=0)
        unscited_total_weight = np.sum(unscited_sorted_similarity,axis=0)

        unscited_mean_score = unscited_weighted_score / unscited_total_weight
        
        # Sort the mean score of the unscited papers
        unscited_sorted_higer_score = np.argsort(unscited_mean_score)
        
        # Get the N unscited papers that 
        self.top_N_list = np.flip(unscited_sorted_higer_score[-self.N:])
        
    # show the top-N list using the dataframe of unscited papers
    def show_top_n_list(self,df_unscited):
        
        for i, entry in enumerate(self.top_N_list):
            suggested_paper = df_unscited.iloc[entry]

            print('{}) {}\n'.format(i+1,suggested_paper['title']))
            print(', '.join(suggested_paper['author_list'])+'\n')
            print(suggested_paper['abstract'])
                        
# Compute hit rate of the recommendation system 
def compute_hit_rate(matrix_unscited,matrix_scited,N_scited,K,N,scores=None,user_centric=True):
    
    hit = 0
    for i in range(N_scited):

        # Get similarity row for the left-out scited paper
        similarity_row_left_out = np.delete(matrix_scited[i],i).reshape((N_scited-1,1))

        # Get similarity matrix for unscited papers without left-out row
        similarity_matrix_unscited = np.delete(matrix_unscited,i,axis=0)

        # Get similarity matrix for leave-one-out cross-validation
        similarity_matrix = np.hstack((similarity_row_left_out,similarity_matrix_unscited))

        model = ContentBasedFiltering(similarity_matrix, K, N)
        
        if not user_centric:
            model.set_score_scited(np.delete(scores,i))
            
        model.fit(user_centric)

        # We placed the left out paper at the beginning of the similarity matrix
        if 0 in model.top_N_list:
            hit += 1

    return hit/N_scited

# Diversity measure for the suggestions of the model
def diversity(similarity_matrix, top_n_list):
    
    # The similarity is obtained by computing the average similarity of each pair of suggestion
    similarity = np.mean([similarity_matrix[i,j] for i,j in combinations(top_n_list, 2)])
    
    return 1 - similarity