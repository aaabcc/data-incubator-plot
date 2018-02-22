import pandas as pd
import gzip
import datetime
from matplotlib import pyplot as plt
#import operator

#read data or partial data based on a subset of variable names

sub_set  = {'asin', 'price', 'brand'}

def parse(path, sub_set):
	g = gzip.open(path, 'rb')
	if sub_set is None:
		for l in g:
			yield eval(l)
	else:
		for l in g:
		  l_dict = eval(l)
		  if sub_set.issubset(set(l_dict.keys())):
			  yield {key:l_dict[key] for key in sub_set}
		  else:
			  yield None
	
"""
	 getDF(r'data\metadata.json.gz', {'asin', 'price', 'brand'}, 100)
	 test = getDF(r'data\metadata.json.gz', None, 10000)                       
	 getDF(r'data\metadata.json.gz', {'asin', 'price', 'brand'}, 0)     #return all
"""
def getDF(path, sub_set, row_num):
    i = 0  
    df = {}
    for d in parse(path, sub_set):
        if d is not None:
            df[i] = d
            i += 1
            if row_num > 0 and i > row_num:
                break
    return pd.DataFrame.from_dict(df, orient='index')		              
                  
path = 'data\metadata.json.gz'

video_game= getDF(r'data\reviews_Video_Games_5.json.gz', None, 0)

videogame_productname =  getDF(path, sub_set,10)


def get_rank_of_meta_data(video_game_meta):
    video_game_meta = video_game_meta.reset_index(drop = True)
    i = 0
    for item in video_game_meta.salesRank:
        if type(item) is dict and len(item)>0:
            video_game_meta.loc[i, 'rank'] = list(item.values())[0]
        else:
            video_game_meta.loc[i, 'rank'] = 10000000
        i += 1
    video_game_meta = video_game_meta.sort_values(by='rank')
    return video_game_meta
video_game_meta = getDF(r'data\meta_Video_Games.json.gz', None, 0)
video_game_meta = get_rank_of_meta_data(video_game_meta)


video_game['asin'].value_counts()

#check attributes of the data: how many products? 
#how many days for each product? any missing data? Fomr 1999-10-11 to 2014-7-22
#Does each rating come with a review?

video_game.isnull().sum()

def day_month_year(dataset):
    day_column = []
    month_column =[]
    year_column=[]
    for date in video_game['reviewTime']:
        print(date)
        date_split_string = date.split()
        month = date_split_string[0]
        month = int(month)
        day = date_split_string[1]
        day = day.replace(',','')
        day = int(day)
        year = date_split_string[2]
        year = int(year)
        day_column.append(day)
        month_column.append(month)
        year_column.append(year)
    dataset['day'] = day_column
    dataset['month'] = month_column
    dataset['year'] = year_column
    return dataset

video_game = day_month_year(video_game)

#select products with certain average rating and certain number of reviews

def rating_count_average(dataset, count_lower, count_upper):
    data_groupby_productID = dataset.groupby(by = 'asin')
    rating_mean_df=data_groupby_productID[['overall']].mean()
    rating_mean_df = rating_mean_df.reset_index()
    rating_count_df = data_groupby_productID[['overall']].count()
    rating_count_df = rating_count_df.reset_index()
    rating_count_average_df = rating_mean_df.merge(rating_count_df, how = 'inner', on = 'asin')
    rating_count_average_df.columns = ['productID', 'meanRating','countRating']
    mask = (rating_count_average_df['countRating'] >= count_lower) & (rating_count_average_df['countRating'] <= count_upper)
    df_select= rating_count_average_df[mask]
    return(df_select)
    
data_select = rating_count_average (video_game, 0, 1000)

# choose a product with 7 ratings
one_product = video_game[video_game['asin']== 'B00HLT0YT0' ]

#rating count: 356; Average rating: 4.53; Product: B00005NZ1G
#rating count: 379; Average rating: 2.64; Product: B00178630A

#identify product features and orientation of comments about those features

#Step 1: part of speech tagging"

import nltk

one_review = video_game['reviewText'].iloc[4]

def sentence_tagging(one_review):
    results_all_nn = []
    results_all_adj = []
    for sentence in nltk.sent_tokenize(one_review):
        #print(sentence)
        text = nltk.word_tokenize(sentence)
        #print(text)
        tagged = nltk.pos_tag(text)
        results_nn_one_sentence = []
        results_adj_one_sentence = []
        for word, tag in tagged:  
            if 'NN' in tag:  #This includes NN, NNS, NNP. more deailed?
                results_nn_one_sentence.append(word)
            if 'JJ' in tag:  #This includes JJ, JJS,More detailed?
                results_adj_one_sentence.append(word)
        results_all_nn.append(results_nn_one_sentence)
        results_all_adj.append(results_adj_one_sentence)
    return results_all_nn, results_all_adj

results_all_nn, results_all_adj = sentence_tagging(one_review)

#Step 2: Find features that are talked about by many people using association rule mining
#Find all frequent itemsets. One customer may say many things unrelated to product feature
#But when they talk about features, the words they use converge
#80% of features talked about by one customer may also be talked about by another
#Support: percent of the dataset that contains the itemset
#Confidence: percent of item 1 mentioned by one cutermer also mentioned by another 
#install the package first using !pip install mlxtend
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import OnehotTransactions
#from mlxtend.frequent_patterns import association_rules
oht = OnehotTransactions()
oht_ary = oht.fit(results_all_nn).transform(results_all_nn)
df_apriori = pd.DataFrame(oht_ary, columns = oht.columns_)
frequent_itemsets = apriori(df_apriori, min_support = 0.06,use_colnames = True)
#frequent_itemsets ['length'] = frequent_itemsets ['itemsets'].apply(lambda x: len(x))
#rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

def frequent_features_of_one_product2(one_product, frequency_threshold):
    one_product_all_nn = []
    for one_review in one_product['reviewText']:
        results_all_nn, results_all_adj = sentence_tagging(one_review)
        one_product_all_nn += results_all_nn
    #return one_product_all_nn
    oht = OnehotTransactions()
    oht_ary = oht.fit(one_product_all_nn).transform(one_product_all_nn)
    df_apriori = pd.DataFrame(oht_ary, columns = oht.columns_)
    print('wait for apriori function...')
    frequent_itemsets = apriori(df_apriori, min_support = frequency_threshold/len(df_apriori), use_colnames = True)
    print('finished apriori function...')
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frequent_itemsets = frequent_itemsets[frequent_itemsets.length==2]
    frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending =False)
    
    return frequent_itemsets

def process_frequent_itemsets(frequent_itemsets):
    sets = set()
    for item in frequent_itemsets.itemsets:
        sets.update(item)
    return sets
    
from textblob import TextBlob
"""
sentiment 2, use Textblob, good.

>>> testimonial = TextBlob("Textblob is n")
>>> testimonial.sentiment
Sentiment(polarity=0.39166666666666666, subjectivity=0.4357142857142857)
>>> testimonial.sentiment.polarity
"""

def get_features_summary(one_product, frequency_threshold):
    frequent_features = frequent_features_of_one_product2(one_product, frequency_threshold)
    feature_sets = process_frequent_itemsets(frequent_features)
    #initialize result
    result = {}
    for feature in feature_sets:
        result[feature] = []
    for index, row in one_product.iterrows():
        one_review = row['reviewText']
        for sentence in nltk.sent_tokenize(one_review):
            #print(sentence)
            sentence_list = nltk.word_tokenize(sentence)
            #print(text)
            for word in sentence_list:
                if word in feature_sets:
                    result[word].append(sentence)
    # not the result is good, get sentiment for each sentence
    return result
    
def get_features_summary_positivity(result):
    df = pd.DataFrame(columns=['positive_review','positive_score','negative_review','negative_score'])
    df = df.astype('object')
    for feature in result.keys():
        df.at[feature, 'positive_review'] = list()
        df.at[feature, 'positive_score']  = list()
        df.at[feature, 'negative_review'] = list()
        df.at[feature, 'negative_score']  = list()
    
    for feature, sentences in result.items():
        
        for sentence in sentences:
            sentence_sentiment = TextBlob(sentence)
            sentiment = sentence_sentiment.sentiment.polarity
            if sentiment > 0: # we can change threshold to 0.1 or -0.1
                df.loc[feature, 'positive_review'].append(sentence)
                df.loc[feature, 'positive_score'].append(sentiment)
            elif sentiment < 0:
                df.loc[feature, 'negative_review'].append(sentence)
                df.loc[feature, 'negative_score'].append(sentiment)
            #print(sentence, sentiment)
    
    df['positive_count'] = [len(x) for x in df.positive_review]
    df['negative_count'] = [len(x) for x in df.negative_review]
    return df

#rating count: 356; Average rating: 4.53; Product: B00005NZ1G
#rating count: 379; Average rating: 2.64; Product: B00178630A
#one_product = video_game[video_game['asin']== '0700099867' ]

product_high = video_game[video_game['asin']== 'B00005NZ1G' ]  
feature_sentence_df_high = get_features_summary(product_high, 10)    
positivity_df_high = get_features_summary_positivity(feature_sentence_df_high) 
    
product_low = video_game[video_game['asin']== 'B00178630A' ]
feature_sentence_df_low = get_features_summary(product_low, 10)  
positivity_df_low = get_features_summary_positivity(feature_sentence_df_low)

#visialize the apriori results
#check out this website: http://pbpython.com/market-basket-analysis.html
#http://aimotion.blogspot.com/2013/01/machine-learning-and-data-mining.html
'''
import seaborn as sns
sns.heatmap(frequent_itemsets)
        


from apriori_new import apriori
itemsets = apriori(results_all_nn, 1.5/len(results_all_nn), 0)  #p support can be dynamic

def frequent_features_of_one_product(one_product):
    one_product_all_nn = []
    for one_review in one_product['reviewText']:
        results_all_nn, results_all_adj = sentence_tagging(one_review)
        one_product_all_nn += results_all_nn
    #return one_product_all_nn
    frequent_features = apriori(results_all_nn, 1.5/len(one_product_all_nn), 0) #min_support does not work
    return frequent_features

frequent_features = frequent_features_of_one_product(one_product)

#sort the features by support values
sorted(frequent_features, key=lambda feature: feature[1])

#read the file into python
#note: always specify the encoding method!
#open this short version for give a try
#myfile=open('98-0 copy.txt','r', encoding = 'utf-8') 
'''
one_review = video_game['reviewText'].iloc[3]
one_review_volumn = len(one_review.split())  #count number of words (not including punctuations)

#get sentiment of a word
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('/tmp/vectors.bin.gz', binary=True)


# Load Google's pre-trained Word2Vec model.
"""
sentiment 1, use word2vec. not good.
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format(r'C:\kaggle\Kaggle_HomeDepot-master\Kaggle_HomeDepot-master\Data\word2vec\GoogleNews-vectors-negative300.bin', binary=True)
model.similarity('good', 'fast')
"""


#draw word clouds for reviews
from wordcloud import WordCloud, STOPWORDS
#combine all the reviews for one product into one giant text
def combine_review(dataset, productID):
    product_data = dataset[dataset['asin']== productID]
    all_reviews = ''
    for index, row in product_data.iterrows():
        all_reviews = all_reviews + product_data['reviewText'].loc[index]
    return all_reviews
    
all_reviews = combine_review(video_game,'B00178630A')

wordcloud = WordCloud(width=960, height=960, margin=1,stopwords=STOPWORDS).generate(all_reviews)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

#go through each word and see if there are periods or commas

#if so, remove those from that word 
    
punctuation_set=[".","!",":",";","?","“","”",","]
    
def remove_punctuation(word_old, punctuation_set):
    word_small = word_old.lower()
    word_new =''
    for letter in word_small:
        if letter in punctuation_set:
            letter=letter.replace(letter,'')
        else:
            letter=letter
        word_new=word_new+letter
    return word_new

#count the frequency of words in one person's review
#return a dictionary with words being the keys and counts being the values
#chop the text in the stopwords file into separate lines, and then return a list of stopwords
def word_frequency(one_review, punctuation_set):
    stopwordfile=open('data\stopwords', encoding = 'utf-8')
    stopwordlist=stopwordfile.read().splitlines()
    all_words = one_review.split()
    wordfreq_dict = {}
    for word in all_words:
        word = remove_punctuation(word, punctuation_set)
        if word not in stopwordlist:
             if word in wordfreq_dict.keys():
                 wordfreq_dict[word] +=1
             else:
                 wordfreq_dict[word] = 1
    #wordfreq_dict = sorted(wordfreq_dict.items(), key=operator.itemgetter(1))
    return wordfreq_dict
    
rating_videogame = pd.read_csv(r'data\ratings_Video_Games.csv')

shoes_ratings = pd.read_csv(r'data\ratings_Clothing_Shoes_and_Jewelry.csv', header= None)
shoes_ratings.columns = ['reviewer_id','asin','rating', 'date']
shoes_ratings['date'] = [datetime.datetime.fromtimestamp(int(x)) for x in shoes_ratings['date']]
shoes_ratings['date'] = [pd.Timestamp(x) for x in shoes_ratings['date']]
                  
                  
meta_data = getDF(r'data\metadata.json.gz', None, 0) 
meta_data.to_pickle(r'data\meta_data')
                  
#plot crocs rating 90 days rolling mean.
crocs_product = meta_data[meta_data.brand == 'Crocs']
crocs_rating = pd.merge(left=shoes_ratings, right = crocs_product, how = 'inner', on = ['asin'])
crocs_rating = crocs_rating.sort_values(by ='date')
crocs_rating.set_index('date', inplace = True)
crocs_rating['rating_rollling_mean'] = crocs_rating['rating'].rolling(window = '90d').mean()
crocs_rating['rating_rollling_mean'][crocs_rating.index>pd.Timestamp(datetime.date(2010, 1, 1))].plot(grid = True, legend = True, figsize = (20,10))
plt.savefig('crocs_rating_rolling_mean.jpg')
#crocs_rating = pd.merge(left = crocs_rating, right = crox_stock, how = 'outer', left_index = True, right_index = True)
                  
#plot crocs stock price
crox_stock = pd.read_csv(r'data\CROX.csv', parse_dates = [0])
crox_stock['stock_price'] = crox_stock['Adj Close']
crox_stock['date'] = crox_stock['Date']
crox_stock = crox_stock[['date','stock_price']]
crox_stock.set_index('date', inplace = True)
crox_stock['stock_price'][(crox_stock.index>pd.Timestamp(datetime.date(2010, 1, 1))) & (crox_stock.index<pd.Timestamp(datetime.date(2015, 1, 1)))].plot(grid = True, legend = True, figsize = (20,10))
                  

                  
                  
                  