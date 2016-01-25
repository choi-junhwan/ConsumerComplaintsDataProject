import pandas as pd
import numpy as np
import csv as csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl
from wordcloud import WordCloud
from scipy.stats.stats import pearsonr
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
#%matplotlib inline


pl.ion()
pl.clf()

# Consumer_Complaints.csv from http://catalog.data.gov/dataset/consumer-complaint-database
#df = pd.read_csv('Consumer_Complaints.csv', header=0)
#df = df.ix[::25]
#df.to_csv('Consumer_Complaints_short.csv', index=False)
df = pd.read_csv('Consumer_Complaints_short.csv', header=0)

# Data overview: contents and size
print df.dtypes
print len(df.index)

# Data cleaning
df['Date received'] = pd.to_datetime(df['Date received'])
df['Response'] = df['Timely response?'].map({'No' : 0, 'Yes' : 1}).astype(int)
df=df.fillna({'Consumer disputed?' : 'No'})
df['Disputed'] = df['Consumer disputed?'].map({'No' : 0, 'Yes' : 1}).astype(int)
df = df[['Date received','Issue','Product','Response','Disputed']]


# Compute Pearson's Correlation between Response and Disputed rate.
corr   = df[['Response','Disputed']]
test_x = corr['Response'].values
test_y = corr['Disputed'].values
print "pearson correlation"
print pearsonr(test_x, test_y)


####################
# Complained Product
df_sub = df[['Product']]
df_sub['count'] = 1
df_sub_grp = df_sub.groupby(['Product']).sum().sort_index(by='count', ascending=False).ix[0:5]
print '\n Top 5 most frequently complained products'
print df_sub_grp

df_sub = df[df['Response'] == 0]
df_sub = df_sub[['Product']]
df_sub['count'] = 1
df_sub_grp = df_sub.groupby(['Product']).sum().sort_index(by='count', ascending=False).ix[0:5]
print '\n Top 5 most frequently complained products which are not Timely responded'
print df_sub_grp

df_sub = df[df['Disputed'] == 1]
df_sub = df_sub[['Product']]
df_sub['count'] = 1
df_sub_grp = df_sub.groupby(['Product']).sum().sort_index(by='count', ascending=False).ix[0:5]
print '\n Top 5 most frequently complained products which are Consumer disputed'
print df_sub_grp

# Annual list viz
df_sub = df[['Product']]
df_sub['count'] = 1

Product_List=[]
for i in range(0,3):
    Product_List.append(df_sub.groupby(['Product']).sum().sort_index(by='count', ascending=False).ix[i].name)


date_list=[]
count_0  =[]
count_1  =[]
count_2  =[]
for i in range(0,len(df.index)):
    date_list.append(df.ix[i]['Date received'])
    if Product_List[0] == df.ix[i]['Product']:
        count_0.append(1)
    else:
        count_0.append(0)
    if Product_List[1] == df.ix[i]['Product']:
        count_1.append(1)
    else:
        count_1.append(0)
    if Product_List[2] == df.ix[i]['Product']:
        count_2.append(1)
    else:
        count_2.append(0)

tdf = pd.DataFrame({ Product_List[0] : count_0,
                     Product_List[1] : count_1,
                     Product_List[2] : count_2,},
                   index=date_list)

tdf_grp = tdf.groupby(level=0).sum().resample('A', how='sum')
xticks = np.arange(2011,2016)
print tdf_grp
tdf_grp.plot(kind='bar', stacked=True, title="Top 3 most complained Products", rot=10)

fig = plt.gcf()
fig.savefig('bar.png')
pl.clf()


###################
# Complained Issues
df_sub = df[['Issue']]
df_sub['count'] = 1
df_sub_grp = df_sub.groupby(['Issue']).sum().sort_index(by='count', ascending=False).ix[0:10]
print '\n Top 10 most frequently complained issues'
print df_sub_grp

df_sub = df[df['Response'] == 0]
df_sub = df_sub[['Issue']]
df_sub['count'] = 1
df_sub_grp = df_sub.groupby(['Issue']).sum().sort_index(by='count', ascending=False).ix[0:10]
print '\n Top 10 most frequent complained issues which are not Timely responded'
print df_sub_grp

df_sub = df[df['Disputed'] == 1]
df_sub = df_sub[['Issue']]
df_sub['count'] = 1
df_sub_grp = df_sub.groupby(['Issue']).sum().sort_index(by='count', ascending=False).ix[0:10]
print '\n Top 10 most frequent complained issues which are Consumer disputed'
print df_sub_grp

# More analysis with top issues
df_sub = df[['Issue']]
df_sub['count'] = 1

Issue_List=[]
for i in range(0,25):
    Issue_List.append(df_sub.groupby(['Issue']).sum().sort_index(by='count', ascending=False).ix[i].name)

# time series viz 
date_list=[]
count_all=[]
count_0  =[]
count_1  =[]
for i in range(0,len(df.index)):
    date_list.append(df.ix[i]['Date received'])
    count_all.append(1)
    if Issue_List[0] == df.ix[i]['Issue']:
        count_0.append(1)
    else:
        count_0.append(0)
    if Issue_List[1] == df.ix[i]['Issue']:
        count_1.append(1)
    else:
        count_1.append(0)

ts = pd.Series(count_all, index=date_list)
ts_grp = ts.groupby(level=0).sum().resample('M', how='sum')

ts0 = pd.Series(count_0, index=date_list)
ts0_grp = ts0.groupby(level=0).sum().resample('M', how='sum')

ts1 = pd.Series(count_1, index=date_list)
ts1_grp = ts1.groupby(level=0).sum().resample('M', how='sum')

plt.plot(ts_grp.index,ts_grp,label='total')
plt.plot(ts0_grp.index,ts0_grp,label='1st complaint')
plt.plot(ts1_grp.index,ts1_grp,label='2st complaint')

plt.legend(loc=0)
plt.ylabel('Number of Complaints in a month')
plt.xlabel('Date')
plt.savefig('ComplainCount')
pl.clf()

# Text analysis for top issuess with LDA
# Initialize text 
tokenizer = RegexpTokenizer(r'[A-Za-z0-9\']+')    # set tokenize Reg 
en_stop = get_stop_words('en')         # create English stop words list
p_stemmer = PorterStemmer()            # Create p_stemmer of class PorterStemmer
texts = []                             # list for tokenized documents in loop
text_view = ''

# loop through document list
for i in Issue_List:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens and add them to list
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    texts.append(stemmed_tokens)

    #print ' '.join(stemmed_tokens)
    text_view += ' '.join(stemmed_tokens)
    text_view += ' '

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word = dictionary, passes=20)
print "\n Topic analysis result for top 25 issues with LDA"
print(ldamodel.print_topics(num_topics=5, num_words=3))

#wordcloud = WordCloud(max_font_size=40).generate(text_view)
wordcloud = WordCloud().generate(text_view)
plt.figure()
plt.title("Top issue words")
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('ComplainCount_WC')
