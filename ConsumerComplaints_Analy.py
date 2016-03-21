import pandas as pd
import numpy as np
#import csv as csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
#import pylab as pl
#from wordcloud import WordCloud
#from scipy.stats.stats import pearsonr
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
#from gensim import corpora, models
#import gensim
import tempfile



def data_wrangling(df):
    # Data overview: contents and size
    #print df.dtypes
    #print len(df.index)

    # Data cleaning
    df['Date received'] = pd.to_datetime(df['Date received'])
    df['Response'] = df['Timely response?'].map({'No' : 0, 'Yes' : 1}).astype(int)
    df=df.fillna({'Consumer disputed?' : 'No'})
    df['Disputed'] = df['Consumer disputed?'].map({'No' : 0, 'Yes' : 1}).astype(int)
    df = df[['Date received','Issue','Product','Response','Disputed']]

    return df

#def Pearson_Correlation(df):
    """
    Pearson's Correlation between Response and Disputed rate.
    """
"""
    corr   = df[['Response','Disputed']]
    test_x = corr['Response'].values
    test_y = corr['Disputed'].values
    return pearsonr(test_x, test_y)
"""

def top_complained_products(df):
    """
    Complained Product
    """
    df_sub = df[['Product']]
    df_sub.insert(0, 'count', 1)
    df_sub_grp = df_sub.groupby(['Product']).sum().sort_index(by='count', ascending=False).ix[0:5]
    #print '\n Top 5 most frequently complained products'
    #print df_sub_grp

    df_sub = df[df['Response'] == 0]
    df_sub = df_sub[['Product']]
    df_sub.insert(0, 'count', 1)
    df_sub_grp = df_sub.groupby(['Product']).sum().sort_index(by='count', ascending=False).ix[0:5]
    #print '\n Top 5 most frequently complained products which are not Timely responded'
    #print df_sub_grp

    df_sub = df[df['Disputed'] == 1]
    df_sub = df_sub[['Product']]
    df_sub.insert(0, 'count', 1)
    df_sub_grp = df_sub.groupby(['Product']).sum().sort_index(by='count', ascending=False).ix[0:5]
    #print '\n Top 5 most frequently complained products which are Consumer disputed'
    #print df_sub_grp

    # Annual list viz
    df_sub = df[['Product']]
    df_sub.insert(0, 'count', 1)

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
    #xticks = np.arange(2011,2016)
    tdf_grp.plot(kind='bar', stacked=True, title="Top 3 most complained Products", rot=10)

    fig = plt.gcf()
    #fig.savefig('bar.png')
    #pl.clf()
    #return 0;

    f = tempfile.NamedTemporaryFile(dir='static/temp',suffix='.png',delete=False)
    fig.savefig(f)
    f.close() # close the file
    # get the file's name
    # (the template will need that)
    plotPng = f.name.split('/')[-1]
    
    return plotPng
                        


def top_complained_issues(df):
    """
    Complained Issues
    """
    df_sub = df[['Issue']]
    df_sub.insert(0, 'count', 1)
    df_sub_grp = df_sub.groupby(['Issue']).sum().sort_index(by='count', ascending=False).ix[0:10]
    #print '\n Top 10 most frequently complained issues'
    #print df_sub_grp

    df_sub = df[df['Response'] == 0]
    df_sub = df_sub[['Issue']]
    df_sub.insert(0, 'count', 1)
    df_sub_grp = df_sub.groupby(['Issue']).sum().sort_index(by='count', ascending=False).ix[0:10]
    #print '\n Top 10 most frequent complained issues which are not Timely responded'
    #print df_sub_grp
    
    df_sub = df[df['Disputed'] == 1]
    df_sub = df_sub[['Issue']]
    df_sub.insert(0, 'count', 1)
    df_sub_grp = df_sub.groupby(['Issue']).sum().sort_index(by='count', ascending=False).ix[0:10]
    #print '\n Top 10 most frequent complained issues which are Consumer disputed'
    #print df_sub_grp
    
    # More analysis with top issues
    df_sub = df[['Issue']]
    df_sub.insert(0, 'count', 1)
    
    Issue_List=[]
    for i in range(0,25):
        Issue_List.append(df_sub.groupby(['Issue']).sum().sort_index(by='count', ascending=False).ix[i].name)

    # time series viz 
    DateList=[]
    CountAll=[]
    Count0  =[]
    Count1  =[]
    Count2  =[]
    CountResp0 = []
    CountDisp0 = []
    CountResp1 = []
    CountDisp1 = []
    CountResp2 = []
    CountDisp2 = []
    for i in range(0,len(df.index)):
        DateList.append(df.ix[i]['Date received'])
        CountAll.append(1)
        if Issue_List[0] == df.ix[i]['Issue']:
            Count0.append(1)
        else:
            Count0.append(0)
        if Issue_List[1] == df.ix[i]['Issue']:
            Count1.append(1)
        else:
            Count1.append(0)
        if Issue_List[2] == df.ix[i]['Issue']:
            Count2.append(1)
        else:
            Count2.append(0)

        if Issue_List[0] == df.ix[i]['Issue'] and df.ix[i]['Response'] == 0:
            CountResp0.append(1)
        else:
            CountResp0.append(0)
        if Issue_List[1] == df.ix[i]['Issue'] and df.ix[i]['Response'] == 0:
            CountResp1.append(1)
        else:
            CountResp1.append(0)
        if Issue_List[2] == df.ix[i]['Issue'] and df.ix[i]['Response'] == 0:
            CountResp2.append(1)
        else:
            CountResp2.append(0)

        if Issue_List[0] == df.ix[i]['Issue'] and df.ix[i]['Disputed'] == 1:
            CountDisp0.append(1)
        else:
            CountDisp0.append(0)
        if Issue_List[1] == df.ix[i]['Issue'] and df.ix[i]['Disputed'] == 1:
            CountDisp1.append(1)
        else:
            CountDisp1.append(0)
        if Issue_List[2] == df.ix[i]['Issue'] and df.ix[i]['Disputed'] == 1:
            CountDisp2.append(1)
        else:
            CountDisp2.append(0)

    ts = pd.Series(CountAll, index=DateList)
    ts_grp = ts.groupby(level=0).sum().resample('A', how='sum')
    
    ts0 = pd.Series(Count0, index=DateList)
    ts0_grp = ts0.groupby(level=0).sum().resample('A', how='sum')
    
    ts1 = pd.Series(Count1, index=DateList)
    ts1_grp = ts1.groupby(level=0).sum().resample('A', how='sum')
        
    ts2 = pd.Series(Count2, index=DateList)
    ts2_grp = ts2.groupby(level=0).sum().resample('A', how='sum')
    
    tr0 = pd.Series(CountResp0, index=DateList)
    tr0_grp = tr0.groupby(level=0).sum().resample('A', how='sum')
        
    tr1 = pd.Series(CountResp1, index=DateList)
    tr1_grp = tr1.groupby(level=0).sum().resample('A', how='sum')
    
    tr2 = pd.Series(CountResp2, index=DateList)
    tr2_grp = tr2.groupby(level=0).sum().resample('A', how='sum')
    
    td0 = pd.Series(CountDisp0, index=DateList)
    td0_grp = td0.groupby(level=0).sum().resample('A', how='sum')
    
    td1 = pd.Series(CountDisp1, index=DateList)
    td1_grp = td1.groupby(level=0).sum().resample('A', how='sum')
        
    td2 = pd.Series(CountDisp2, index=DateList)
    td2_grp = td2.groupby(level=0).sum().resample('A', how='sum')
        

    fig = plt.figure(figsize=(23,8))
    fig1 = fig.add_subplot(1,3,1)
    fig2 = fig.add_subplot(1,3,2)
    fig3 = fig.add_subplot(1,3,3)

    fig1.plot(ts_grp.index,ts_grp,label='total')
    fig1.bar(ts0_grp.index,ts0_grp,label='%s' % Issue_List[0],color="red",width=100)
    fig1.bar(ts1_grp.index,ts1_grp,label='%s' % Issue_List[1],color="blue",bottom=ts0_grp,width=100)
    fig1.bar(ts2_grp.index,ts2_grp,label='%s' % Issue_List[2],color="green",bottom=(ts1_grp+ts0_grp),width=100)
    fig1.legend(loc=2, prop={'size':10})
    fig1.set_title('Number of Complaints per year', fontsize=20)
    fig1.set_ylabel('Number of Complaints', fontsize=15)
    fig1.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    fig1.set_xlabel('Date',fontsize=15)
        

    fig2.plot(ts0_grp.index,tr0_grp/ts0_grp,label='%s' % Issue_List[0],color="red")
    fig2.plot(ts1_grp.index,tr1_grp/ts1_grp,label='%s' % Issue_List[1],color="blue")
    fig2.plot(ts2_grp.index,tr2_grp/ts2_grp,label='%s' % Issue_List[2],color="green")
    fig2.legend(loc=0, prop={'size':10})
    fig2.set_title('NOT Timely responded rate',fontsize=20)
    fig2.set_ylabel('Rate',fontsize=15)
    fig2.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    fig2.set_xlabel('Date',fontsize=15)
    plt.xticks(rotation=30)
        
    fig3.plot(ts0_grp.index,td0_grp/ts0_grp,label='%s' % Issue_List[0],color="red")
    fig3.plot(ts1_grp.index,td1_grp/ts1_grp,label='%s' % Issue_List[1],color="blue")
    fig3.plot(ts2_grp.index,td2_grp/ts2_grp,label='%s' % Issue_List[2],color="green")
    fig3.legend(loc=0, prop={'size':10})
    fig3.set_title('Consumer disputed rate',fontsize=20)
    fig3.set_ylabel('Rate',fontsize=15)
    fig3.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    fig3.set_xlabel('Date',fontsize=15)
        
    matplotlib.pyplot.sca(fig1)
    plt.xticks(rotation=20)
    matplotlib.pyplot.sca(fig2)
    plt.xticks(rotation=20)
    matplotlib.pyplot.sca(fig3)
    plt.xticks(rotation=20)

    f = tempfile.NamedTemporaryFile(dir='static/temp', suffix='.png', delete=False)
    plt.savefig(f)
    f.close()
    plotPng = f.name.split('/')[-1]
    return plotPng


#def text_analysis(df):
    """
    Text analysis for top issuess with LDA
    Initialize text 
    """
"""
    #Complained Issues
    # More analysis with top issues
    df_sub = df[['Issue']]
    df_sub.insert(0, 'count', 1)
    
    Issue_List=[]
    for i in range(0,25):
        Issue_List.append(df_sub.groupby(['Issue']).sum().sort_index(by='count', ascending=False).ix[i].name)

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
    LDAText =  ldamodel.print_topics(num_topics=5, num_words=3)
    print "\n Topic analysis result for top 25 issues with LDA"
    print(LDAText)

    wordcloud = WordCloud().generate(text_view)
    fig = plt.figure(figsize=(15,8))
    fig1 = fig.add_subplot(1,2,1)
    fig1.set_title('5 Issue Topics from LDA Topic Modeling',fontdict={'fontsize':25})
    fig1.text(0.5, 0.9, LDAText[0], size=15, rotation=0.,
              ha="center", va="center",
              bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))
    fig1.text(0.5, 0.75, LDAText[1], size=15, rotation=0.,
              ha="center", va="center",
              bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))
    fig1.text(0.5, 0.6, LDAText[2], size=15, rotation=0.,
              ha="center", va="center",
              bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))
    fig1.text(0.5, 0.45, LDAText[3], size=15, rotation=0.,
              ha="center", va="center",
              bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))
    fig1.text(0.5, 0.3, LDAText[4], size=15, rotation=0.,
              ha="center", va="center",
              bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))
    fig1.axis('off')

    fig2 = fig.add_subplot(1,2,2)
    fig2.set_title("Top issue words", fontdict={'fontsize':25})
    fig2.imshow(wordcloud)
    fig2.axis("off")

    f = tempfile.NamedTemporaryFile(dir='static/temp', suffix='.png', delete=False)
    plt.savefig(f)
    f.close()
    plotPng = f.name.split('/')[-1]
    return plotPng
"""

if __name__=="__main__":
    df = pd.read_csv('Consumer_Complaints_short.csv', header=0)
    df = data_wrangling(df)

    ##### Pearson Correaltion betwen two features
    #print Pearson_Correlation(df[['Response','Disputed']])    
    
    #####
    top_complained_products(df[['Date received','Product','Response','Disputed']])

    ####
    top_complained_issues(df[['Date received','Issue','Response','Disputed']])

    ####
    text_analysis(df[['Issue']])
