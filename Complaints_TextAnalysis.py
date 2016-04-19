import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
from matplotlib.dates import DateFormatter
import pylab as pl
from wordcloud import WordCloud
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
from gensim import corpora, models
import gensim
import tempfile
import pyLDAvis.gensim as gensimvis
import pyLDAvis



def data_wrangling(df):
    df['Date received'] = pd.to_datetime(df['Date received'])
    df['narrative'] = df['Consumer complaint narrative']
    df = df[['Date received','Issue','narrative']]

    return df
                        

def issue_analysis(df):
    df_sub = df[['Issue']]
    df_sub.insert(0, 'count', 1)

    Issue_List=[]
    for i in range(0,50):
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

    wordcloud = WordCloud().generate(text_view)
    fig = plt.figure(figsize=(8,6))
    fig1 = fig.add_subplot(1,1,1)
    fig1.set_title("Top issued words", fontdict={'fontsize':25})
    fig1.imshow(wordcloud)
    fig1.axis("off")
    plt.savefig('ComplainCount_WC.png')
    
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=25, id2word = dictionary)
    LDAText =  ldamodel.print_topics(num_topics=5, num_words=3)
    #print "\n Topic analysis result for top 25 issues with LDA"
    #print(LDAText)
       
    vis_data = gensimvis.prepare(ldamodel, corpus, dictionary)
    #pyLDAvis.show(vis_data)
    pyLDAvis.save_html(vis_data, "issue_lda.html")
    pyLDAvis.save_json(vis_data, "issue_lda.json")

    return 0


def narrative_analysis(df):
    tokenizer = RegexpTokenizer(r'[A-Za-z0-9\']+')    # set tokenize Reg
    en_stop = get_stop_words('en')         # create English stop words list
    p_stemmer = PorterStemmer()            # Create p_stemmer of class PorterStemmer
    texts = []                             # list for tokenized documents in loop

    for index in range(0,len(df.index)):
        if str(df['narrative'].ix[index]) != 'nan':
            intext = df['narrative'].ix[index]
            intext = re.sub(r"X+", "", intext)
            raw = intext.lower()
            tokens = tokenizer.tokenize(raw)
       
            # remove stop words from tokens
            stopped_tokens = [i for i in tokens if not i in en_stop]
        
            # stem tokens and add them to list
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            texts.append(stemmed_tokens)

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=25, id2word = dictionary)
    LDAText =  ldamodel.print_topics(num_topics=5, num_words=3)
    #print "\n Topic analysis result for top 25 issues with LDA"
    #print(LDAText)
       
    vis_data = gensimvis.prepare(ldamodel, corpus, dictionary)
    #pyLDAvis.show(vis_data)
    pyLDAvis.save_html(vis_data, "narrative_lda.html")
    pyLDAvis.save_json(vis_data, "narrative_lda.json")

    return 0
            
            

if __name__=="__main__":
    
    # https://data.consumerfinance.gov/resource/jhzv-w97w.json
    df = pd.read_csv('../Data/Consumer_Complaints.csv', header=0)
    df = data_wrangling(df)
    issue_analysis(df)

    #df = pd.read_csv('../Data/Consumer_Complaints_2015.csv', header=0)
    #df = data_wrangling(df)
    narrative_analysis(df)

