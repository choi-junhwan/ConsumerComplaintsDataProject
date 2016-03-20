import requests
import numpy as np
from flask import Flask, render_template, request, redirect
import ConsumerComplaints_Analy as CCAnaly
import pandas as pd

app = Flask(__name__)
app.vars={}

@app.route('/')
def main():
  return redirect('/index')

@app.route('/index', methods=['GET','POST'])
def index():
  if request.method == "GET":
    return render_template('index.html')
  else:
    app.vars['key'] = request.form['key']
    df       = pd.read_csv('Consumer_Complaints_short.csv', header=0)
    df       = CCAnaly.data_wrangling(df)    

    if app.vars['key'] == 'Top Product':
      plotPng = CCAnaly.top_complained_products(df[['Date received','Product','Response','Disputed']])
    elif app.vars['key'] == 'Top Issues':
      plotPng = CCAnaly.top_complained_issues(df[['Date received','Issue','Response','Disputed']])
    elif app.vars['key'] == 'Issue Analysis':
      plotPng = CCAnaly.text_analysis(df[['Issue']])
    else:
      pass

    return render_template('plot.html', name=app.vars['key'], plotPng=plotPng)
    
    
if __name__ == '__main__':
  #app.run(debug=True)
  app.run(host='0.0.0.0')
