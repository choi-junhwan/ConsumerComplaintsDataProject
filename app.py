import requests
import numpy as np
from flask import Flask, render_template, request, redirect
import myquandl as myquandl

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
    app.vars['stock'] = request.form['stock']    
    app.vars['key'] = request.form['key']
    success, script, div = myquandl.stock_plot(app.vars['stock'], app.vars['key'])
    if success:
      return render_template('graph.html', name=app.vars['stock'], script=script, div=div)
    else:
      return render_template('error.html', name=app.vars['stock'])

    
if __name__ == '__main__':
  #app.run(debug=True)
  app.run(host='0.0.0.0')
