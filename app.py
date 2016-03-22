import numpy as np
from flask import Flask, render_template, request, redirect
import Complaints_Map as cmap

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

    if app.vars['key'] == 'Firstlook':
      return render_template('first.html', plotPng1='bar_product.png', plotPng2='ComplainCount.png')
    elif app.vars['key'] == 'Map':
      script, div = cmap.map_view()
      return render_template('graph.html', script=script, div=div)
    elif app.vars['key'] == 'Text':
      return render_template('textanalysis.html', plotPng ='ComplainCount_WC.png')
    else:
      return render_template('error.html')

    
    
if __name__ == '__main__':
  app.run(debug=True)
  #app.run(host='0.0.0.0')
