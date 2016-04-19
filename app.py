import numpy as np
from flask import Flask, render_template, request, redirect
import Complaints_Map as cmap
import json

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
      return render_template('first.html', plotPng1='bar_top_product.png', plotPng2='bar_product_tevolve.png', plotPng3='ComplainCount_ratio.png')
    elif app.vars['key'] == 'Map':
      script_resp, div_resp, script_disp, div_disp, script_corr, div_corr = cmap.map_view()
      return render_template('graph.html', script1=script_resp, div1=div_resp, script2=script_disp, div2=div_disp, script3=script_corr, div3=div_corr)
    elif app.vars['key'] == 'Text':
      with open("static/figures/issue_lda.json", "r") as issue_file:
        d_issue = json.load(issue_file)
      with open("static/figures/narrative_lda.json", "r") as narr_file:
        d_narr = json.load(narr_file)
      return render_template('textanalysis.html', plotPng ='ComplainCount_WC.png', data_issue = json.dumps(d_issue), data_narrative = json.dumps(d_narr))
    else:
      return render_template('error.html')

    
    
if __name__ == '__main__':
  #app.run(debug=True)
  app.run(host='0.0.0.0')
