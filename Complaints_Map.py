import csv  as csv
import pandas as pd
import numpy as np
import math  as math
from bokeh.plotting import figure, show, save, output_file, ColumnDataSource
from bokeh.embed import components
from bokeh.sampledata.us_states import data as states
from bokeh.models import HoverTool

def data_wrangling(df):
    #Data overview: contents and size
    #print df.dtypes
    #print len(df.index)

    # Data cleaning
    df['Date received'] = pd.to_datetime(df['Date received'])
    df['Response'] = df['Timely response?'].map({'No' : 0, 'Yes' : 1}).astype(int)
    df=df.fillna({'Consumer disputed?' : 'No'})
    df['Disputed'] = df['Consumer disputed?'].map({'No' : 0, 'Yes' : 1}).astype(int)
    df = df[['Date received','Issue','Product','Response','Disputed','State']]
    
    return df
                                    
def pvalue(rate, n, p):
    import scipy.stats
    k = rate*n
    if rate > p:
        return (1-scipy.stats.binom.cdf(k,n,p))
    else:
        return scipy.stats.binom.cdf(k,n,p)

def generate_map(df):
    """
    sort Complaint response and disputed rate for each state
    the result is also divided by yearly
    """
    df_sub = df[['State','Response','Disputed']]
    df_sub.insert(0, 'count', 1)
    df_sub_grp = df_sub.groupby(['State']).sum().sort_index(by='count', ascending=False)
    states = [ x for x in df['State'].unique() if str(x) != 'nan']

    n_count = np.zeros(len(states))
    n_resp  = np.zeros(len(states))
    n_disp  = np.zeros(len(states))
    for index in df_sub.index:
        if str(df_sub.ix[index]['State']) != 'nan':
            n_count[states.index(df_sub.ix[index]['State'])] += 1
            if df_sub.ix[index]['Response'] == 1:
                n_resp[states.index(df_sub.ix[index]['State'])] += 1
            if df_sub.ix[index]['Disputed'] == 1:
                n_disp[states.index(df_sub.ix[index]['State'])] += 1
        
    #print n_count, n_resp, n_disp
    States_DF = pd.DataFrame({'states': states,
                              'count': n_count,
                              'Response' : n_resp/n_count,
                              'Disputed' : n_disp/n_count})

    # total count
    total = States_DF['count'].sum()

    # resp_rate
    States_DF['resp_count'] = States_DF.apply(lambda row: row['Response']*row['count'], axis=1)
    total_resp = States_DF['resp_count'].sum()
    resp_rate = total_resp/total

    # disp_rate
    States_DF['disp_count'] = States_DF.apply(lambda row: row['Disputed']*row['count'], axis=1)
    total_disp = States_DF['disp_count'].sum()
    disp_rate = total_disp/total


        
    States_DF['resp_pvalue'] = States_DF.apply(lambda row: pvalue(row['Response'],row['count'],resp_rate), axis=1)
    States_DF['disp_pvalue'] = States_DF.apply(lambda row: pvalue(row['Disputed'],row['count'],disp_rate), axis=1)

    States_DF = States_DF.drop(States_DF.columns[[4, 5]], axis=1)

    States_DF.to_csv('Complaints_State.csv', index=False)    
    return 0


def map_view():
    try:
        del states["HI"]
        del states["AK"]
    except:
        pass
    
    EXCLUDED = ("ak", "hi", "pr", "gu", "vi", "mp", "as")
    colors = ["#F1EEF6", "#D4B9DA", "#C994C7", "#DF65B0", "#DD1C77", "#980043"]
    
    complaints = pd.read_csv('Complaints_State.csv', header=0, index_col=3)
    cc = []
    rr = []
    dd = []

    for state_id in states:
        if states[state_id] in EXCLUDED:
            continue
        try :
            cc.append(complaints.at[state_id,'count'])
            rr.append(complaints.at[state_id,'Response'])
            dd.append(complaints.at[state_id,'Disputed'])
        except:
            pass

    cmax = max(cc)
    rmax = max(rr)
    dmax = max(dd)
    cmin = min(cc)
    rmin = min(rr)
    dmin = min(dd)

    state_xs = [states[code]["lons"] for code in states]
    state_ys = [states[code]["lats"] for code in states]

    state_colors_count = []
    state_colors_resp = []
    state_colors_disp = []
    for state_id in states:
        if states[state_id] in EXCLUDED:
            continue
        try:
            cidx = int((complaints.at[state_id,'count']-cmin)*5/(cmax-cmin))
            ridx = int((complaints.at[state_id,'Response']-rmin)*5/(rmax-rmin))
            didx = int((complaints.at[state_id,'Disputed']-dmin)*5/(dmax-dmin))
            state_colors_count.append(colors[cidx])
            state_colors_resp.append(colors[ridx])
            state_colors_disp.append(colors[didx])
        except KeyError:
            state_colors_count.append("black")
            state_colors_resp.append("black")
            state_colors_disp.append("balck")

    """
    plot_count = figure(title="Statewise Complaints Counts", toolbar_location="left",
                  plot_width=1100, plot_height=700)
    plot_count.patches(state_xs, state_ys,
                 fill_color=state_colors_count, fill_alpha=0.7,
                 line_color="#884444", line_width=2, line_alpha=0.3)
    script_count, div_count = components(plot_count)
    """
    state_resp = [complaints.at[state_id,'Response'] for state_id in states]
    state_disp = [complaints.at[state_id,'Disputed'] for state_id in states]
    state_rp   = [complaints.at[state_id,'resp_pvalue'] for state_id in states]
    state_dp   = [complaints.at[state_id,'disp_pvalue'] for state_id in states]
    state_name = [states[state_id]["name"] for state_id in states]

    # responded rate
    source_resp = ColumnDataSource(
        data=dict(
            x = state_xs,
            y = state_ys,
            color=state_colors_resp,        
            name=state_name,
            rate=state_resp,
            pvalue=state_rp, 
        ))
    hover_resp = HoverTool(
        tooltips=[
            ("Name", "@name"),
            ("Rate", "@rate"),
            ("p-value", "@pvalue"),
        ]
    )
    plot_resp = figure(title="State-wise Complaints Respond Rate", tools=[hover_resp], toolbar_location="left",
                  plot_width=1100, plot_height=700)
    plot_resp.patches('x', 'y', source=source_resp,
                      fill_color='color', fill_alpha=0.7,
                      line_color="#884444", line_width=0.3, line_alpha=0.3)
    script_resp, div_resp = components(plot_resp)

    # disputed rate
    source_disp = ColumnDataSource(
        data=dict(
            x = state_xs,
            y = state_ys,
            color=state_colors_disp,
            name=state_name,
            rate=state_disp,
            pvalue=state_dp,
        ))
    hover_disp = HoverTool(
        tooltips=[
            ("State", "@name"),
            ("Rate", "@rate"),
            ("p-value","@pvalue"),
        ]
    )
    plot_disp = figure(title="State-wise Complaints Disputed Rate", tools=[hover_disp], toolbar_location="left",
                  plot_width=1100, plot_height=700)
    plot_disp.patches('x', 'y', source=source_disp,
                      fill_color='color', fill_alpha=0.7,
                      line_color="#884444", line_width=0.3, line_alpha=0.3)
    script_disp, div_disp = components(plot_disp)

    # correlation
    plot_corr = figure(title="Correlation: Respond Rate vs Disputed Rate")
    plot_corr.grid.grid_line_alpha=0.3
    plot_corr.xaxis.axis_label = 'Respond Rate'
    plot_corr.yaxis.axis_label =  'Disputed Rate'

    plot_corr.scatter(complaints['Response'] , complaints['Disputed'])
    script_corr, div_corr = components(plot_corr)
    #output_file("plot_corr.html")
    #save(plot_corr)                                  

    return script_resp, div_resp, script_disp, div_disp, script_corr, div_corr 

if __name__=="__main__":
    df = pd.read_csv('../Data/Consumer_Complaints.csv', header=0)
    df = data_wrangling(df)
    df = generate_map(df)
    #map_view()
