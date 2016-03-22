import csv  as csv
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.embed import components
from bokeh.sampledata.us_states import data as states

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
                                    
def generate_map(df):
    """
    sort Complaint response and disputed rate for each state
    the result is also divided by yearly
    """
    df_sub = df[['State']]
    df_sub.insert(0, 'count', 1)
    df_sub_grp = df_sub.groupby(['State']).sum().sort_index(by='count', ascending=False)
                            
    df_sub_grp.to_csv('Complaints_State.csv', index=True)
    return 0

def map_view():
    del states["HI"]
    del states["AK"]

    complaints = pd.read_csv('Complaints_State.csv', header=0,index_col=0)
    cmax = complaints.loc[complaints['count'].idxmax()]['count']
    cmin = complaints.loc[complaints['count'].idxmin()]['count']
    EXCLUDED = ("ak", "hi", "pr", "gu", "vi", "mp", "as")

    state_xs = [states[code]["lons"] for code in states]
    state_ys = [states[code]["lats"] for code in states]

    colors = ["#F1EEF6", "#D4B9DA", "#C994C7", "#DF65B0", "#DD1C77", "#980043"]
    
    state_colors = []
    for state_id in states:
        if states[state_id] in EXCLUDED:
            continue
        try:
            idx = int((complaints.at[state_id,'count']-cmin)*6/cmax)
            state_colors.append(colors[idx])
        except KeyError:
            state_colors.append("black")

    plot = figure(title="US Complaints Counts", toolbar_location="left",
                  plot_width=1100, plot_height=700)

    plot.patches(state_xs, state_ys,
                 fill_color=state_colors, fill_alpha=0.7,
                 line_color="#884444", line_width=2, line_alpha=0.3)

    script, div = components(plot)
    return script, div


if __name__=="__main__":
    #df = pd.read_csv('../Data/Consumer_Complaints.csv', header=0)
    #df = data_wrangling(df)
    #df = generate_map(df)
    map_view()
