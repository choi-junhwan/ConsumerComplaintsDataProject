import Quandl 
import numpy as np
from bokeh.plotting import figure, show, save, output_file, vplot
from bokeh.embed import components

def datetime(x):
    return np.array(x, dtype=np.datetime64)

def stock_plot(stock, key):
    MyAPIKey = "95ujF6cdywLzzj9uPSeS"
    try:
        in_data = Quandl.get("WIKI/"+stock, collapse="yearly", authtoken=MyAPIKey)
    except:
        return False, 0.0, 0.0
        
    plot = figure(x_axis_type = "datetime")
    plot.title = "Stock Price Data from Quandl"
    plot.grid.grid_line_alpha=0.3
    plot.xaxis.axis_label = 'Date'
    plot.yaxis.axis_label = key + ' Price [$]'

    plot.line(datetime(in_data[key].index), in_data[key], line_color='blue', line_width=2, legend='%s' % stock)
    #output_file("stocks.html", title="Quandl")
    #save(plot)  # open a browser
    script, div = components(plot)
    return True, script, div


if __name__=="__main__":    
    stock_plot("YHOO","Open")
