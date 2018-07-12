# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:17:27 2018

@author: Pedro
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Data Import Code
files = "NBA_2017_2018.txt"

data = pd.read_csv(files, index_col = "Player")
del data["Rk"]

#Create Raw Stats Data Frame
raw_stats = pd.concat([data["MP?"], data["TRB"], data["AST"], 
                       data["STL"], data["BLK"], data["TOV"],
                       data["PTS"]], axis=1)
per_game_label = ["MP/G", "TRB/G", "AST/G", "STL/G", "BLK/G", "TOV/G",
                  "PTS/G"] 
raw_stats.columns = per_game_label


#Normalize & Merging Per Game Stats
games = data["G"]

def normalize (series1):
    return ((series1/games))

raw_stats = raw_stats.apply(normalize)
data = pd.concat([data, raw_stats], axis=1)
data = data.round(2)
del raw_stats

#Limits Data Set
restrict = data.loc[data["MP/G"]>12]

#Assigning Catagorical Data (still need to do)

PG = restrict.loc[restrict["Pos"]=="PG"]
SG = restrict.loc[restrict["Pos"]=="SG"]
SF = restrict.loc[restrict["Pos"]=="SF"]
PF = restrict.loc[restrict["Pos"]=="PF"]
C = restrict.loc[restrict["Pos"]=="C"]

all_pos = [PG,SG,SF,PF,C]
pos_string =(["PG", "SG", "SF", "PF", "C"])
color =["red", "magenta", "blue","cyan", "green"]

#Creating Position Statistics
def pearson(x_stat,y_stat):
    
    covar = np.corrcoef(x_stat, y_stat)
    return covar[0,1].round(3)


#Creating Statistic Arrays
pears_stat=[]
std_stat =[]
scoring_mean =[]

n=0
for place in all_pos:
    pears_stat.append(pearson(place["MP/G"], place["PTS/G"]))
    std_stat.append(np.std(place["PTS/G"]/place["MP/G"]))
    scoring_mean.append(np.mean(place["PTS/G"]/place["MP/G"]))
    n=n+1 
del n,
    

#Plot Code

subplot = []
n=0
for pos in all_pos:
    print(pos_string[n]+ "  Standard Deviation: " + str(round(std_stat[n],
     round(2))))
    print(pos_string[n]+ "  Average Scoring Rate: " + 
         str(round(scoring_mean[n], round(2))))
    plt.hist(pos["PTS/G"]/pos["MP/G"], bins=20, color= color[n])
    plt.title(pos_string[n] + " Scoring Rate Distribution")
    plt.xlabel("Points Per Minute")
#    plt.ylabel("Bin Count")
    plt.show()
    n=n+1
del n

n=0    
for pos in all_pos:   
    x= np.sort(pos["PTS/G"]/pos["MP/G"])
    y = np.arange(1, len(x)+1)/len(x) 
    cdf = plt.plot(x, y, marker=".", linestyle="none", color=color[n])
    n=n+1
del n

plt.title("Scoring Rate Cummulative Distribution Functions")
plt.xlabel("Points Per Minute")
#plt.ylabel("Probability")
plt.show()
plt.savefig("Scoring_Rate_CDF.png")

plt.margins(0.02)

#Creating Best Fit Line
from sklearn.linear_model import LinearRegression
line_model = LinearRegression()
line_length = np.linspace(min(restrict["MP/G"]),
                          max(restrict["MP/G"])).reshape(-1,1)
line_model.fit(restrict[["MP/G"]], restrict[["PTS/G"]])
bfline = line_model.predict(line_length)

import bokeh.plotting as bp
import bokeh.models as bm
import bokeh.io as bi

source = bp.ColumnDataSource(restrict)
color_scat = bm.CategoricalColorMapper(factors=["PG", "SG", "SF", "PF", "C"],
                                       palette= color)
hover = bm.HoverTool(tooltips=[("Player", "@Player"), ("Position", "@Pos"),
                               ("Tm", "@Tm")])
plot = bp.figure(title="2017-2018 NBA Season", title_location = "above",
                 x_axis_label= "Minutes Played",
                 y_axis_label="Points Scored",
                 tools=[hover, "pan", "wheel_zoom"])

plot.circle(x="MP/G", y="PTS/G", source= source,
            color=dict(field="Pos",transform=color_scat))

#plot.line([line_length], [bfline], line_color = "black", line_width=3)

from bokeh.io import output_file
output_file("2017_2018_Scoring_Summary.html")

bi.show(plot)

#scatter Plot
n=0
plt.plot(line_length, bfline, color="black")
for pos in all_pos:
    plt.scatter(pos["MP/G"], pos["PTS/G"])
    print(pos_string[n]+ "  Pearson Coefficient: " + str(pears_stat[n])) 
    n=n+1
plt.xlabel("Minutes Played Per Game")
plt.ylabel = ("Points Per Game")
plt.title("Minutes Played Vs Points Per Game")
plt.show()
