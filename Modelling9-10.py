from IPython.display import HTML

# Use the direct URL to the GIF
HTML('<img src="https://media1.tenor.com/m/xrUe4KFY0dsAAAAC/brother-ew-ew.gif" width="800">')

from IPython.display import HTML

# Use the direct URL to the GIF
HTML('<img src="https://i.imgflip.com/7dsbg9.jpg" width="800">')

# these are the imports
import numpy as np
import pandas as pd

s = pd.Series([1, 3, 5, np.nan, 6, 8], index=[1,2,4,5,6,9])
print(s)
print(s[4])
print(s[9])
print(s.index)
print(s.values)

dates = pd.date_range("20240329", periods=60000, freq="D")

# df = pd.DataFrame(np.random.randn(60000, 4), index=dates, columns=list("ABCD"))
df = pd.DataFrame(np.random.randint(1, 100, (6000, 4)), index=dates, columns=list("ABCD"))

df

# df.T.index

# @title A

from matplotlib import pyplot as plt
df['A'].plot(kind='hist', bins=50, title='A')
plt.gca().spines[['top', 'right',]].set_visible(False)

df2 = pd.DataFrame(
    {
        "A": 1.0,
        "B": pd.Timestamp("20130102"),
        "C": pd.Series(1, index=list(range(4)), dtype="float32"),
        "D": np.array([3] * 4, dtype="int32"),
        "E": pd.Categorical(["test", "train", "test", "train"]),
        "F": "foo"
    }
)
df2


print(df.index)
print("-----")
df.describe()


print(df.T)

print(df.sort_values(by="B", ascending=False))
print(df)

print(df)
print(df["A"]) # df.A dot notation
print("----")
print(type(df.A))
print(df.A.values)

# slice
df[1:3]["A"]

# selection withb loc (by value)

dates[1] # this is the index for the second row
print(df)
print("-----")
print(df.loc[dates[1]])

# multiple selection with loc
print(df)
print("------")
print(df.loc[[df.index[1],dates[4]], ["A", "C"]])

df = pd.DataFrame(np.random.randn(5, 5), index=[1, 2, 3, 4, 5], columns=list("ABCDE"))
df

df.loc[[3,1], ["C", "D"]]

print(df)
print("----")
print(df.iloc[1])
print("----")
print(df.iloc[1:3, 1:2])
print("----")
print(df.iloc[[0,2,4], [1,3]])

# boolean idexing
print(df)
print("---")
print(df[df["A"] > 0])

df["E"] = ["car", "toy"] * 2 + ["ship"]
df["James"] = [7] * 5
df["James"] = 11
print(df)

import matplotlib.pylab as plt
print(df)

#

plt.plot(df.index, df.A)

# df[['A', 'B']].plot()

#PRACTICAL EXAMPLE
!pip install --upgrade pandas-datareader
!pip install yfinance

import requests

r = requests.get("https://github.com/itb-ie/pandas-apple-stock/raw/master/iphone-dates-2019.xlsx")
with open("sample_data/launch.xlsx", "wb") as f:
  f.write(r.content)

!wget https://github.com/itb-ie/pandas-apple-stock/raw/master/iphone-dates-2019.xlsx -O sample_data/launch2.xlsx

from matplotlib import pylab as plt
import pandas as pd
import pandas_datareader.data as web
import yfinance as yfin

yfin.pdr_override()

df1 = web.DataReader('AAPL', start='2006-01-01', end='2024-04-01')
df2 = pd.read_excel("sample_data/launch.xlsx")

df2['Date'] = pd.to_datetime(df2.date)
df1['Date'] = pd.to_datetime(df1.index)
df1.insert(0, 'ID', range(0, len(df1)))
print(df1)

index2 = []
for date2 in df2.Date:
    if df1.ID[df1.Date == date2].values.size:
        index2.append(int(df1.ID[df1.Date == date2].values[0]))
    elif df1.ID[df1.Date == date2 + pd.DateOffset(1)].values.size:
        index2.append(int(df1.ID[df1.Date == date2 + pd.DateOffset(1)].values[0]))
    elif df1.ID[df1.Date == date2 + pd.DateOffset(2)].values.size:
        index2.append(int(df1.ID[df1.Date == date2 + pd.DateOffset(2)].values[0]))

    else:
        print(f"Did not find {date2}")

print(index2)

plt.figure("Apple Stock")
plt.figure(figsize = (20, 10))
plt.plot(df1["Date"], df1["Close"], '-^', ms=11, mfc="red", linewidth=0.6,
         markevery=index2, label="Iphone launch date")
plt.xlabel("Dates")
plt.legend(loc="upper left")
plt.show()
df1.to_csv("sample_data/result.csv")

# Plotting using pandas' built-in functionality
ax = df1['Close'].plot(
    figsize=(20, 10),
    style='-ob',
    linewidth=0.6,
    markevery=index2,  # Highlight the launch dates
    markerfacecolor='red',
    markersize=11,
    label='Close Price'
)

# Highlight iPhone launch dates with vertical lines
for launch_date in df1.iloc[index2].index:
    ax.axvline(x=launch_date, color='gray', linestyle='--', linewidth=0.5)

# Labeling and legend
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.set_title("Apple Stock Prices with iPhone Launch Dates Highlighted")
ax.legend(loc="upper left")

# Show the plot
plt.show()

from matplotlib import pylab as plt
import pandas as pd
import pandas_datareader.data as web

company = "TSLA"
df = web.DataReader(company, start='2024-01-01', end='2024-04-18')
# print(df)

plt.style.use("default")

plt.figure(f"{company} CandleStick", figsize=(20,10))
plt.title(f"{company} CandleStick")
# plt.plot(df.index, df.Close)
plt.vlines(x=df.index, ymin=df.Low, ymax=df.High, color="gray")

green = df[df.Open <= df.Close].copy()
green['Height'] = green.Close - green.Open

red = df[df.Open > df.Close].copy()
red['Height'] = red.Open - red.Close

plt.bar(x=green.index, height=green.Height, bottom=green.Open, color="green")
plt.bar(x=red.index, height=red.Height, bottom=red.Close, color="red")


plt.show()

from IPython.display import HTML

# Use the direct URL to the GIF
HTML('<img src="https://media.tenor.com/9icIB76KxhgAAAAC/captain-america-i-can-do-this-all-day.gif" width="800">')

import pandas as pd

# Load the data
df = pd.read_csv("https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv")

# Filter data according to specific criteria
df_filtered = df[(df.year >= 2005) & (df.sex == "girl") & (df.percent > 0.005)]

# Group by 'name' and calculate the mean of the 'percent' column only
grouped = df_filtered.groupby('name')['percent'].mean()
print(grouped)

# Convert the Series back to DataFrame for further manipulation if necessary
mean_df = grouped.reset_index()

# Sort the data by 'percent' in descending order
sorted_mean_df = mean_df.sort_values(by="percent", ascending=False)

# Display the results
print(sorted_mean_df.values)

import pandas as pd

# Load the data
df = pd.read_csv("https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv")

# Filter data according to specific criteria
df_filtered = df[(df.year <= 1900) & (df.sex == "girl") & (df.percent > 0.005)]

# Group by 'name' and calculate the mean of the 'percent' column only
grouped = df_filtered.groupby('name')['percent'].mean()

# Convert the Series back to DataFrame for further manipulation if necessary
mean_df = grouped.reset_index()

# Sort the data by 'percent' in descending order
sorted_mean_df = mean_df.sort_values(by="percent", ascending=False)

# Display the results
print(sorted_mean_df.values)

df = pd.read_json("https://covid.cdc.gov/covid-data-tracker/COVIDData/getAjaxData?id=us_trend_by_USA_v2")
df

df2 = df['us_trend_by_Geography_v2'].apply(pd.Series)
df2


# remove Geography, runid
df2.drop(columns=["Geography", "runid"], inplace=True) # in place means there is no need to assign to another df
df2

df3 = df2.set_index(keys="week_ending_date")
df3
df3.index

df3.index = pd.to_datetime(df3.index)


df3.index

import matplotlib.pyplot as plt
plt.ticklabel_format(style='plain')
df3['COVID_deaths_total'].plot(figsize=(30, 15))

import matplotlib.dates as mdates
plt.ticklabel_format(style='plain')


df3['COVID_deaths_weekly'].plot(figsize=(25, 10), kind="bar")

new_df = df3.loc[:, ['total_adm_all_covid_confirmed_past_7days', 'COVID_deaths_weekly']]
new_df['COVID_deaths_weekly'] *= 4
new_df = new_df.iloc[30:100]
# plt.ticklabel_format(style='plain')
new_df.plot(figsize=(20, 10), kind="bar")

new_df = df3.loc[:, ['total_adm_all_covid_confirmed_past_7days', 'COVID_deaths_weekly']]
new_df['COVID_deaths_weekly'] *= 4
new_df = new_df.iloc[-52:]
# plt.ticklabel_format(style='plain')
new_df.plot(figsize=(20, 10), kind="bar")

import numpy as np
df1 = pd.DataFrame() # empty data frame
s1 = np.arange(0, 1000)
s2 = np.sin(s1/50)

df1.index = s1
df1['sin'] = s2

df1.plot()

df2 = df1.copy()
df2[df2['sin'] > 0.5] = 0.3
df2[df2['sin'] < -0.5] = -0.8

df2.plot()

s3 = pd.Series(np.random.randn(1000)/10)
s4 = s3.cumsum()
s4


df2['random'] = s4
df2.plot(figsize=(20, 10))

