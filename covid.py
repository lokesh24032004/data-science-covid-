import pandas as pd 
import numpy as np

covid_df=pd.read_csv('corona_dataset01.csv')

# print("dataset installed sucessfully")
# print(covid_df.head(20))
# print(covid_df)
# print(covid_df.tail(10))
# print(covid_df.index)
# print(covid_df.Deaths) 
# print(covid_df.loc[0])
# print(covid_df.loc[1].index)
# print(covid_df.set_index(np.arange(10,30),inplace=True))
# print(covid_df.set_index(np.arange(10,30)))
# print(type(covid_df))
# print(type(covid_df.Deaths))
# print(covid_df.Deaths.index)

# print(covid_df.iloc[5].Deaths)
# print(covid_df.Deaths.loc[15])
# print(covid_df['Deaths'].iloc[5])
# print(covid_df.at[5,'Deaths']) 
# print(covid_df.iloc[5].loc['Deaths'])

# row_series=covid_df.loc[5]
# print(row_series.Deaths)
# print(row_series.iloc[2])
# print(row_series.loc['Deaths'])
# print(row_series['Deaths'])

# columns_series=covid_df.Active
# print(columns_series.iloc[6])
# print(columns_series.loc[6])
# print(columns_series[6])

# my_array=np.array([[9,5,1,4], [6,2,7,9], [6,4,9,5], [1,4,6,2], [7,9,6,4]])
# print(my_array)

# print(my_array[1,:])
# print(my_array[:,3])
# print(my_array[2,2])
# print(my_array[1:3,1:3])

# selected_columns = covid_df.loc[0:10, 'Deaths':'New deaths']
# print("Selected Columns using loc:")
# print(selected_columns) 

# sorted_df = covid_df.sort_values('Deaths').reset_index()
# selected_rows = sorted_df.iloc[0:187:10]
# print("Selected Rows after Sorting:")
# print(selected_rows)

# corona=(covid_df._series)
# print(corona)
# c=0
# r=0
# for x in corona:
#     c=c+x
#     r=r+1
# mean=c/r
# print(mean)
# mopa=np.mean(corona)
# print(corona)
# print("the mean value is:",mopa)

# corona=(covid_df.Confirmed)
# confirmed=np.mean(corona)
# print("confirmed =",confirmed)

# corona=(covid_df.Deaths)
# deaths=np.mean(corona)
# print("deaths =",deaths)

# corona=(covid_df.Recovered)
# recovered=np.mean(corona)
# print("recovered =",recovered)

# corona=(covid_df.Active)
# active=np.mean(corona)
# print("active =",active)

# corona=(covid_df.Newcases)
# newcases=np.mean(corona)
# print("newcases =",newcases)

# corona=(covid_df.Newdeaths)
# newdeaths=np.mean(corona)
# print("newdeaths =",newdeaths)

# corona=(covid_df.Newrecovered)
# newrecovered=np.mean(corona)
# print("newrecovered =",newrecovered)

# corona=(covid_df.DeathshunderdCases)
# deathshunderdcases=np.mean(corona)
# print("deathshunderdcases =",deathshunderdcases)

# corona=(covid_df.RecoveredhunderdCases)
# recoveredhunderdcases=np.mean(corona)
# print("recoveredhunderdcases =",recoveredhunderdcases)

# corona=(covid_df.DeathshunderdRecovered)
# deathshunderdrecovred=np.mean(corona)
# print("deathshunderdrecovered =",deathshunderdrecovred)

# corona=(covid_df.Confirmedlastweek)
# confirmedlastweek=np.mean(corona)
# print("confirmedlastweek =",confirmedlastweek)

# corona=(covid_df.aweekchange)
# aweekchange=np.mean(corona)
# print("aweekchange =",aweekchange)

# corona=(covid_df.aweekincrease)
# aweekincrease=np.mean(corona)
# print("aweekincrease =",aweekincrease)

# a=confirmed
# b=deaths
# c=recovered
# d=active
# e=newcases
# f=newdeaths
# g=newrecovered
# h=deathshunderdcases
# i=recoveredhunderdcases
# j=deathshunderdrecovred
# k=confirmedlastweek
# l=aweekchange
# m=aweekincrease

# high=max(a,b,c,d,e,f,g,h,i,j,k,l,m)
# print(high)

# import matplotlib.pyplot as plt

# covid_df.Deaths.plot.hist()
# plt.show()

# print("Unique values 'WHORegion':", covid_df.WHORegion.unique())
# print("Value counts 'WHORegion':")
# print(covid_df.WHORegion.value_counts())
# covid_df.WHORegion.value_counts.plot.bar()
# plt.show()

# print(covid_df.shape)
# print(covid_df.describe())
# print(covid_df.info())
# print(covid_df.nunique())
# print(covid_df.isnull().sum())

# def MultiplyBy2(n):
#     return n*2

# print(covid_df.Deaths.apply(MultiplyBy2))

# def Multiplyby2(n):
#     return n*2
# print(covid_df.Active.apply(Multiplyby2))

# import seaborn as sns
# import matplotlib.pyplot as plt
 
 
# sns.boxplot( x="WHORegion", y='Newdeaths', data=covid_df )
# plt.show()

# import seaborn as sns
# import matplotlib.pyplot as plt 

# sns.pairplot(covid_df, hue='WHORegion', height=10)
# plt.show()

# import numpy as np

# # List of columns to process
# columns = [
#     'Confirmed', 'Deaths', 'Recovered', 'Active', 'Newcases',
#     'Newdeaths', 'Newrecovered', 'DeathshunderdCases',
#     'RecoveredhunderdCases', 'DeathshunderdRecovered',
#     'Confirmedlastweek', 'aweekchange', 'aweekincrease'
# ]

# # Dictionary to hold the mean values
# mean_values = {}

# # Calculate the mean for each column and store in dictionary
# for col in columns:
#     corona = covid_df[col]
#     mean_values[col] = np.mean(corona)
#     print(f"{col} = {mean_values[col]}")

# # Find the maximum value among all means
# high = max(mean_values.values())
# print(high)

# from scipy.stats import skew
# from scipy.stats import kurtosis
# skewness = skew(covid_df["Confirmed"], axis=0, bias=True)
# print("Skewness:", skewness)

# import matplotlib.pyplot as plt

# plt.hist(covid_df["WHORegion"], bins=25, color='green', alpha=0.5)
# plt.title("who region Distribution")
# plt.xlabel("who region")
# plt.ylabel("Frequency")
# plt.show()

# from scipy.stats import kurtosis
# import matplotlib.pyplot as plt
# kurtosis1 = kurtosis(covid_df["Deaths"], bias=True)
# print("Kurtosis:", kurtosis1)