import pandas as pd
import numpy as np

sample_df=pd.read_csv('student-dataset.csv')

print("dataset install sucessfully")
# print(sample_df.head(20))
# print(sample_df)
# print(sample_df.tail(20))
# print(sample_df.index)
# print(sample_df.city)
# print(type(sample_df))
# print(type(sample_df.nationality))
# print(type(sample_df.latitude))
# print(sample_df.loc[3])
# print(sample_df.loc[1].index)
# print(sample_df.latitude.index)
# print(sample_df.set_index(np.arange(0,307),inplace=True))
# print(sample_df.set_index(np.arange(0,307)))
# print(sample_df.iloc[3].latitude)

# row_series=sample_df.loc[54]
# print(row_series.latitude)

# columns_series=sample_df.age
# print(columns_series.loc[7])

# selected_columns = sample_df.loc[0:5, 'id':'longitude']
# print(selected_columns)
# sorted_df=sample_df.sort_values('age').reset_index()
# selected_rows = sorted_df.iloc[0:308:10]
# print("Selected Rows after Sorting:")
# print(selected_rows)

# thanos=sample_df.age
# print(thanos)

# spidey=np.mean(sample_df.age)
# print(sample_df.age)
# print("the mean value of",spidey)

# columns=['latitude','longitude','age','englishgrade','mathgrade',
#          'sciencesgrade','languagegrade','portfoliorating',
#          'coverletterrating','refletterrating'
#          ]
# mean_values={}
# for x in columns:
#     thanos=sample_df[x]
#     mean_values[x]=np.mean(thanos)
#     print(f"{x}={mean_values[x]}")

# high=max(mean_values.values())
# print(high)    

# import matplotlib.pyplot as plt
# sample_df.age.plot.hist()
# plt.show()
# print("unique values age :",sample_df.age.unique())
# print("value counts of age :")
# sample_df.age.value_counts().plot.bar()
# plt.show()

# print(sample_df.shape)
# print(sample_df.describe())
# print(sample_df.info())
# print(sample_df.nunique())
# print(sample_df.isnull().sum())

# def multiplyby2(n):
#     return n*2
# print(sample_df.age.apply(multiplyby2))

# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.boxplot(x="gender",y="age",data=sample_df)
# plt.show()

# from scipy.stats import skew
# from scipy.stats import kurtosis
# skewness=skew(sample_df["age"],axis=0,bias=True)
# print("skewness :",skewness)

# import matplotlib.pyplot as plt
# plt.hist(sample_df["age"],bins=25,color='violet',alpha=0.6)
# plt.title("age distribution")
# plt.xlabel("age")
# plt.ylabel("frequency")
# plt.show()

from scipy.stats import kurtosis
import matplotlib.pyplot as plt
kurtosis1=kurtosis(sample_df["age"],bias=True)
print("kurtosis :",kurtosis1)