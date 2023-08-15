import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
sns.set_theme(style='darkgrid')

#self-made Siskel and Ebert dataset merged with international box office dataset in Excel
#adjusted 2023 box office columns also already calculated in Excel 
#importing Siskel and Ebert dataset
se_df = pd.read_csv('/Users/erikrice/Downloads/Siskel + Ebert 2 - Sheet1 (1).csv')
print(se_df.head())
print(se_df.columns)

#cleaning data
se_df['Global_Box_Office'] = se_df['Global_Box_Office'].str.replace('$', '')
se_df['Domestic_Box_Office'] = se_df['Domestic_Box_Office'].str.replace('$', '')
se_df['2023_Global_Box_Office'] = se_df['2023_Global_Box_Office'].str.replace('$', '')
se_df['2023_Domestic_Box_Office'] = se_df['2023_Domestic_Box_Office'].str.replace('$', '')
se_df['Global_Box_Office'] = se_df['Global_Box_Office'].str.replace(',', '')
se_df['Domestic_Box_Office'] = se_df['Domestic_Box_Office'].str.replace(',', '')
se_df['2023_Global_Box_Office'] = se_df['2023_Global_Box_Office'].str.replace(',', '')
se_df['2023_Domestic_Box_Office'] = se_df['2023_Domestic_Box_Office'].str.replace(',', '')
se_df['Siskel'] = se_df['Siskel'].replace('x', None)
se_df['Ebert'] = se_df['Ebert'].replace('x', None)
se_df.dropna(axis=0, inplace=True)
se_df['Global_Box_Office'] = se_df['Global_Box_Office'].astype(int)
se_df['Domestic_Box_Office'] = se_df['Domestic_Box_Office'].astype(int)
se_df['Siskel'] = se_df['Siskel'].astype(int)
se_df['Ebert'] = se_df['Ebert'].astype(int)
se_df['2023_Global_Box_Office'] = se_df['2023_Global_Box_Office'].astype(int)
se_df['2023_Domestic_Box_Office'] = se_df['2023_Domestic_Box_Office'].astype(int)
se_df.drop('Percent_Domestic', axis=1, inplace=True)
print(se_df.dtypes)
print(se_df.head())

#what does the box office look like during Siskel and Ebert's most popular decade? 
#looking at both actual and adjusted
mean_domestic_se_era = se_df.groupby('Year')[['Domestic_Box_Office', '2023_Domestic_Box_Office']].mean()
print(mean_domestic_se_era.head())
sns.lineplot(x='Year', y='Domestic_Box_Office', data=mean_domestic_se_era)
sns.lineplot(x='Year', y='2023_Domestic_Box_Office', data=mean_domestic_se_era)
plt.show()

#creating "Thumbs" column
se_df['Thumbs'] = se_df['Siskel'] + se_df['Ebert']

#before checking correlation, what was the average rating Siskel and Ebert gave each year?
annual_se = se_df.groupby('Year').agg(
    Siskel_Thumbs = ('Siskel', 'mean'),
    Ebert_Thumbs = ('Ebert', 'mean'),
    Combined_Thumbs = ('Thumbs', 'mean'),
)
annual_se['Mean_Thumbs'] = annual_se['Combined_Thumbs'] / 2
print(annual_se.head())

#line graph of average S+E score over observed decade
sns.set_palette('Accent')
sns.lineplot(data=annual_se, x='Year', y='Siskel_Thumbs', label='Siskel')
sns.lineplot(data=annual_se, x='Year', y='Ebert_Thumbs', label='Ebert')
sns.lineplot(data=annual_se, x='Year', y='Mean_Thumbs', label='Average')
plt.axhline(.5, linewidth=2, color='r')
plt.xlabel('Year')
plt.ylabel('Mean "Thumbs" Rating')
plt.title('Mean Siskel + Ebert Ratings Over Most Popular Decade')
plt.show()

#qucik visual comparing Siskel and Ebert response (number of "Thumbs") to adjusted box office
sns.catplot(data=se_df, x='Thumbs', y='2023_Domestic_Box_Office', kind='bar')
plt.show()

#heatmap to look at a variety of correlations 
se_numeric = se_df.drop('Film', axis=1)
sns.heatmap(se_numeric.corr(), annot=True, cmap="YlGnBu")
plt.title('Correlation Analysis: Siskel + Ebert and Box Office Results')
plt.show()

#no point in exploring box office potency of Siskel and Ebert further (by year, say). no real correlation between S+E and box office. 
#importing Rotten Tomatoes dataset. 2023 adjusted inflation columns calculated in Excel before importing
rt_df = pd.read_csv('/Users/erikrice/Downloads/RT - Sheet1 (1).csv')
print(rt_df.head())
print(rt_df.columns)

#cleaning RT dataset
rt_df['Global'] = rt_df['Global'].str.replace('$', '')
rt_df['Domestic'] = rt_df['Domestic'].str.replace('$', '')
rt_df['International %'] = rt_df['International %'].str.replace('$', '')
rt_df['2023_Global_Box_Office'] = rt_df['2023_Global_Box_Office'].str.replace('$', '')
rt_df['2023_Domestic_Box_Office'] = rt_df['2023_Domestic_Box_Office'].str.replace('$', '')
rt_df['Global'] = rt_df['Global'].str.replace(',', '')
rt_df['Domestic'] = rt_df['Domestic'].str.replace(',', '')
rt_df['International %'] = rt_df['International %'].str.replace(',', '')
rt_df['2023_Global_Box_Office'] = rt_df['2023_Global_Box_Office'].str.replace(',', '')
rt_df['2023_Domestic_Box_Office'] = rt_df['2023_Domestic_Box_Office'].str.replace(',', '')
rt_df['Domestic'] = rt_df['Domestic'].str.replace('-', '0')
rt_df['2023_Global_Box_Office'] = rt_df['2023_Global_Box_Office'].replace('#VALUE!', None)
rt_df['2023_Domestic_Box_Office'] = rt_df['2023_Domestic_Box_Office'].replace('#VALUE!', None)
rt_df['Global'] = rt_df['Global'].astype(float)
rt_df['Domestic'] = rt_df['Domestic'].astype(float)
rt_df['International %'] = rt_df['International %'].astype(float)
rt_df['2023_Global_Box_Office'] = rt_df['2023_Global_Box_Office'].astype(float)
rt_df['2023_Domestic_Box_Office'] = rt_df['2023_Domestic_Box_Office'].astype(float)
print(rt_df.dtypes)
print(rt_df.info())
rt_df.dropna(axis=0, inplace=True)
print(rt_df.info())

#Clarifying labels and removing all superfluous columns
print(rt_df.columns)
rt_df['Year_Rank'] = rt_df['Rank']
rt_df['Global_Box_Office'] = rt_df['Global']
rt_df['Domestic_Box_Office'] = rt_df['Domestic']
rt_df['International_Box_Office'] = rt_df['International %']
rt_df['RT_Critic_Score'] = rt_df['RT Score']
rt_df['RT_Audience_Score'] = rt_df['Audience Score']
rt_df = rt_df[['Year', 'Year_Rank', 'Global_Box_Office', 'Domestic_Box_Office', 'International_Box_Office', 'RT_Critic_Score', 'RT_Audience_Score', '2023_Global_Box_Office', '2023_Domestic_Box_Office']]

#a look at the box office performance during the last decade of movies. (pandemic is obviously a major factor)
mean_domestic_rt_era = rt_df.groupby('Year')[['Domestic_Box_Office', '2023_Domestic_Box_Office']].mean()
print(mean_domestic_rt_era.head())
sns.lineplot(x='Year', y='Domestic_Box_Office', data=mean_domestic_rt_era)
sns.lineplot(x='Year', y='2023_Domestic_Box_Office', data=mean_domestic_rt_era)
plt.show()

#comapring box office of peak Siskel and Ebert years against the last decade of Rotten Tomatoes' prominence
sns.lineplot(x='Year', y='Domestic_Box_Office', data=mean_domestic_se_era)
sns.lineplot(x='Year', y='2023_Domestic_Box_Office', data=mean_domestic_se_era)
sns.lineplot(x='Year', y='Domestic_Box_Office', data=mean_domestic_rt_era)
sns.lineplot(x='Year', y='2023_Domestic_Box_Office', data=mean_domestic_rt_era)
plt.show()

#adding a "fresh/rotten" column
Fresh_or_Rotten = []
for x in rt_df['RT_Critic_Score']:
    if x >= 60:
        Fresh_or_Rotten.append('Fresh')
    else:
        Fresh_or_Rotten.append('Rotten')
rt_df['Fresh_or_Rotten'] = Fresh_or_Rotten
print(rt_df.head())

#making numeric version as well
rt_binary1 = rt_df['Fresh_or_Rotten'].str.replace('Fresh', '1')
rt_binary2 = rt_binary1.str.replace('Rotten', '0')
rt_binary2 = rt_binary2.astype(float)

#adding this column 
rt_df['Fresh_Rotten_Numeric'] = rt_binary2

#for fun, also adding an aggregated critic and audience column. is there a wisdom of the crowds element to this?
rt_df['RT_Combo_Score'] = (rt_df['RT_Critic_Score'] + rt_df['RT_Audience_Score']) / 2
print(rt_df.head())

#time for a correlation analysis and visualizations
sns.set_palette('Set1')
sns.scatterplot(data=rt_df, x='RT_Critic_Score', y='2023_Domestic_Box_Office')
plt.xlabel('Rotten Tomatoes Critic Score')
plt.ylabel('Adjusted US Domestic Box Office')
plt.title('Rotten Tomatoes Score Measured Against Box Office Performance')
plt.show()

#with categorical variable
sns.set_palette('Set2')
sns.scatterplot(data=rt_df, x='RT_Critic_Score', y='2023_Domestic_Box_Office', hue='Fresh_or_Rotten', palette='Accent')
plt.xlabel('Rotten Tomatoes Critic Score')
plt.ylabel('Adjusted US Domestic Box Office')
plt.title('Rotten Tomatoes Score Measured Against Box Office Performance')
plt.legend()
plt.show()

#and a heatmap
rt_numeric = rt_df[['Year', 'Year_Rank', 'Global_Box_Office', 'Domestic_Box_Office', 'International_Box_Office', 'RT_Critic_Score', 'RT_Audience_Score',
       '2023_Global_Box_Office', '2023_Domestic_Box_Office', 'Fresh_Rotten_Numeric',
       'RT_Combo_Score']]
sns.heatmap(rt_numeric.corr(), annot=True)
plt.show()

#only a weak correlation for RT and box office. what if we remove the COVID years from the analysis?
no_covid_rt_df = rt_numeric[rt_numeric['Year'] < 2020]
sns.scatterplot(data=no_covid_rt_df, x='RT_Critic_Score', y='2023_Domestic_Box_Office', hue='Fresh_Rotten_Numeric')
plt.show()
sns.heatmap(no_covid_rt_df.corr(), annot=True)
plt.show()

#more of an effect, but stil weak.
#is Rotten Tomatoes the new Siskel and Ebert? yes, but not in the way people think. 

#checking for linear regression analysis
x1 = se_df['Thumbs'].to_numpy()
y1 = se_df['2023_Domestic_Box_Office'].to_numpy()
x1 = x1.reshape((-1, 1))
model1 = LinearRegression().fit(x1,y1)
r_sq1 = model1.score(x1,y1)
print(f'coefficient of determination: {r_sq1}')
print(f'intercept: {model1.intercept_}')
print(f'slope: {model1.coef_}')

#checking for polynomial regression
x2 = x1
y2 = y1
x2_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x2)
model2 = LinearRegression().fit(x2_, y2)
r_sq2 = model2.score(x2_, y2)
print(f'coefficient of determination: {r_sq2}')
print(f'intercept: {model2.intercept_}')
print(f'coefficients: {model2.coef_}')

#not a good fit for regression.