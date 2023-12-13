# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:47:59 2023

@author: nihad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def getdata(life):
    df_life = pd.read_csv(life)
    return df_life


# Reading the data into df
df_melt = getdata('Life Expectancy.csv').melt(id_vars=[
    'Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='Value')

# Pivot the table to have Indicator Names as columns
df_pivot = df_melt.pivot_table(
    index=['Country Name', 'Country Code', 'Year'], columns='Indicator Name', values='Value').reset_index()
df_pivot.to_csv('pivoted_df.csv')
df_cleaned = df_pivot.fillna(df_pivot.mean())

# Applying Statistical Methods on cleaned dataset
df_cleaned = df_pivot.fillna(df_pivot.mean())

df_cleaned_2 = df_cleaned.drop(['Year', 'Country Name'], axis='columns')
print(df_cleaned_2.describe())
getdata("Life Expectancy.csv")

def Life_Expectancy_Bargraph_1(Life_Expectancy):
    # Read the data from the CSV file
    HIV_data = pd.read_csv(Life_Expectancy)

    # Filter data for the specified countries and years
    countries = ['Nigeria', 'Africa Eastern and Southern', 'Namibia', 'South Africa', 
                 'Zimbabwe']
    years = [2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022]
    df_filtered = HIV_data[(HIV_data['Country Name'].isin(
        countries)) & (HIV_data['Year'].isin(years))]
    
    # Pivot the data for grouped bar chart
    df_pivoted = df_filtered.pivot_table(
        index='Country Name', columns='Year', values='Antiretroviral therapy coverage (% of people living with HIV)')

    # Reordering rows to make Nigeria the first country
    df_pivoted = df_pivoted.reindex(['Nigeria', 'Africa Eastern and Southern', 'Namibia', 'South Africa', 'Zimbabwe'])
   
    # Plotting the grouped bar chart
    df_pivoted.plot(kind='bar', figsize= (10,6))
    plt.xlabel('Countries')
    plt.ylabel('Prevalence of HIV, male (% ages 15-24)')
    plt.title('Antiretroviral therapy coverage (% of people living with HIV)')
    plt.legend(title='Years')
    plt.grid(True)
    plt.show()
Life_Expectancy_Bargraph_1('df_cleaned.csv')


