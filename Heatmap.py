# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:01:18 2023

@author: nihad
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def getdata(Population):
    df_life = pd.read_csv(Population)
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

#Heatmap for India

def India_HeatMap(Population):
    data = pd.read_csv(Population)
    India_data = data[data['Country Name'] == 'India']

# Select relevant indicators
    indicators = [
       'Population, total',
       'Death rate, crude (per 1,000 people)',
       'Life expectancy at birth, total (years)',
       'Mortality rate, neonatal (per 1,000 live births)',
       'Mortality rate, infant (per 1,000 live births)',
       'International migrant stock (% of population)',
       'Number of infant deaths',
       'Cause of death, by non-communicable diseases (% of total)']

# Create a subset of data with selected indicators
    India_subset = India_data[indicators]

# Plotting the heatmap
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(India_subset.corr(), annot=True, cmap='magma', fmt='.2f', annot_kws={"size": 10})
    plt.title('Correlation Heatmap of Indicators for India')
    plt.show()
India_HeatMap('df_cleaned.csv')

#Heatmap for United States

def United_States_HeatMap(Population):
    data = pd.read_csv(Population)
    United_States_data = data[data['Country Name'] == 'United States']

# Select relevant indicators
    indicators = [
        'Population, total',
        'Mortality caused by road traffic injury (per 100,000 population)',
        'Suicide mortality rate (per 100,000 population)',
        'Suicide mortality rate, male (per 100,000 male population)',
        'Suicide mortality rate, female (per 100,000 female population)',
        'Mortality rate, infant (per 1,000 live births)',
        'Adolescent fertility rate (births per 1,000 women ages 15-19)',
        'Cause of death, by non-communicable diseases (% of total)']
       

# Create a subset of data with selected indicators
    United_States_subset = United_States_data[indicators]

# Plotting the heatmap
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(United_States_subset.corr(), annot=True, cmap='magma', fmt='.2f', annot_kws={"size": 10})
    plt.title('Correlation Heatmap of Indicators for United States')
    plt.show()
United_States_HeatMap('df_cleaned.csv')
