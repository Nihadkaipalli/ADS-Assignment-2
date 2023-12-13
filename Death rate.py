


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

def population_Bargraph_2(Life_Expectancy):
    # Read the data from the CSV file
    population_data = pd.read_csv(Life_Expectancy)

    # Filter data for the specified countries and years
    countries = ['United States', 'Pakistan', 'India', 'Indonesia', 
                 'Mexico']
    years = [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
    df_filtered = population_data[(population_data['Country Name'].isin(
        countries)) & (population_data['Year'].isin(years))]
    
    # Pivot the data for grouped bar chart
    df_pivoted = df_filtered.pivot_table(
        index='Country Name', columns='Year', values='Death rate, crude (per 1,000 people)')

    # Reordering rows to make Nigeria the first country
    df_pivoted = df_pivoted.reindex(['Mexico', 'Pakistan', 'Indonesia','United States', 'India'])
   
    # Plotting the grouped bar chart
    df_pivoted.plot(kind='bar', figsize= (10,8))
    plt.xlabel('Countries')
    plt.ylabel('Death rate, crude (per 1,000 people)')
    plt.title('Death rate, crude (per 1,000 people)')
    plt.legend(title='Years')
    plt.grid(True)
    plt.show()
population_Bargraph_2('df_cleaned.csv')