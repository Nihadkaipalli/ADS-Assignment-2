


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

def population_Bargraph_1(Life_Expectancy):
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
        index='Country Name', columns='Year', values='Population, total')

    # Reordering rows to make Nigeria the first country
    df_pivoted = df_pivoted.reindex(['Mexico', 'Pakistan', 'Indonesia','United States', 'India'])
   
    # Plotting the grouped bar chart
    df_pivoted.plot(kind='bar', figsize= (10,8))
    plt.xlabel('Countries')
    plt.ylabel('Population, Total')
    plt.title('Population, Total')
    plt.legend(title='Years')
    plt.grid(True)
    plt.show()

# Call the function and pass the CSV file path as an argument
population_Bargraph_1('df_cleaned.csv')


#Death Rate

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

# Call the function and pass the CSV file path as an argument
population_Bargraph_2('df_cleaned.csv')


#Life Expectancy

def life_expectancy_line(Life_Expectancy):
    # Load the dataset from the CSV file
    Life_exp_data = pd.read_csv(Life_Expectancy)

    # Filter data for the specified countries and years
    countries = ['United States', 'Pakistan', 'India', 'Indonesia', 
                 'Mexico']
    years = [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]

    df_filtered = Life_exp_data[(Life_exp_data['Country Name'].isin(countries)) & (Life_exp_data['Year'].isin(years))]

    # Pivot the data for easier plotting
    df_pivoted = df_filtered.pivot(index='Year', columns='Country Name', values='Life expectancy at birth, total (years)')

    # Plotting
    plt.figure(figsize=(10, 6))

    for country in countries:
        plt.plot(df_pivoted.index, df_pivoted[country], label=country)

    plt.title('Life expectancy at birth')
    plt.xlabel('Year')
    plt.ylabel('Life Expectancy')
    plt.legend(title = 'Countries')
    plt.grid(True)
    plt.show()
   

# Call the function and pass the CSV file path as an argument
life_expectancy_line('df_cleaned.csv')


#Neonatal Mortality Rate


def Neonatal_line(Neonatal):
    # Load the dataset from the CSV file
    Neonatal_data = pd.read_csv(Neonatal)

    # Filter data for the specified countries and years
    countries = ['United States', 'Pakistan', 'India', 'Indonesia', 
                 'Mexico']
    years = [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]

    df_filtered = Neonatal_data[(Neonatal_data['Country Name'].isin(countries)) & (Neonatal_data['Year'].isin(years))]

    # Pivot the data for easier plotting
    df_pivoted = df_filtered.pivot(index='Year', columns='Country Name', values='Mortality rate, neonatal (per 1,000 live births)')

    # Plotting
    plt.figure(figsize=(10, 6))

    for country in countries:
        plt.plot(df_pivoted.index, df_pivoted[country], label=country)

    plt.title('Mortality rate, Neonatal (per 1,000 live births)')
    plt.xlabel('Year')
    plt.ylabel('Mortality rate, Neonatal (per 1,000 live births)')
    plt.legend(title = 'Countries')
    plt.grid(True)
    plt.show()
   

# Call the function and pass the CSV file path as an argument
Neonatal_line('df_cleaned.csv')


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

# Call the function and pass the CSV file path as an argument
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

# Call the function and pass the CSV file path as an argument
United_States_HeatMap('df_cleaned.csv')
