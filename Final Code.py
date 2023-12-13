

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.stats import kurtosis


def getdata(Population):
    """
    Read data from a CSV file and return it as a pandas DataFrame.

    Args:
    - Population: str, path to the CSV file containing population data

    Returns:
    - df_life: pandas DataFrame containing the data read from the CSV
    """
    df_life = pd.read_csv(Population)
    return df_life


# Reading the data into dataframe
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

# Skewness
India_data = df_cleaned[df_cleaned['Country Name'] == "India"]
India_skewness = skew(India_data['Population, total'])
print("Skewness of India Population", India_skewness)

# Kurtosis
India_kurtosis = kurtosis(India_data['Population, total'])
print("Kurtosis of India Population", India_kurtosis)

getdata("Life Expectancy.csv")


def population_Bargraph_1(Life_Expectancy):
    """
    Generate a grouped bar chart showing the total population of specified countries over specific years.

    Args:
    - Life_Expectancy: str, path to the CSV file containing life expectancy data
    """
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
    df_pivoted = df_pivoted.reindex(
        ['Mexico', 'Pakistan', 'Indonesia', 'United States', 'India'])

    # Plotting the grouped bar chart
    df_pivoted.plot(kind='bar', figsize=(10, 6))
    plt.xlabel('Countries')
    plt.ylabel('Population, Total')
    plt.title('Population, Total')
    plt.legend(title='Years')
    plt.grid(True)
    plt.show()


# Call the function and pass the CSV file path as an argument
population_Bargraph_1('df_cleaned.csv')


# Death Rate

def population_Bargraph_2(Life_Expectancy):
    """
    Generate a grouped bar chart showing the death rate of specified countries over specific years.

    Args:
    - Life_Expectancy: str, path to the CSV file containing life expectancy data
    """
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
    df_pivoted = df_pivoted.reindex(
        ['Mexico', 'Pakistan', 'Indonesia', 'United States', 'India'])

    # Plotting the grouped bar chart
    df_pivoted.plot(kind='bar', figsize=(10, 6))
    plt.xlabel('Countries')
    plt.ylabel('Death rate, crude (per 1,000 people)')
    plt.title('Death rate, crude (per 1,000 people)')
    plt.legend(title='Years')
    plt.grid(True)
    plt.show()


# Call the function and pass the CSV file path as an argument
population_Bargraph_2('df_cleaned.csv')


# Life Expectancy

def life_expectancy_line(Life_Expectancy):
    """
    Generate a line plot showing the life expectancy trend of specified countries over specific years.

    Args:
    - Life_Expectancy: str, path to the CSV file containing life expectancy data
    """
    # Load the dataset from the CSV file
    Life_exp_data = pd.read_csv(Life_Expectancy)

    # Filter data for the specified countries and years
    countries = ['United States', 'Pakistan', 'India', 'Indonesia',
                 'Mexico']
    years = [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]

    df_filtered = Life_exp_data[(Life_exp_data['Country Name'].isin(
        countries)) & (Life_exp_data['Year'].isin(years))]

    # Pivot the data for easier plotting
    df_pivoted = df_filtered.pivot(
        index='Year', columns='Country Name', values='Life expectancy at birth, total (years)')

    # Plotting
    plt.figure(figsize=(10, 6))

    for country in countries:
        plt.plot(df_pivoted.index, df_pivoted[country], label=country)

    plt.title('Life expectancy at birth')
    plt.xlabel('Year')
    plt.ylabel('Life Expectancy')
    plt.legend(title='Countries')
    plt.grid(True)
    plt.show()


# Call the function and pass the CSV file path as an argument
life_expectancy_line('df_cleaned.csv')


# Neonatal Mortality Rate

def Neonatal_line(Neonatal):
    """
    Generate a line plot showing the neonatal mortality rate trend of specified countries over specific years.

    Args:
    - Neonatal: str, path to the CSV file containing neonatal mortality rate data
    """
    # Load the dataset from the CSV file
    Neonatal_data = pd.read_csv(Neonatal)

    # Filter data for the specified countries and years
    countries = ['United States', 'Pakistan', 'India', 'Indonesia',
                 'Mexico']
    years = [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]

    df_filtered = Neonatal_data[(Neonatal_data['Country Name'].isin(
        countries)) & (Neonatal_data['Year'].isin(years))]

    # Pivot the data for easier plotting
    df_pivoted = df_filtered.pivot(
        index='Year', columns='Country Name', values='Mortality rate, neonatal (per 1,000 live births)')

    # Plotting
    plt.figure(figsize=(10, 6))

    for country in countries:
        plt.plot(df_pivoted.index, df_pivoted[country], label=country)

    plt.title('Mortality rate, Neonatal (per 1,000 live births)')
    plt.xlabel('Year')
    plt.ylabel('Mortality rate, Neonatal (per 1,000 live births)')
    plt.legend(title='Countries')
    plt.grid(True)
    plt.show()


# Call the function and pass the CSV file path as an argument
Neonatal_line('df_cleaned.csv')

# Heatmap for India


def India_HeatMap(Population):
    """
   Generate a heatmap showing the correlation between different indicators for India.

   Args:
   - Population: str, path to the CSV file containing population data
   """
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
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(India_subset.corr(), annot=True,
                          cmap='magma', fmt='.2f', annot_kws={"size": 10})

    # Adjusting x-axis and y-axis labels
    plt.xticks(rotation=45, ha='right')  # Rotating x-axis labels by 45 degrees
    plt.yticks(rotation=0)  # Keeping y-axis labels horizontal

    plt.title('Correlation Heatmap of Indicators for India')
    plt.tight_layout()  # Adjust layout for better visualization
    plt.show()


# Call the function and pass the CSV file path as an argument
India_HeatMap('df_cleaned.csv')


# Heatmap for Mexico
def Mexico_HeatMap(Population):
    """
    Generate a heatmap showing the correlation between different indicators for Mexico.

    Args:
    - Population: str, path to the CSV file containing population data
    """
    data = pd.read_csv(Population)
    Mexico_data = data[data['Country Name'] == 'Mexico']

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
    Mexico_subset = Mexico_data[indicators]

    # Plotting the heatmap
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(Mexico_subset.corr(), annot=True,
                          cmap='magma', fmt='.2f', annot_kws={"size": 10})

    # Adjusting x-axis and y-axis labels
    plt.xticks(rotation=45, ha='right')  # Rotating x-axis labels by 45 degrees
    plt.yticks(rotation=0)  # Keeping y-axis labels horizontal

    plt.title('Correlation Heatmap of Mexico')
    plt.tight_layout()  # Adjust layout for better visualization
    plt.show()


# Call the function and pass the CSV file path as an argument
Mexico_HeatMap('df_cleaned.csv')
