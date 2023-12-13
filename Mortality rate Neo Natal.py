
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