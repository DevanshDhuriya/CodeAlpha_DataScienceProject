import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. Load the dataset
csv_path = Path(__file__).resolve().parent / 'Unemployment in India.csv'
df = pd.read_csv(csv_path)

# 2. Clean column names by stripping whitespaces
df.columns = df.columns.str.strip()

# 3. Drop completely null rows
df_clean = df.dropna().copy()

# 4. Standardize text values and clean whitespaces in categories
df_clean['Frequency'] = df_clean['Frequency'].str.strip()
df_clean['Area'] = df_clean['Area'].str.strip()
df_clean['Region'] = df_clean['Region'].str.strip()

# 5. Convert the Date column to datetime format
df_clean['Date'] = pd.to_datetime(df_clean['Date'].str.strip(), format='%d-%m-%Y')

print(f"Data successfully cleaned. Rows retained: {df_clean.shape[0]}")