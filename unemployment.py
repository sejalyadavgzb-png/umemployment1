# ============================================
# UNEMPLOYMENT ANALYSIS IN INDIA USING PYTHON
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------
# Load Dataset
# --------------------------------------------
df = pd.read_csv("Unemployment in India.csv")

# --------------------------------------------
# Data Cleaning
# --------------------------------------------
df.columns = df.columns.str.strip()
df.dropna(inplace=True)

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Rename columns for convenience
df.rename(columns={
    'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour_Participation_Rate'
}, inplace=True)

print("Dataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# --------------------------------------------
# Monthly Unemployment Trend
# --------------------------------------------
monthly_unemployment = df.groupby(
    df['Date'].dt.to_period('M')
)['Unemployment_Rate'].mean()

monthly_unemployment.index = monthly_unemployment.index.astype(str)

plt.figure(figsize=(12,6))
sns.lineplot(
    x=monthly_unemployment.index,
    y=monthly_unemployment.values,
    marker='o'
)
plt.xticks(rotation=45)
plt.title("Monthly Average Unemployment Rate in India")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# --------------------------------------------
# State-wise Average Unemployment Rate
# --------------------------------------------
state_unemployment = (
    df.groupby('Region')['Unemployment_Rate']
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12,8))
sns.barplot(
    x=state_unemployment.values,
    y=state_unemployment.index
)
plt.title("Average Unemployment Rate by State")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# --------------------------------------------
# Urban vs Rural Unemployment
# --------------------------------------------
plt.figure(figsize=(8,5))
sns.barplot(
    x='Area',
    y='Unemployment_Rate',
    data=df
)
plt.title("Urban vs Rural Unemployment Rate")
plt.xlabel("Area")
plt.ylabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# --------------------------------------------
# COVID-19 Impact Analysis
# --------------------------------------------
pre_covid = df[df['Date'] < '2020-03-01']
during_covid = df[df['Date'] >= '2020-03-01']

pre_avg = pre_covid['Unemployment_Rate'].mean()
covid_avg = during_covid['Unemployment_Rate'].mean()

covid_df = pd.DataFrame({
    'Period': ['Pre-COVID', 'During COVID'],
    'Average_Unemployment': [pre_avg, covid_avg]
})

plt.figure(figsize=(7,5))
sns.barplot(
    x='Period',
    y='Average_Unemployment',
    data=covid_df
)
plt.title("Impact of COVID-19 on Unemployment")
plt.ylabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# --------------------------------------------
# Correlation Heatmap
# --------------------------------------------
plt.figure(figsize=(8,6))
sns.heatmap(
    df[['Unemployment_Rate',
        'Employed',
        'Labour_Participation_Rate']]
    .corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# --------------------------------------------
# Distribution of Unemployment Rate
# --------------------------------------------
plt.figure(figsize=(8,5))
sns.histplot(
    df['Unemployment_Rate'],
    bins=20,
    kde=True
)
plt.title("Distribution of Unemployment Rate")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# --------------------------------------------
# Findings
# --------------------------------------------
print("\n===== KEY INSIGHTS =====")
print(f"Average Unemployment Before COVID: {pre_avg:.2f}%")
print(f"Average Unemployment During COVID: {covid_avg:.2f}%")

highest_state = state_unemployment.idxmax()
highest_rate = state_unemployment.max()

print(f"\nHighest Average Unemployment State: {highest_state}")
print(f"Average Rate: {highest_rate:.2f}%")

print("\nCOVID-19 caused a significant rise in unemployment.")
print("Urban and rural areas were both affected.")
print("Certain states experienced much higher unemployment rates than others.")

# ============================================
# END OF PROJECT
# ============================================