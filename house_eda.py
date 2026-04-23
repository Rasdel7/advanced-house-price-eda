import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load Dataset
df = pd.read_csv('train.csv')
print("Dataset loaded! Shape:", df.shape)
print("\nMissing values (top 10):")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# Basic Stats
print("\nBasic Statistics:")
print(df['SalePrice'].describe())

# Price Distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['SalePrice'], bins=50, color='#3498db', edgecolor='black')
plt.title('House Price Distribution', fontsize=13)
plt.xlabel('Sale Price')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.hist(np.log1p(df['SalePrice']), bins=50, color='#2ecc71', edgecolor='black')
plt.title('Log Price Distribution (Normalized)', fontsize=13)
plt.xlabel('Log Sale Price')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('price_distribution.png')
print("\nPrice distribution chart saved!")

# Correlation Heatmap — top 15 features only
numeric_df  = df.select_dtypes(include=[np.number])
corr        = numeric_df.corr()
top_features = corr['SalePrice'].abs().sort_values(ascending=False).head(15).index
corr_subset  = numeric_df[top_features].corr()

plt.figure(figsize=(13, 9))
mask = np.triu(np.ones_like(corr_subset, dtype=bool))
sns.heatmap(corr_subset, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Top 15 Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("Correlation heatmap saved!")

# Top features correlated with SalePrice
price_corr = corr['SalePrice'].drop('SalePrice').sort_values(ascending=False)
top_corr   = price_corr.head(10)

plt.figure(figsize=(10, 6))
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_corr.values]
plt.barh(top_corr.index, top_corr.values, color=colors, edgecolor='black')
plt.axvline(x=0, color='black', linewidth=0.8)
plt.title('Top 10 Features Correlated with Sale Price', fontsize=14)
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig('price_correlation.png')
print("Price correlation chart saved!")

# GrLivArea vs SalePrice
plt.figure(figsize=(10, 6))
plt.scatter(df['GrLivArea'], df['SalePrice'] / 1e3,
            alpha=0.3, color='#9b59b6', s=10)
plt.title('Living Area vs Sale Price', fontsize=14)
plt.xlabel('Above Ground Living Area (sqft)')
plt.ylabel('Sale Price (Thousands USD)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('area_vs_price.png')
print("Area vs price chart saved!")

# Overall Quality vs Average Price
plt.figure(figsize=(10, 6))
qual_price = df.groupby('OverallQual')['SalePrice'].mean() / 1e3
colors = sns.color_palette('RdYlGn', len(qual_price))
bars = plt.bar(qual_price.index, qual_price.values,
               color=colors, edgecolor='black')
for bar, val in zip(bars, qual_price.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 2,
             f'${val:.0f}K', ha='center', fontsize=8)
plt.title('Average Sale Price by Overall Quality Rating', fontsize=14)
plt.xlabel('Overall Quality (1=Poor, 10=Excellent)')
plt.ylabel('Average Sale Price (Thousands USD)')
plt.tight_layout()
plt.savefig('quality_vs_price.png')
print("Quality vs price chart saved!")

# Neighbourhood vs Average Price
plt.figure(figsize=(14, 6))
neigh_price = df.groupby('Neighborhood')['SalePrice'].mean().sort_values(
    ascending=False) / 1e3
colors = sns.color_palette('Blues_r', len(neigh_price))
plt.bar(neigh_price.index, neigh_price.values,
        color=colors, edgecolor='black')
plt.title('Average Sale Price by Neighbourhood', fontsize=14)
plt.xlabel('Neighbourhood')
plt.ylabel('Average Sale Price (Thousands USD)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('neighbourhood_vs_price.png')
print("Neighbourhood chart saved!")

# Outlier Detection
Q1       = df['SalePrice'].quantile(0.25)
Q3       = df['SalePrice'].quantile(0.75)
IQR      = Q3 - Q1
outliers = df[(df['SalePrice'] < Q1 - 1.5 * IQR) |
              (df['SalePrice'] > Q3 + 1.5 * IQR)]

plt.figure(figsize=(10, 5))
plt.boxplot(df['SalePrice'] / 1e3, vert=False,
            patch_artist=True,
            boxprops=dict(facecolor='#3498db', alpha=0.7))
plt.title('Sale Price Boxplot - Outlier Detection', fontsize=14)
plt.xlabel('Sale Price (Thousands USD)')
plt.tight_layout()
plt.savefig('outliers.png')
print("Outlier chart saved!")

# Key Insights
print("\nKey Insights:")
print(f"  Total houses          : {len(df):,}")
print(f"  Average price         : ${df['SalePrice'].mean():,.0f}")
print(f"  Median price          : ${df['SalePrice'].median():,.0f}")
print(f"  Most correlated feat  : {price_corr.index[0]}")
print(f"  Outliers detected     : {len(outliers):,}")
print(f"  Price range           : ${df['SalePrice'].min():,} - ${df['SalePrice'].max():,}")
print("\nEDA Complete! 7 charts saved.")