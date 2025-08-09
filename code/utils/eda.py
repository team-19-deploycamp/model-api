import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

df = pd.read_csv('../../dataset/places.csv')
ratings = pd.read_csv('../../dataset/ratings.csv')

print(df.head())
print(ratings['Place_Ratings'].describe())

# Hitung jumlah Place_Name unik
unique_place_names = df['Place_Name'].nunique()
print(f"Jumlah Place_Name yang unik: {unique_place_names}\n")

# Cetak semua kategori unik
categories = df['Category'].unique()
print(f"Kategori unik: {categories}\n")

# Untuk setiap kategori, cetak jumlah data dan 10 data acak (hanya id dan Place_Name)
for category in categories:
    subset = df[df['Category'] == category]
    print(f"Kategori: {category}")
    print(f"Jumlah data: {len(subset)}")
    print("10 data acak (id dan Place_Name):")
    print(subset[['Place_Id', 'Place_Name']].sample(n=min(10, len(subset)), random_state=42))
    print("="*80)