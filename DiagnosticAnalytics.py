import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Contoh data dummy (ubah sesuai strukturmu jika perlu)
df = pd.DataFrame({
    'MPG': [21, 23, 19, 30, 25, 16, 22, 28, 17, 20, 35, 18],
    'Origin': ['Asia', 'Asia', 'Europe', 'Europe', 'USA', 'Asia', 'Europe', 'USA', 'USA', 'Asia', 'Europe', 'USA'],
    'Type': ['Sedan', 'Sports', 'Sedan', 'Sports', 'Sports', 'Sedan', 'Sedan', 'Sedan', 'Sports', 'Sports', 'Sedan', 'Sports']
})

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Origin', y='MPG', hue='Type', palette={'Sedan': 'blue', 'Sports': 'red'})
plt.title("Mileage by Type and Origin")
plt.ylabel("MPG (City)")
plt.xlabel("")
plt.show()
