import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data dummy untuk GPA
np.random.seed(42)
x = np.random.uniform(2, 4, 100)
y = 0.8 * x + np.random.normal(0, 0.2, 100)

df = pd.DataFrame({'High School GPA': x, 'University GPA': y})

sns.lmplot(data=df, x='High School GPA', y='University GPA', line_kws={'color': 'red'})
plt.title("Linear Regression: High School GPA vs University GPA")
plt.show()
