import matplotlib.pyplot as plt

# Posisi fitur
features = {
    'Standard Reporting': (1, 1),
    'Descriptive Statistics': (2, 1.2),
    'Data Visualization': (3, 1.4),
    'Data Query': (2.5, 0.9),
    'Data Mining': (4, 2),
    'Forecasting': (5, 2.5),
    'Predictive Modeling': (5.2, 2.7),
    'Simulation': (6, 3),
    'Decision Analysis': (6.5, 3.5),
    'Optimization': (7, 4)
}

plt.figure(figsize=(10, 6))

for name, (x, y) in features.items():
    plt.scatter(x, y, s=100)
    plt.text(x+0.1, y, name, fontsize=9)

plt.title("Excel Usability across Spectrum of Analytics")
plt.xlabel("Degree of Complexity")
plt.ylabel("Competitive Advantage")
plt.grid(True)
plt.tight_layout()
plt.show()
