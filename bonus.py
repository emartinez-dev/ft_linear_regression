import os

import matplotlib.pyplot as plt

from linear_regression import read_csv, get_minmax
from settings import CSV_PATH, T0_ENVAR_NAME, T1_ENVAR_NAME


intersect = float(os.getenv(T0_ENVAR_NAME, "0.0"))
slope = float(os.getenv(T1_ENVAR_NAME, "0.0"))

cars = read_csv(CSV_PATH)
min_km, max_km = get_minmax(cars)

x = [car.km for car in cars]
y = [car.price for car in cars]

y0 = slope * min_km + intersect
y1 = slope * max_km + intersect

fig, ax = plt.subplots()

# Scatter plot
ax.scatter(x, y, label="Data")

# Regression line
ax.axline(xy1=(min_km, y0), xy2=(max_km, y1), color="red", label=f"{slope}y + {intersect}")

plt.legend()
plt.xlabel("Mileage (km)")
plt.ylabel("Price")
plt.title("Regression representation")
plt.show()
