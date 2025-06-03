"""Bonus: Visualizing the results of a linear regression model applied to car mileage and price."""

import os

import matplotlib.pyplot as plt

from linear_regression import read_csv, get_minmax
from settings import CSV_PATH, T0_ENVAR_NAME, T1_ENVAR_NAME


def main() -> None:
    """
    Gets the t0 and t1 values, and draws a graph with the line over the cars data in the CSV.
    """
    t0 = float(os.getenv(T0_ENVAR_NAME, "0.0"))
    t1 = float(os.getenv(T1_ENVAR_NAME, "0.0"))

    cars = read_csv(CSV_PATH)
    min_km, max_km = get_minmax(cars)

    x = [car.km for car in cars]
    y = [car.price for car in cars]

    y0 = t1 * min_km + t0
    y1 = t1 * max_km + t0

    fig, ax = plt.subplots()

    # Scatter plot
    ax.scatter(x, y, label="Cars")

    # Regression line
    ax.axline(xy1=(min_km, y0), xy2=(max_km, y1), color="red", label=f"{t1:.2f}y + {t0:.2f}")

    plt.legend()
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("ft-linear-regression ")
    plt.show()

if __name__ == "__main__":
    main()
