"""Trains a linear regression model on car price data using gradient descent"""

import argparse
from csv import reader
from dataclasses import dataclass
import math
from pathlib import Path
from typing import List

from predict import estimate_price
from settings import CSV_PATH, T0_ENVAR_NAME, T1_ENVAR_NAME


@dataclass
class Car:
    """
    Represents a car with its distance traveled in kilometers and its price.

    :param km: The distance the car has traveled in kilometers.
    :type km: int
    :param price: The price of the car.
    :type price: int
    """

    km: int
    price: int

    def __repr__(self) -> str:
        return f"Car - km: {self.km} -> price: {self.price}"


def read_csv(path: Path) -> List[Car]:
    """
    Reads a CSV file containing car data and returns a list of Car objects.

    :param path: The file path of the CSV file that will be read
    :type path: Path
    :return: The list of Cars parsed from the CSV
    :rtype: List[Car]
    """

    with open(path, "r", encoding="utf8") as csv_file:
        cars_csv = reader(csv_file)
        headings = next(cars_csv)

        assert headings[0] == "km", "First column must be 'km'"
        assert headings[1] == "price", "Second column must be 'price'"

        return [Car(km=int(car[0]), price=int(car[1])) for car in cars_csv]


def main(learning_rate: float, epochs: int) -> None:
    """
    Trains a simple linear regression model using gradient descent on car data.
    This function reads car data from a CSV file and performs one epoch of gradient descent.

    :param learning_rate: The learning rate for gradient descent.
    :type learning_rate: float
    :param epochs: The number of epochs (data points) to use for normalization in the update step.
    :type epochs: int
    """

    cars = read_csv(Path(CSV_PATH))

    t0 = 0.0
    t1 = 0.0

    for _ in range(epochs):
        accum_error_t0 = 0.0
        accum_error_t1 = 0.0

        for car in cars:
            error_t0 = estimate_price(t0, t1, car.km) - car.price
            error_t1 = error_t0 * car.km
            accum_error_t0 += error_t0
            accum_error_t1 += error_t1

        temp_t0 = learning_rate * (1 / epochs) * accum_error_t0
        temp_t1 = learning_rate * (1 / epochs) * accum_error_t1

        t0 -= temp_t0
        t1 -= temp_t1

        if math.isnan(t0) or math.isnan(t1):
            print(f"On epoch {_}, {t0=}, {t1=}, {temp_t0=}, {temp_t1=}")
            return

    print(
        "Optimized theta values:",
        f"- t0: {t0}",
        f"- t1: {t1}\n",
        "Export them with:",
        f"export {T0_ENVAR_NAME}={t0}",
        f"export {T1_ENVAR_NAME}={t1}",
        sep="\n",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--learning_rate",
        required=True,
        type=float,
        help="The learning rate for gradient descent",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        required=True,
        type=int,
        help="The number of epochs (data points) to use for normalization in the update step",
    )
    args = parser.parse_args()

    try:
        main(args.learning_rate, args.epochs)

    except Exception as err:
        print(f"ft_linear_regression: {err}")
        exit(1)
