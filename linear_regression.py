"""Trains a linear regression model on car price data using gradient descent"""

import argparse
from csv import reader
from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Tuple

from predict import estimate_price
from settings import CSV_PATH, T0_ENVAR_NAME, T1_ENVAR_NAME


@dataclass
class Car:
    """
    Represents a car with its distance traveled in kilometers and its price.

    :param km: The distance the car has traveled in kilometers.
    :type km: float
    :param price: The price of the car.
    :type price: float
    """

    km: float
    price: float

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

        return [Car(km=float(car[0]), price=float(car[1])) for car in cars_csv]


def normalize(n: float, min_range: float, max_range: float) -> float:
    """
    Normalizes a value to a range between 0 and 1 based on the minimum and maximum values.

    :param n: The value to normalize.
    :type n: float
    :param min_range: The minimum value.
    :type min_range: float
    :param max_range: The maximum value.
    :type max_range: float
    :return: The normalized value in the range [0, 1].
    :rtype: float
    """
    return (n - min_range) / (max_range - min_range)


def denormalize_theta(
    t0: float, t1: float, min_range: float, max_range: float
) -> Tuple[float, float]:
    """
    Denormalizes theta values after calculating them with normalized 'km' values.

    :param t0: The intercept (theta 0) calculated on normalized data.
    :type t0: float
    :param t1: The slope (theta 1) calculated on normalized data.
    :type t1: float
    :param min_range: The minimum value of 'km' before normalization.
    :type min_range: float
    :param max_range: The maximum value of 'km' before normalization.
    :type max_range: float
    :return: The denormalized theta values (t0, t1).
    :rtype: tuple
    """
    t1_denorm = t1 / (max_range - min_range)
    t0_denorm = t0 - (t1_denorm * min_range)
    return t0_denorm, t1_denorm


def get_minmax(cars: List[Car]) -> Tuple[float, float]:
    """
    Get the minimum and maximum 'km' values from a list of Car objects.

    :param cars: List of Car objects to analyze.
    :type cars: List[Car]
    :returns: A tuple containing the minimum and maximum 'km' values.
    :rtype: Tuple[float, float]
    """
    min_km = min(car.km for car in cars)
    max_km = max(car.km for car in cars)
    return (min_km, max_km)


def normalize_cars(cars: List[Car]) -> List[Car]:
    """
    Normalizes cars 'km' in the list using min-max normalization.

    :param cars: List of Car objects to be normalized.
    :type cars: List[Car]
    :return: List of Car objects with normalized 'km' values.
    :rtype: List[Car]
    """
    min_km, max_km = get_minmax(cars)
    return [Car(km=normalize(car.km, min_km, max_km), price=car.price) for car in cars]


def main(learning_rate: float, epochs: int, normalize_km: bool) -> None:
    """
    Trains a simple linear regression model using gradient descent on car data.
    This function reads car data from a CSV file and performs one epoch of gradient descent.

    :param learning_rate: The learning rate for gradient descent.
    :type learning_rate: float
    :param epochs: The number of full dataset iterations for gradient descent.
    :type epochs: int
    :param normalize_km: Whether to normalize the km variable so training doesn't converge
    :type normalize_km: bool
    """

    cars = read_csv(Path(CSV_PATH))
    n_cars = len(cars)
    min_km, max_km = get_minmax(cars)

    if normalize_km:
        cars = normalize_cars(cars)

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

        temp_t0 = learning_rate * (1 / n_cars) * accum_error_t0
        temp_t1 = learning_rate * (1 / n_cars) * accum_error_t1

        t0 -= temp_t0
        t1 -= temp_t1

        if math.isnan(t0) or math.isnan(t1):
            print(f"On epoch {_}, {t0=}, {t1=}, {temp_t0=}, {temp_t1=}")
            return

    if normalize_km:
        t0, t1 = denormalize_theta(t0, t1, min_km, max_km)

    print(
        "Optimized theta values:",
        f"- t0: {t0}",
        f"- t1: {t1}\n",
        "Export them for the 'predictor' with:",
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
    parser.add_argument(
        "-n",
        "--normalize",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to normalize the km variable so training converges faster",
    )
    args = parser.parse_args()

    try:
        main(args.learning_rate, args.epochs, args.normalize)

    except Exception as err:
        print(f"ft_linear_regression: {err}")
        exit(1)
