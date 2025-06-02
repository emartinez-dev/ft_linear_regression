"""Estimate car price from mileage using linear regression parameters from environment variables"""

import argparse
import os

from settings import T0_ENVAR_NAME, T1_ENVAR_NAME

def estimate_price(t0: float, t1: float, mileage: float) -> float:
    """
    Estimates the price of a car based on a linear regression model.

    :param t0: The intercept (theta 0) of the linear regression model.
    :type t0: float
    :param t1: The slope (theta 1) of the linear regression model.
    :type t1: float
    :param mileage: The mileage of the car for which to estimate the price.
    :type mileage: float
    :return: The estimated price of the car.
    :rtype: float
    """
    return t0 + t1 * mileage


def main(mileage: int) -> None:
    """
    Calculates and prints the estimated price of a car based on its mileage using linear regression
    parameters.

    :param mileage: The mileage of the car for which to estimate the price.
    :type mileage: int
    """
    t0 = float(os.getenv(T0_ENVAR_NAME, "0.0"))
    t1 = float(os.getenv(T1_ENVAR_NAME, "0.0"))

    print(estimate_price(t0, t1, mileage))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mileage",
        required=True,
        type=int,
        help="Mileage of the car whose price will be estimated",
    )
    args = parser.parse_args()
    try:
        main(args.mileage)

    except ValueError as err:
        print(f"ft_linear_regression: {err}")
        exit(1)
