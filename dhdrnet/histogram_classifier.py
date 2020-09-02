from argparse import Namespace
import numpy as np
import pandas as pd
import argparse


def main(args:Namespace):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate statistics for histogram classifier"
    )

    args = parser.parse_args()
    main(args)
