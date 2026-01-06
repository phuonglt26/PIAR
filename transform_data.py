import argparse

from src.get_data import get_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment")
    parser.add_argument(
        "--dataset",
        type=str,
        default="toys",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    get_data(dataset=args.dataset, mode=args.mode)
