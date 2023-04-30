import argparse

from Architecture.models.learner import Learner
from config.config import read_config


def main():
    parser = argparse.ArgumentParser(description='Train controller.')
    parser.add_argument('--config_file', help='Path to config yaml', required=True)

    args = parser.parse_args()
    config_filepath = args.config_file

    config = read_config(config_filepath)
    trainer = Learner(config)
    trainer.train()

if __name__ == '__main__':
    main()
