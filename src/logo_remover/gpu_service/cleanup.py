"""Expiry maintenance independent of GPU/model/credentials, for operator schedulers."""
import argparse
from .config import load_config, ServiceConfig
from .store import Store


def cleanup_once(config: ServiceConfig):
    # Do not validate/load the model: expired private media must still be deletable
    # when weights, CUDA, license configuration or the API are unavailable.
    Store(config).maintenance()


def main():
    parser = argparse.ArgumentParser(description="Remove expired/cancelled private GPU job files; no model/API required")
    parser.parse_args()
    cleanup_once(load_config())
    print("Expiry maintenance completed")


if __name__ == "__main__":
    main()
