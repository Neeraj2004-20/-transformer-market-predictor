from train import train, load_config

if __name__ == "__main__":
    config = load_config()
    train(config)
