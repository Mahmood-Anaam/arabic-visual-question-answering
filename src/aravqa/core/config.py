import yaml
import argparse

class Config:
    def __init__(self, config_file=None, **kwargs):
        self.config = {}
        if config_file:
            self.load_from_file(config_file)
        self.load_from_kwargs(**kwargs)

    def load_from_file(self, config_file):
        with open(config_file, 'r', encoding='utf-8') as file:
            self.config.update(yaml.safe_load(file))

    def load_from_kwargs(self, **kwargs):
        self.config.update(kwargs)

    def get(self, key, default=None):
        return self.config.get(key, default)

def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for Arabic Visual Question Answering")
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--param1', type=str, help='Parameter 1')
    parser.add_argument('--param2', type=str, help='Parameter 2')
    # Add more parameters as needed
    return parser.parse_args()

def load_config(config_file=None, **kwargs):
    args = parse_args()
    config_file = config_file or args.config
    config = Config(config_file=config_file, **kwargs)
    return config

if __name__ == "__main__":
    args = parse_args()
    config = load_config(config_file=args.config, param1=args.param1, param2=args.param2)
    # Example usage
    print(config.get('param1'))
    print(config.get('param2'))