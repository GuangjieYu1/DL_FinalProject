import yaml
import sys

class Config(object):

    def __init__(self,config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.__data = yaml.safe_load(f)

    def __call__(self):
        return self.__data

if __name__ == "__main__":
    c = Config()
    print(c())
