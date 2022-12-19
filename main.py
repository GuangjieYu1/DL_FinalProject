import argparse

from Trainer import Trainer
from tboard import initiateTensorboard


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Configuration File Path')
	parser.add_argument("-c", "--conf", action="store", dest="conf_file",help="Path to config file")
	parser.add_argument("-tb", "--tensorboard", action="store_true", dest="tb_flag",help="tensorboard flag")
	parser.add_argument("-tbpth", "--tensorboard_path", action="store", dest="tb_path",help="tensorboard flag")
	args = parser.parse_args()

	initiateTensorboard(args.tb_flag,args.tb_path)


	conf_path =  args.conf_file
	if conf_path is None:
		conf_path = 'configs/config.yaml'
	Trainer(conf_path).train()
