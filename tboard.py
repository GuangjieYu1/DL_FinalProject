from tensorboard import program

def initiateTensorboard(startFlag=True,log_path='./logs'):
    if startFlag is True:
        log_path = './logs' if log_path is None else log_path
        print(startFlag,log_path)
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_path])
        url = tb.launch()
        print(f"Tensorflow listening on {url}")


if __name__ == "__main__":
    initiateTensorboard()
