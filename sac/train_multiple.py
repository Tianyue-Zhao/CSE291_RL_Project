import sys
import getopt
import os

# get the input of which training script to run


def main(argv):
    train_script = sys.argv[1]
    print("Training script: {}".format(train_script))

    # use predefined seeds
    seeds = [1, 2, 3]

    for seed in seeds:
        # run the training script with the seed on the command line
        print("Training seed: {}".format(seed))
        print("python3 " + train_script + " " + str(seed))
        os.system("python3 " + train_script + " " + str(seed))


if __name__ == "__main__":
    main(sys.argv[1:])
