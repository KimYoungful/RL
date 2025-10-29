
import argparse
from src.utils.logging import get_logger
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',help='path to config file')
    # parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'test'])
    
    return parser.parse_args()

def main():
    print("Hello, world!")
    

    args = parse_args()
    logger = get_logger()
    logger.info(f"Running {args.mode} with config {args.config}")

    if args.mode == 'train':
        # train code here
        pass

if __name__ == '__main__':
    main()