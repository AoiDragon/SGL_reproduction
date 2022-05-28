import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')

    # for gcn
    parser.add_argument('--embed_dim', default=64, type=int)
    parser.add_argument('--layer_num', default=3, type=int)

    # for ssl
    parser.add_argument('--SSL_reg', default=0.1, type=float)
    parser.add_argument('--SSL_dropout_ratio', default=0.1, type=float)
    parser.add_argument('--SSL_temp', default=0.2, type=float)

    # for train
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--epoch_num', default=500, type=int)
    parser.add_argument('--stop_cnt', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--reg', default=0.0001, type=float)

    # for test
    parser.add_argument('--k', default=20, type=int)

    # for save and read
    parser.add_argument('--train_data_path', default='./dataset/yelp2018/yelp2018.train', type=str)
    parser.add_argument('--test_data_path', default='./dataset/yelp2018/yelp2018.test', type=str)

    return parser.parse_args()


args = parse_args()
