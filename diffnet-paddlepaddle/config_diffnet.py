import argparse

def get_params():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--data_dir", type=str, default='/home/aistudio/yelp/data/yelp')
    parser.add_argument("--links_filename", type=str, default='/home/aistudio/yelp/data/yelp/yelp.links')
    parser.add_argument("--user_review_vector_matrix", type=str, default='/home/aistudio/yelp/data/yelp/user_vector.npy')
    parser.add_argument("--item_review_vector_matrix", type=str, default='/home/aistudio/yelp/data/yelp/item_vector.npy')

    # data
    parser.add_argument("--num_users", type=int, default=17237)
    parser.add_argument("--num_items", type=int, default=38342)
    parser.add_argument("--gpu_device", type=int, default=1)
    
    parser.add_argument("--data_name", type=str, default='yelp')
    parser.add_argument("--model_name", type=str, default='diffnet')

    # model 
    parser.add_argument("--review_feature_dim", type=int, default=150)
    parser.add_argument("--gnn_dim", type=int, default=32)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--num_negatives", type=int, default=8)
    parser.add_argument("--num_evaluate", type=int, default=1000)
    parser.add_argument("--num_procs", type=int, default=16)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--evaluate_batch_size", type=int, default=2560)
    parser.add_argument("--training_batch_size", type=int, default=512)
    parser.add_argument("--epoch_notice", type=int, default=300)

    args, _ = parser.parse_known_args()
    return args