import argparse

parser = argparse.ArgumentParser(description='DELU')
parser.add_argument('--path_dataset', type=str, default='path/to/Thumos14', help='the path of data feature')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate (default: 0.0001)')
parser.add_argument('--batch_size', type=int, default=10, help='number of instances in a batch of data (default: 10)')
parser.add_argument('--model_name', default='weakloc', help='name to save model')
parser.add_argument('--pretrained_ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--feature_size', default=2048, help='size of feature (default: 2048)')
parser.add_argument('--num_class', type=int, default=20, help='number of classes (default: )')
parser.add_argument('--dataset_name', default='Thumos14reduced', help='dataset to train on (default: )')
parser.add_argument('--max_seqlen', type=int, default=320,
                    help='maximum sequence length during training (default: 750)')
parser.add_argument('--num_similar', default=3, type=int,
                    help='number of similar pairs in a batch of data  (default: 3)')
parser.add_argument('--seed', type=int, default=3552, help='random seed (default: 1)')
parser.add_argument('--max_iter', type=int, default=20000, help='maximum iteration to train (default: 50000)')
parser.add_argument('--feature_type', type=str, default='I3D',
                    help='type of feature to be used I3D or UNT (default: I3D)')
parser.add_argument('--use_model', type=str, default='DELU', help='model used to train the network')
parser.add_argument('--interval', type=int, default=200, help='time interval of performing the test')
parser.add_argument('--similar_size', type=int, default=2)

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dataset', type=str, default='SampleDataset')
parser.add_argument('--proposal_method', type=str, default='multiple_threshold_hamnet')

# for proposal genration
parser.add_argument('--scale', type=float, default=1)
parser.add_argument("--feature_fps", type=int, default=25)
parser.add_argument('--gamma_oic', type=float, default=0.2)

parser.add_argument('--k', type=float, default=7)
# for testing time usage
parser.add_argument("--topk2", type=float, default=10)
parser.add_argument("--topk", type=float, default=60)

parser.add_argument('--dropout_ratio', type=float, default=0.7)
parser.add_argument('--reduce_ratio', type=int, default=16)
# for pooling kernel size calculate
parser.add_argument('--t', type=int, default=5)

# -------------loss weight---------------
parser.add_argument("--alpha1", type=float, default=0.8)
parser.add_argument("--alpha2", type=float, default=0.8)  # origin guide_loss coefficient
parser.add_argument("--alpha3", type=float, default=1)
parser.add_argument('--alpha4', type=float, default=1)

parser.add_argument("--AWM", type=str, default='BWA_fusion_dropout_feat_v2')

parser.add_argument("--without_wandb", action='store_true')
parser.add_argument("--alpha_edl", type=float, default=1)
parser.add_argument("--rat_atn", type=int, default=9)
parser.add_argument("--amplitude", type=float, default=0.7)
parser.add_argument("--alpha_uct_guide", type=float, default=0.4)
