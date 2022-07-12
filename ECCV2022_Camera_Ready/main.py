from __future__ import print_function

import os
import random
import time

import numpy as np
import torch
import wandb
from tqdm import tqdm

import model
import options
import wsad_dataset
from test import test
from train import train

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


import torch.optim as optim

if __name__ == '__main__':
    args = options.parser.parse_args()

    if not args.without_wandb:
        wandb.init(
            name=time.asctime()[:-4] + args.model_name,
            config=args,
            project=f"DELU_{args.dataset}",
            sync_tensorboard=True)

    seed = args.seed
    print('=============seed: {}, pid: {}============='.format(seed, os.getpid()))
    setup_seed(seed)
    device = torch.device("cuda")
    dataset = getattr(wsad_dataset, args.dataset)(args)
    if 'Thumos' in args.dataset_name:
        max_map = [0] * 9
    else:
        max_map = [0] * 10
    log_model_path = os.path.join(args.path_dataset, 'logs', args.model_name)
    ckpt_path = os.path.join(args.path_dataset, 'ckpt')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    print(args)
    model = getattr(model, args.use_model)(dataset.feature_size, dataset.num_class, opt=args).to(device)

    if args.pretrained_ckpt is not None:
        model.load_state_dict(torch.load(args.pretrained_ckpt))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_loss = 0
    lrs = [args.lr, args.lr / 5, args.lr / 5 / 5]
    print(model)
    for itr in tqdm(range(args.max_iter)):
        loss = train(itr, dataset, args, model, optimizer, device)
        total_loss += loss
        if itr % args.interval == 0 and not itr == 0:
            print('Iteration: %d, Loss: %.5f' % (itr, total_loss / args.interval))
            total_loss = 0
            torch.save(model.state_dict(), ckpt_path + '/last_' + args.model_name + '.pkl')
            iou, dmap = test(itr, dataset, args, model, device)
            if 'Thumos' in args.dataset_name:
                cond = sum(dmap[:7]) > sum(max_map[:7])
            else:
                cond = np.mean(dmap) > np.mean(max_map)
            if cond:
                torch.save(model.state_dict(), ckpt_path + '/best_' + args.model_name + '.pkl')
                max_map = dmap

            if not args.without_wandb:
                wandb.log({'MAX mAP Avg 0.1-0.7': np.mean(max_map[:7]) * 100})

            print('||'.join(['MAX map @ {} = {:.3f} '.format(iou[i], max_map[i] * 100) for i in range(len(iou))]))
            max_map = np.array(max_map)
            print('mAP Avg 0.1-0.5: {}, mAP Avg 0.1-0.7: {}, mAP Avg ALL: {}'.format(np.mean(max_map[:5]) * 100,
                                                                                     np.mean(max_map[:7]) * 100,
                                                                                     np.mean(max_map) * 100))
            print("------------------pid: {}--------------------".format(os.getpid()))
