import torch
import numpy as np
import wandb

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def train(itr, dataset, args, model, optimizer, device):
    model.train()
    features, labels, pairs_id = dataset.load_data(n_similar=args.num_similar)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, :np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    outputs = model(features, seq_len=seq_len, is_training=True, itr=itr, opt=args)
    total_loss, loss_dict = model.criterion(outputs, labels, seq_len=seq_len, device=device, opt=args,
                                            itr=itr, pairs_id=pairs_id, inputs=features)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if not args.without_wandb:
        if itr % 20 == 0 and itr != 0:
            wandb.log(loss_dict)

    return total_loss.data.cpu().numpy()
