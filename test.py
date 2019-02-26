import os
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import get_instance
import matplotlib.pyplot as plt
from numpy import array


def main(config, resume):
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2,
    )

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    rho_pred = []
    rho_orig = []
    T_pred = []
    T_orig = []
    mu_pred = []
    mu_orig = []
    cp_pred = []
    cp_orig = []
    psi_pred = []
    psi_orig = []
    alpha_pred = []
    alpha_orig = []
    as_pred = []
    as_orig = []
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            rho_pred.append(output[:, 0])
            rho_orig.append(target[:, 0])
            T_pred.append(output[:, 1])
            T_orig.append(target[:, 1])
            mu_pred.append(output[:, 2])
            mu_orig.append(target[:, 2])
            cp_pred.append(output[:, 3])
            cp_orig.append(target[:, 3])
            psi_pred.append(output[:, 4])
            psi_orig.append(target[:,4])
            alpha_pred.append(output[:, 5])
            alpha_orig.append(target[:, 5])
            as_pred.append(output[:, 6])
            as_orig.append(target[:, 6])
            #
            # save sample images, or do something with output here
            #
            
            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size
    
    rho_pred = [item for sublist in rho_pred for item in sublist]
    rho_orig = [item for sublist in rho_orig for item in sublist]
    T_pred = [item for sublist in T_pred for item in sublist]
    T_orig = [item for sublist in T_orig for item in sublist]
    mu_pred = [item for sublist in mu_pred for item in sublist]
    mu_orig = [item for sublist in mu_orig for item in sublist]
    cp_pred = [item for sublist in cp_pred for item in sublist]
    cp_orig = [item for sublist in cp_orig for item in sublist]
    psi_pred = [item for sublist in psi_pred for item in sublist]
    psi_orig = [item for sublist in psi_orig for item in sublist]
    alpha_pred = [item for sublist in alpha_pred for item in sublist]
    alpha_orig = [item for sublist in alpha_orig for item in sublist]
    as_pred = [item for sublist in as_pred for item in sublist]
    as_orig = [item for sublist in as_orig for item in sublist]
    
    fig, arr_sp = plt.subplots(2,4)
    arr_sp[0, 0].scatter(array(rho_orig), array(rho_pred), label = 'rho', c = 'r')
    arr_sp[0, 1].scatter(array(T_orig), array(T_pred), label = 'T', color = 'orange')
    arr_sp[0, 2].scatter(array(mu_orig), array(mu_pred), label = 'thermo:mu', color = 'yellow')
    arr_sp[0, 3].scatter(array(cp_orig), array(cp_pred), label = 'Cp', color = 'green')
    arr_sp[1, 0].scatter(array(psi_orig), array(psi_pred), label = 'thermo:psi', color = 'blue')
    arr_sp[1, 1].scatter(array(alpha_orig), array(alpha_pred), label = 'thermo:alpha', color = 'indigo')
    arr_sp[1, 2].scatter(array(as_orig), array(as_pred), label = 'thermo:as', color = 'violet')
    fig.delaxes(arr_sp[1, 3])
    fig.legend(loc='lower right', prop={'size': 22})
    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__ : total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    print(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default='saved/CombustionModel/0115_151609/model_best.pt', type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    main(config, args.resume)
