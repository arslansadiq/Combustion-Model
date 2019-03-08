import torch
import os
import argparse
#import data_loader.data_loaders as module_data
#import model.loss as module_loss
#import model.metric as module_metric
import model.model as module_arch
from train import get_instance

def main(config, resume):
    '''data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2,
    )'''
    model = get_instance(module_arch, 'arch', config)
    #model.summary()
    #loss_fn = getattr(module_loss, config['loss'])
    #metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    example = torch.rand(1, 2)
    traced_script_module = torch.jit.trace(model, example)
    output = traced_script_module(torch.tensor([[-0.8037, -0.2691]]))
    print(output)
    traced_script_module.save("model.pt")
    

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
