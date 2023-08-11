'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    2022 Benjamin Kellenberger
'''

import os
import argparse
import yaml
import glob
from tqdm import trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD

# let's import our own classes and functions!
from util import init_seed
from dataset import CTDataset
from model import CustomResNet18
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import class_weight
import numpy as np
import wandb

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy
import torch.distributed as dist

def get_class_weights(cfg):
    ''' 
        Get number of instances per class in the train and val set 
        in order to do weighted cross entropy loss
    '''
    classes_for_weighting_train_and_val = []
    
    
    for split in ['train', 'val']:
        dataset_instance = CTDataset(cfg, split)        # create an object instance of our CTDataset class

        dataLoader = DataLoader(
                dataset=dataset_instance,
                batch_size=32,
                shuffle=False,
                pin_memory=True,
                num_workers=2,
            )

        ### compute weights for class balancing
        classes_for_weighting = []
        
        for data, labels in dataLoader:
            classes_for_weighting.extend(list(labels.numpy()))  

        class_weights=class_weight.compute_class_weight('balanced',classes = np.unique(classes_for_weighting),y = np.array(classes_for_weighting))
        class_weights = class_weights/np.sum(class_weights)
        class_weights=torch.tensor(class_weights,dtype=torch.float32)
        
        classes_for_weighting_train_and_val.append(class_weights)
    
    return classes_for_weighting_train_and_val[0], classes_for_weighting_train_and_val[1]
    


def create_dataloader(cfg, split='train'):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = CTDataset(cfg, split)        # create an object instance of our CTDataset class

    #device = cfg['device']

    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            shuffle=False,
            pin_memory=True,
            num_workers=cfg['num_workers'],
            sampler=DistributedSampler(dataset_instance)
        )
    
    ### compute weights for class balancing
#     classes_for_weighting = []
#     for data, labels in dataLoader:
#         classes_for_weighting.extend(list(labels.numpy()))  

#     class_weights=class_weight.compute_class_weight('balanced',classes = np.unique(classes_for_weighting),y = np.array(classes_for_weighting))
#     class_weights = class_weights/np.sum(class_weights)
#     class_weights=torch.tensor(class_weights,dtype=torch.float).to(device)

    #return dataLoader, class_weights
    return dataLoader


def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CustomResNet18(cfg['num_classes'])         # create an object instance of our CustomResNet18 class

    # load latest model state
#     model_states = glob.glob('model_states/*.pt')
#     if len(model_states):
#         # at least one save state found; get latest
#         model_epochs = [int(m.replace('model_states/','').replace('.pt','')) for m in model_states]
#         start_epoch = max(model_epochs)

#         # load state dict and apply weights to model
#         print(f'Resuming from epoch {start_epoch}')
#         state = torch.load(open(f'model_states/{start_epoch}.pt', 'rb'), map_location='cpu')
#         model_instance.load_state_dict(state['model'])

#     else:
#         # no save state found; start anew
#         print('Starting new model')
#         start_epoch = 0
    start_epoch = 0

    return model_instance, start_epoch

def load_pretrained_weights_for_finetuning(cfg, model, custom_weights=None):
    device = cfg['device']
    if custom_weights:
        state = torch.load(open(custom_weights, 'rb'), map_location=device)
        pretrained_dict = state['state_dict']
        model_dict = model.state_dict()

        ################## method from https://github.com/sthalles/SimCLR/blob/simclr-refactor/feature_eval/mini_batch_logistic_regression_evaluator.ipynb #################
        
        # for k in list(pretrained_dict.keys()):
        #     if k.startswith('backbone.'):
        #         if k.startswith('backbone') and not k.startswith('backbone.fc'):
        #             # removes prefix i.e backbone.conv1.weight becomes conv1.weight
        #             pretrained_dict[k[len("backbone."):]] = pretrained_dict[k]
        #     del pretrained_dict[k]

        # # remove prefixes for our model too so that the keys match
        # for k in list(model_dict.keys()):
        #     starts_with = k.split('.')[0]
        #     model_dict[k[len(starts_with)+1:]] = model_dict[k]
        #     del model_dict[k]
        # #import ipdb;ipdb.set_trace()
        # model.load_state_dict(model_dict)
        # log = model.load_state_dict(pretrained_dict, strict=False)
        # import ipdb;ipdb.set_trace()

        # assert log.missing_keys == ['fc.weight', 'fc.bias']

        # import ipdb;ipdb.set_trace()
        # # freeze all layers but the last fc
        # for name, param in model.named_parameters():
        #     if name not in ['fc.weight', 'fc.bias']:
        #         param.requires_grad = False

        # parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        # assert len(parameters) == 2  # fc.weight, fc.bias



        
        # ###########################################################################################################################################################################

        
        ################### my method of loading weights ###########################################################################################################################
        # only update the weights of layers with the same names and also don't update the last layer because of size mismatch between num classes
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in ['classifier.weight', 'classifier.bias']}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        ## freeze all layers but the last fc - finetuning only
        for name, param in model.named_parameters():
            if name not in ['classifier.weight', 'classifier.bias']:
                param.requires_grad = False

        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias


        ## we need to copy everything except the last layer
        #for key in state['model'].keys():
            #if not(key == 'classifier.weight' or key== 'classifier.bias'):
            ###### Tarun : LESSON LEARNT : THE LINE BELOW DOES NOT WORK FOR SOME REASON ..you must do it like mentioned in the most liked answer here https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2 ###
            #model.state_dict()[key] = state['model'][key]
        #model.load_state_dict(state['model'])
        ##import ipdb;ipdb.set_trace()

        ##############################################################################################################################################################################

    return model



def save_model(cfg, epoch, model, stats):
    # make sure save directory exists; create if not
    os.makedirs('model_states', exist_ok=True)

    # get model parameters and add to stats...
    stats['model'] = model.module.state_dict()

    # ...and save
    torch.save(stats, open(f'model_states/{epoch}.pt', 'wb'))
    
    # also save config file if not present
    cfpath = 'model_states/config.yaml'
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)

            

def setup_optimizer(cfg, model):
    '''
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    '''
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer



def train(cfg, dataLoader, class_weights_train, model, optimizer, metric, rank):
    '''
        Our actual training function.
    '''

    device = rank

    # put model on device
    
    # put the model into training mode
    # this is required for some layers that behave differently during training
    # and validation (examples: Batch Normalization, Dropout, etc.)
    model.train()
    
    ####### TARUN LESSON LEARNT : class weights needs to be a float32 and not a float64 ####################
    class_weights_train = class_weights_train.to(device, dtype=torch.float32)
    
    # loss function
    criterion = nn.CrossEntropyLoss(class_weights_train)
    #criterion = nn.CrossEntropyLoss()
    
    # running averages
    loss_total, oa_total = 0.0, 0.0                         # for now, we just log the loss and overall accuracy (OA)
    
    all_predicted_labels = []
    all_ground_truth_labels = []

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    for idx, (data, labels) in enumerate(dataLoader):       # see the last line of file "dataset.py" where we return the image tensor (data) and label

        # put data and labels on device
        data, labels = data.to(device), labels.to(device)

        # forward pass
        prediction = model(data)

        # reset gradients to zero
        optimizer.zero_grad()

        # loss
        
        loss = criterion(prediction, labels)
        acc = metric(prediction, labels)
        

        # backward pass (calculate gradients of current batch)
        loss.backward()

        # apply gradients to model parameters
        optimizer.step()

        ## log statistics
        #loss_total += loss.item()                       # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor
        ## tensor version for ddp all_gather
        loss_total += loss
        
        pred_label = torch.argmax(prediction, dim=1)    # the predicted label is the one at position (class index) with highest predicted value
        all_predicted_labels.extend(pred_label.cpu().numpy()) # this moves all predicted labels to a list above
        all_ground_truth_labels.extend(labels.cpu().numpy())
            
        oa = torch.mean((pred_label == labels).float()) # OA: number of correct predictions divided by batch size (i.e., average/mean)
        oa_total += oa.item()

        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            )
        )
        progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)           # shorthand notation for: loss_total = loss_total / len(dataLoader)
    oa_total /= len(dataLoader)

    bac = balanced_accuracy_score(all_ground_truth_labels, all_predicted_labels)
    acc = metric.compute()
    print(f"Train Accuracy on all data: {acc*100}, accelerator rank: {rank}")

    # Reseting internal state such that metric ready for new data
    metric.reset()
    #print (f'Device: {rank}, length of predicted labels: {len(all_predicted_labels)}, length of gt labels: {len(all_ground_truth_labels)}')
    return loss_total, oa_total, acc, all_predicted_labels, all_ground_truth_labels



def validate(cfg, dataLoader, class_weights_val, model, metric, rank):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = rank
    
    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    class_weights_val = class_weights_val.to(device, dtype=torch.float32)
    
    criterion = nn.CrossEntropyLoss(class_weights_val)   # we still need a criterion to calculate the validation loss
    #criterion = nn.CrossEntropyLoss()   # we still need a criterion to calculate the validation loss
    
    # running averages
    loss_total, oa_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)

    all_predicted_labels = []
    all_ground_truth_labels = []

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    
    with torch.no_grad():               # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels) in enumerate(dataLoader):

            # put data and labels on device
            data, labels = data.to(device), labels.to(device)

            # forward pass
            prediction = model(data)

            # loss
            loss = criterion(prediction, labels)
            
            acc = metric(prediction, labels)

            # log statistics
            loss_total += loss.item()

            pred_label = torch.argmax(prediction, dim=1)
            all_predicted_labels.extend(pred_label.cpu()) # this moves all predicted labels to a list above
            all_ground_truth_labels.extend(labels.cpu())
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)
    bac = balanced_accuracy_score(all_ground_truth_labels, all_predicted_labels)
    
    acc = metric.compute()
    print(f"Val Accuracy on all data: {acc*100}, accelerator rank: {rank}")

    # Reseting internal state such that metric ready for new data
    metric.reset()
    return loss_total, acc, acc

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main(rank, world_size, class_weights_train, class_weights_val):

    ddp_setup(rank, world_size)

    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet18.yaml')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))
    #cfg['batch_size'] = int(cfg['batch_size']/world_size)
    cfg['learning_rate'] = cfg['learning_rate']*world_size
    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))

    # check if GPU is available
    device = rank
    class_weights_train = class_weights_train.to(device)
    class_weights_val = class_weights_val.to(device)
    
    # if device != 'cpu' and not torch.cuda.is_available():
    #     print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
    #     cfg['device'] = 'cpu'

    if device==0:
        wandb.init(
        # set the wandb project where this run will be logged
        project="standard resnet18 15 percent labeled data",

        # track hyperparameters and run metadata
        config={
        "learning_rate": cfg['learning_rate'],
        "architecture": "resnet 18",
        "dataset": "15 percent labeled",
        "epochs": cfg['num_epochs'],
        "weight_decay": cfg['weight_decay'],
        "batch_size": cfg['batch_size']
        })


    # initialize data loaders for training and validation set
    #dl_train, class_weights_train = create_dataloader(cfg, split='train')
    #dl_val, class_weights_val = create_dataloader(cfg, split='val')
    
    dl_train = create_dataloader(cfg, split='train')
    dl_val = create_dataloader(cfg, split='val')

    # initialize model
    model, current_epoch = load_model(cfg)
    #metric = torchmetrics.classification.MulticlassAccuracy(num_classes=cfg['num_classes'], average='micro')
    metric = torchmetrics.Accuracy(task='multiclass', num_classes=cfg['num_classes'])
    model.metric = metric
    model.to(device)
    model = DDP(model, device_ids=[rank])

    #model = load_pretrained_weights_for_finetuning(cfg, model, '/home/tsharma/Downloads/checkpoint_0020.pth.tar')

    # set up model optimizer
    optim = setup_optimizer(cfg, model)

    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    numEpochs = cfg['num_epochs']
    while current_epoch < numEpochs:
        dl_train.sampler.set_epoch(current_epoch)
        dl_val.sampler.set_epoch(current_epoch)
        
        print(f'GPU {rank} Epoch {current_epoch}/{numEpochs}')
        
        
        loss_train, oa_train, bac_train, all_predicted_labels, all_ground_truth_labels = train(cfg, dl_train, class_weights_train, model, optim, metric, rank)
        ### gather loss_train from all the processes here
        loss_train_all_devices = [torch.zeros_like(loss_train) for _ in range(world_size)]
        all_predicted_labels = torch.tensor(all_predicted_labels).to(device)
        all_ground_truth_labels = torch.tensor(all_ground_truth_labels).to(device)
        
        predicted_labels_all_devices = [torch.zeros_like(all_predicted_labels) for _ in range(world_size)]
        ground_truth_labels_all_devices = [torch.zeros_like(all_ground_truth_labels) for _ in range(world_size)]
        
        dist.all_gather(predicted_labels_all_devices, all_predicted_labels)
        dist.all_gather(ground_truth_labels_all_devices, all_ground_truth_labels)
        
        predicted_labels_all_devices = torch.cat(predicted_labels_all_devices)
        ground_truth_labels_all_devices = torch.cat(ground_truth_labels_all_devices)
        
        oa_train = torch.mean((predicted_labels_all_devices == ground_truth_labels_all_devices).float())
        
        bac_train = balanced_accuracy_score(ground_truth_labels_all_devices.cpu().numpy(), predicted_labels_all_devices.cpu().numpy()) 
        
        
        
        loss_val, oa_val, bac_val = validate(cfg, dl_val, class_weights_val, model, metric, rank)

        # combine stats and save
        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'bac_train':bac_train,
            'oa_val': oa_val,
            'bac_val':bac_val
        }
        if device==0:
            wandb.log({'loss_train': loss_train,
                'loss_val': loss_val,
                'oa_train': oa_train,
                'bac_train':bac_train,
                'oa_val': oa_val,
                'bac_val':bac_val})

        if device==0 and current_epoch%20==0:
            save_model(cfg, current_epoch, model, stats)
        current_epoch += 1

    

    wandb.finish()
    destroy_process_group()
    

def do_before_multiprocessing():
    ''' 
        Stuff that needs to get done only once before we start parallel processing.
        We can't do calculating class weights (class distribution) in any of the spawned processes because
        of sampler = distributed_sampler in the dataloader which results in no single process containing all 
        the batches.
    '''
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet18.yaml')
    args = parser.parse_args()

    # load config
    #print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))
    class_weights_train, class_weights_val = get_class_weights(cfg)
    
    return class_weights_train, class_weights_val
    

if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    world_size = torch.cuda.device_count()
    class_weights_train, class_weights_val = do_before_multiprocessing()
    mp.spawn(main, args=(world_size,class_weights_train, class_weights_val,), nprocs=world_size)
    