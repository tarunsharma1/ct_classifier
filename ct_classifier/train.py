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



def create_dataloader(cfg, split='train'):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = CTDataset(cfg, split)        # create an object instance of our CTDataset class

    device = cfg['device']

    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['num_workers']
        )
    
    ### compute weights for class balancing
    classes_for_weighting = []
    for data, labels in dataLoader:
        classes_for_weighting.extend(list(labels.numpy()))  

    class_weights=class_weight.compute_class_weight('balanced',classes = np.unique(classes_for_weighting),y = np.array(classes_for_weighting))
    class_weights = class_weights/np.sum(class_weights)
    class_weights=torch.tensor(class_weights,dtype=torch.float).to(device)

    return dataLoader, class_weights
    


def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CustomResNet18(cfg['num_classes'])         # create an object instance of our CustomResNet18 class

    # load latest model state
    model_states = glob.glob('model_states/*.pt')
    if len(model_states):
        # at least one save state found; get latest
        model_epochs = [int(m.replace('model_states/','').replace('.pt','')) for m in model_states]
        start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        state = torch.load(open(f'model_states/{start_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])

    else:
        # no save state found; start anew
        print('Starting new model')
        start_epoch = 0

    return model_instance, start_epoch

def load_pretrained_weights(model, custom_weights=None):
    if custom_weights:
        state = torch.load(open(custom_weights, 'rb'), map_location='cpu')
        pretrained_dict = state['state_dict']
        model_dict = model.state_dict()
        ## only update the weights of layers with the same names and also don't update the last layer because of size mismatch between num classes
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in ['classifier.weight', 'classifier.bias']}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


        ## we need to copy everything except the last layer
        #for key in state['model'].keys():
            #if not(key == 'classifier.weight' or key== 'classifier.bias'):
            ###### Tarun : LESSON LEARNT : THE LINE BELOW DOES NOT WORK FOR SOME REASON ..you must do it like mentioned in the most liked answer here https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2 ###
            #model.state_dict()[key] = state['model'][key]
        #model.load_state_dict(state['model'])
        ##import ipdb;ipdb.set_trace()
    return model



def save_model(cfg, epoch, model, stats):
    # make sure save directory exists; create if not
    os.makedirs('model_states', exist_ok=True)

    # get model parameters and add to stats...
    stats['model'] = model.state_dict()

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



def train(cfg, dataLoader, model, optimizer, class_weights_train):
    '''
        Our actual training function.
    '''

    device = cfg['device']

    # put model on device
    model.to(device)
    
    # put the model into training mode
    # this is required for some layers that behave differently during training
    # and validation (examples: Batch Normalization, Dropout, etc.)
    model.train()

    # loss function
    criterion = nn.CrossEntropyLoss(class_weights_train)

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

        # backward pass (calculate gradients of current batch)
        loss.backward()

        # apply gradients to model parameters
        optimizer.step()

        # log statistics
        loss_total += loss.item()                       # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor

        pred_label = torch.argmax(prediction, dim=1)    # the predicted label is the one at position (class index) with highest predicted value
        all_predicted_labels.extend(pred_label.cpu()) # this moves all predicted labels to a list above
        all_ground_truth_labels.extend(labels.cpu())
            
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

    return loss_total, oa_total, bac



def validate(cfg, dataLoader, model, class_weights_val):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    
    criterion = nn.CrossEntropyLoss(class_weights_val)   # we still need a criterion to calculate the validation loss

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

    return loss_total, oa_total, bac



def main():

    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet18.yaml')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'


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
    dl_train, class_weights_train = create_dataloader(cfg, split='train')
    dl_val, class_weights_val = create_dataloader(cfg, split='val')

    # initialize model
    model, current_epoch = load_model(cfg)

    model = load_pretrained_weights(model, '/home/tsharma/Downloads/checkpoint_0020.pth.tar')

    # set up model optimizer
    optim = setup_optimizer(cfg, model)

    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    numEpochs = cfg['num_epochs']
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        loss_train, oa_train, bac_train = train(cfg, dl_train, model, optim, class_weights_train)
        loss_val, oa_val, bac_val = validate(cfg, dl_val, model, class_weights_val)

        # combine stats and save
        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'bac_train':bac_train,
            'oa_val': oa_val,
            'bac_val':bac_val
        }
        wandb.log({'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'bac_train':bac_train,
            'oa_val': oa_val,
            'bac_val':bac_val})

        save_model(cfg, current_epoch, model, stats)
    

    wandb.finish()
        


if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()
