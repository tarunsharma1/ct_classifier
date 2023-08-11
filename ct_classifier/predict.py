import os
import argparse
import yaml
import glob
from tqdm import trange

import numpy as np
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.nn as nn
import seaborn as sns
# let's import our own classes and functions!
from util import init_seed
from dataset import CTDataset
from model import CustomResNet18, CustomResNet50, SimClrPytorchResNet50, PAWSResNet50
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support

def create_dataloader(cfg, split='test'): #CHANGED THE PATH TO TEST DATA
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = CTDataset(cfg, split)        # create an object instance of our AudioDataset class
    
    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'], #whatever I said was my batch size in the config file, this will be represented here bc it is pulled from that file 
            shuffle=True, #shuffles files read in, which changes them every epoch. Usually turned off for train and val sets. Must do manually. 
            num_workers=cfg['num_workers']
        )
    return dataLoader, len(dataset_instance)


# This tells us how to start a model that we have previoussly stopped or paused, and we need to start from the same epoch 
def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    #model_instance = CustomResNet18(cfg['num_classes'])         # create an object instance of our CustomResNet18 class
    #model_instance = CustomResNet50(cfg['num_classes'])         # create an object instance of our CustomResNet18 class
    #model_instance = SimClrPytorchResNet50(cfg['num_classes'])
    model_instance = PAWSResNet50(cfg['num_classes'])
    ## load latest model state
    #eval_state = glob.glob('/home/tsharma/Downloads/200.pt') #this looks for the saved model files
    #if len(model_states):
    # at least one save state found; get latest
    #model_epochs = [int(m.replace('/datadrive/audio_classifier/model_states-weighted2/','').replace('.pt','')) for m in model_states] #you can define what epoch you want to start on 
    #eval_epoch = '63'

    # load state dict and apply weights to model
    #print(f'Evaluating from epoch {eval_epoch}')

    #eval_epoch = '200'
    state = torch.load(open(cfg['inference_weights'], 'rb'), map_location='cpu')
    model_instance.load_state_dict(state['model'])

    # MOST OF THIS WILL BE COMMENTED OUT FOR US BECAUSE WE WONT BE STARTING WITH SOMETHING NEW


    return model_instance


# THIS IS HOW THE MODEL TRAINS. The validation function is almost the same, some key differences: no backward pass here. We do not run the optimizer here: optimize
# on the training data, but not validate. 
def predict(cfg, dataLoader, model):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    
    # # iterate over dataLoader
    # progressBar = trange(len(dataLoader))
    
    confidence_prediction_list = []

    with torch.no_grad(): 
        true_labels = []
        predicted_labels = []           
        
        # to - do: add individual  
        # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels) in enumerate(dataLoader):
            true_labels.extend(labels)
            # put data and labels on device
            data, labels = data.to(device), labels.to(device)
            
            # forward pass
            prediction = model(data)

            pred_label = torch.argmax(prediction, dim=1).cpu().numpy()
            predicted_labels.extend(pred_label)

            confidence = torch.nn.Softmax(dim=1)(prediction).cpu().numpy() #this is a long confidence probability vector
            confidence_prediction_list.append(confidence)

            
    return predicted_labels, true_labels, confidence_prediction_list

def predict_on_unlabeled(cfg, dataLoader, model, thresh=0.7):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    
    # # iterate over dataLoader
    # progressBar = trange(len(dataLoader))
    
    confidence_prediction_list = []

    with torch.no_grad(): 
        #true_labels = []
        predicted_labels = []
        filename = "/root/10p_predictions_on_unlabeled_thresh_0.1_simclr.csv"
        csv_file = open(filename, 'w')
        csv_writer = csv.writer(csv_file)
                
        
        # to - do: add individual  
        # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data,img_name) in enumerate(dataLoader):
            #true_labels.extend(labels)
            # put data and labels on device
            data = data.to(device)
            
            # forward pass
            prediction = model(data)

            pred_label = torch.argmax(prediction, dim=1).cpu().numpy()
            
            confidence = torch.nn.Softmax(dim=1)(prediction).cpu().numpy() #this is a long confidence probability vector
            
            confidence = np.array([confidence[i,p] for (i,p) in enumerate(pred_label)])
            
            confident_predictions_idx = list(np.where(confidence>=thresh)[0])
            for pred_idx in confident_predictions_idx:
                csv_writer.writerow([img_name[pred_idx], pred_label[pred_idx]])
                predicted_labels.append(pred_label[pred_idx])
                confidence_prediction_list.append(confidence[pred_idx])

            
    return predicted_labels, confidence_prediction_list




def save_confusion_matrix(true_labels, predicted_labels, cfg):
    # make figures folder if not there

    matrix_path = '../figs'
    #### make the path if it doesn't exist
    if not os.path.exists(matrix_path):  
        os.makedirs(matrix_path, exist_ok=True)
    
    ## ROV dataset
    #classes_to_keep = [87, 63, 56, 93, 70, 36, 2, 69, 14, 8, 64, 77, 79, 7, 78, 15, 90, 3, 84, 66, 11, 1, 35, 6, 72, 16, 0, 5, 45, 47, 44, 76, 99, 12, 43, 18, 68, 21, 74, 13, 54, 65, 23, 86, 48, 24, 61, 81, 58, 27]
    ## i2map dataset
    ### i2map dataset sorted by number of instances per class from 10% train set. I removed classes 65,66,68 and 69. They will be part of the unknown -1 class.
    classes_to_keep = [9, 24, 22, 32, 40, 55, 1, 0, 61, 59, 34, 17, 2, 49, 52, 6, 19, 60, 53, 14, 12, 4, 39, 31, 44, 15, 57, 29, 21, 64, 7, 11, 37, 35, 18, 50, 20, 58, 56, 38, 48, 25, 62, 16, 45, 30, 23, 36, 10, 28]

    mapping_indices = {}
    idx = 0
    for c in classes_to_keep:
        if c in list(mapping_indices.keys()):
            continue
        mapping_indices[c] = idx
        idx += 1

    ## add unknown class as last idx 
    mapping_indices[-1] = idx
    
    
    ### only for ROV dataset: Since I initially picked classes_to_keep based on sorted order of the val set and not train set, 
    ### I want to reorder the results in sorted order of the train set. I can do this by giving the desired order as labels to the 
    ### confusion matrix
    
    desired_order_idx = [56, 87, 63, 93, 36, 70, 78, 2, 77, 8, 64, 7, 69, 15, 14, 84, 90, 3, 35, 79, 72, 1, 11, 66, 12, 6, 43, 76, 44, 74, 16, 45, 58, 0, 13, 5, 86, 47, 99, 61, 68, 18, 27, 48, 21, 54, 65, 23, 24, 81,-1]
    
    #desired_order = [mapping_indices[z] for z in desired_order_idx]
    
    confmatrix = confusion_matrix(true_labels, predicted_labels, labels=range(0,cfg['num_classes']))
    
    ## get only the diagnol because conf matrix is too large to visualize
    #print (confmatrix)
    print (type(confmatrix))
    print (confmatrix.shape)
    
    y = [confmatrix[i,i]/np.sum(confmatrix[i,:]) for i in range(0, confmatrix.shape[0])]
    print (y)
    import matplotlib.pyplot as plt
    
    ## reorder y and tick labels
    #reordered_y = [y[i] for i in desired_order]
    #y = reordered_y
    
    
    ## map back to actual class index 
    tick_labels = [x for x,y in mapping_indices.items() for z in range(0,cfg['num_classes']) if y==z]
    print (f' length of tick labels is {len(tick_labels)} length of diagnol of conf matrix is {len(y)}')
    
    
    
    ### map back from actual class idx to class name
    f = open('/root/datasets_ROV/midwater.names', 'r')
    lines = f.readlines()
    f.close()
    map_idx_to_name = {}
    for idx,line in enumerate(lines):
        map_idx_to_name[idx] = line.strip()
    map_idx_to_name[-1] = 'unknown'
    
    #tick_labels_names = [map_idx_to_name[i] for i in tick_labels]
    tick_labels_names = tick_labels
        
    fig = plt.figure(figsize = (9, 5))
    #plt.stem(range(0,len(y)),y, label=desired_order_idx)
    plt.stem(range(0,len(y)),y, label=tick_labels)
    
    print (f'this is what is being plotted: {y} labels are : {tick_labels}')
    #plt.bar(range(0,len(y)),y, tick_label=tick_labels_names)
    plt.xticks(ticks = range(0,len(tick_labels_names)), labels = tick_labels_names, rotation=90)
    #plt.set_xticklabels(list(ticks.values()), fontsize=10, rotation="vertical")

    #plt.title('normalized accuracy per class')
    plt.ylim(0,1)
    plt.tight_layout()
    
    #plt.show()
    
    ## just check number of instances in each class
    #for idx,instance in enumerate(confmatrix):
    #  print(f'class {classes_to_keep[idx]}, # of instances {sum(instance)}')


    #df_cm = pd.DataFrame(confmatrix, range(10), range(10))
    #sn.set(font_scale=1.4) # for label size
    #sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

    #disp = ConfusionMatrixDisplay(confmatrix)
    #confmatrix.save(cfg['data_root'] + '/experiments/'+(args.exp_name)+'/figs/confusion_matrix_epoch'+'_'+ str(split) +'.png', facecolor="white")
    #disp.plot()
    fig.savefig('/root/figs/confusion_matrix-10p-train-i2map-paws.png', dpi=100)
    
    #plt.savefig('/root/figs/confusion_matrix-10p-train-i2map-simclr.png', facecolor="white")
       ## took out epoch)
    return confmatrix

def save_prevision_recall_curve(cfg, true_labels, predicted_labels, confidence_prediction_list): 
    number_of_classes = cfg['num_classes']
    f1_score_per_class = []
    for entry in range (0, number_of_classes):
        # true_labels == entry
        binarized_true_labels = []
        for l in true_labels:
            if l == entry:
                binarized_true_labels.append(1)
            else:
                binarized_true_labels.append(0)
        y_true = np.array(binarized_true_labels)
        y_scores = confidence_prediction_list[:,entry]
        #ipdb.set_trace()
        #print(average_precision_score(y_true, y_scores)) 
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores) #recall is x axis, precision is y axis
        if (precision+recall)!=0:
            f1_score_per_class.append((2*precision*recall)/(precision+recall))
        else:
            f1_score_per_class.append(0)
        #plt.plot(recall, precision)
        #plt.savefig('../figs/PR-curve-class-'+ str(entry) + '-overlay' + '.png', facecolor="white")
        #plt.clf()
        #ipdb.set_trace()
    print ('f1 scores:')
    print (f1_score_per_class)
    
# When you call train.py, Main is what starts running. It has all of the functions we defined above, and it puts them all in here. When you call the file, it looks 
# through Main, and then when it hits a function, it goes to the to the top to understand the function, and then goes back to Main. 
def main():

    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.') # this is how it knows to look at different things 
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet18.yaml') # change path to config using https://github.com/CV4EcologySchool/audio_classifier_example and scrolling down. Change on command line when you run. 
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))
    
    # initialize model
    model = load_model(cfg)
    #### predictions on unlabeled images #####
    # dl_unlabeled, number_of_examples = create_dataloader(cfg, split='unlabeled')
    # predicted_labels, confidence_prediction_list = predict_on_unlabeled(cfg, dl_unlabeled, model, thresh=0.1)
    # print (f'number of confident predictions: {len(predicted_labels)}')
    # print (f'confidence prediction list: {confidence_prediction_list}')
    
    #### metrics on test dataset ###
    dl_val, number_of_examples = create_dataloader(cfg, split='test') #dl_val means dataloader validation 

    predicted_labels, true_labels, confidence_prediction_list = predict(cfg, dl_val, model)
    
    #print (f'len of true labels {np.unique(true_labels)} len of predicted labels {np.unique(predicted_labels)}')
    
    confidence_prediction_list = np.concatenate(np.array(confidence_prediction_list, dtype='object'))
    bac = balanced_accuracy_score(true_labels, predicted_labels)
    oa = (np.where(np.array(true_labels)==np.array(predicted_labels))[0].shape[0])*100/(number_of_examples)
    print(f'balanced accuracy is {bac*100}')
    print(f'oa is {oa}')
    print (f'test set size {number_of_examples}')
    p_r_f = precision_recall_fscore_support(true_labels,predicted_labels, average='macro')
    print (f'precision recall f score support {p_r_f}')
    
    confmatrix = save_confusion_matrix(true_labels, predicted_labels, cfg)
    
    #save_prevision_recall_curve(cfg, true_labels, predicted_labels, confidence_prediction_list)

    #print("confusion matrix saved")

if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()