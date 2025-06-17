import numpy as np

import torch

import math
from torch.optim.lr_scheduler import LambdaLR,MultiStepLR
from math import cos
from tqdm import tqdm
import torch.nn as nn
import time
import os

from sklearn import metrics
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from model import LNet

def share_fc_parameters(model):
    source_params = model.branch_DDFB.fc.state_dict()

    model.branch_FDFB_fc.sequential_classifier.load_state_dict(source_params)


def pretrain(model, trainLoader, trainLoader2, devLoader, devLoader2, numkfold):
    num_models = 2
    d_models = [LNet().branch_DDFB.to(device) for _ in range(num_models)]
    d_optimizers = [torch.optim.Adam(model.parameters(), lr=0.0001, 
                                   betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
                   for model in d_models]
    
    d_loss_func = nn.CrossEntropyLoss().cuda(0)
    best_d_accs = [64.99 for _ in range(num_models)]
    patience = 10
    no_improve_epochs = [0 for _ in range(num_models)]
    d_epochs = 50

    d_save_dir = ""
    if not os.path.exists(d_save_dir):
        os.makedirs(d_save_dir)
    
    model_save_paths = ["" for _ in range(num_models)]
    stopacc = [0.0] * num_models
    for epoch in range(1, d_epochs + 1):
        # break

        for model_idx in range(num_models):
            d_models[model_idx].train()
            train_loss = 0
            correct = 0
            total = 0
            
            loop = tqdm(enumerate(trainLoader), total=len(trainLoader), 
                       desc=f'Model {model_idx+1} Train Epoch [{epoch}/{d_epochs}]')
            
            for batch_idx, (videoData, audioData, label) in loop:
                if torch.cuda.is_available():
                    videoData, audioData, label = videoData.cuda(0), audioData.cuda(0), label.cuda(0)

                _, outputs = d_models[model_idx](videoData, audioData)
                loss = d_loss_func(outputs, label.long())

                d_optimizers[model_idx].zero_grad()
                loss.backward()
                d_optimizers[model_idx].step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += predicted.eq(label.data).cpu().sum()
                
                stopacc[model_idx]=100.0*correct/total
                
                loop.set_postfix(loss=train_loss/(batch_idx+1), acc=100.0*correct/total)

            d_models[model_idx].eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, (videoData, audioData, label) in enumerate(devLoader):
                    if torch.cuda.is_available():
                        videoData, audioData, label = videoData.cuda(0), audioData.cuda(0), label.cuda(0)
                        
                    _, outputs = d_models[model_idx](videoData, audioData)
                    loss = d_loss_func(outputs, label.long())
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += label.size(0)
                    correct += predicted.eq(label.data).cpu().sum()

            current_acc = 100.0 * correct / total

            if current_acc > max(best_d_accs):
                best_d_accs[model_idx] = current_acc
                no_improve_epochs[model_idx] = 0

                timestamp = time.strftime('%m_%d__%H_%M', time.localtime())
                save_path = f"{d_save_dir}/d_{numkfold}_model{model_idx+1}_{timestamp}_{best_d_accs[model_idx]:.4f}.pth"
                torch.save(d_models[model_idx].state_dict(), save_path)
                model_save_paths[model_idx] = save_path
            else:
                no_improve_epochs[model_idx] += 1
        

        stop_accuracy_count = sum(1 for i, acc in enumerate(best_d_accs) if acc < stopacc[i])
        # if stop_accuracy_count == num_models:
        #     break
            
        accuracy_count = sum(1 for acc in stopacc if acc > 75.0)
        # if accuracy_count == num_models and stop_accuracy_count == num_models:
        #     break
            
        early_stop_count = sum(1 for no_improve in no_improve_epochs if no_improve >= patience)
        if early_stop_count == num_models:
            break

        high_accuracy_count = sum(1 for acc in best_d_accs if acc > 80.0)
        if high_accuracy_count == num_models:
            break

    if max(best_d_accs) > 0:

        d_model_dir = ""

        model_files = [f for f in os.listdir(d_model_dir) if f.endswith('.pth')]

        model_files.sort(
            key=lambda x: (
                int(x.split('_')[1]),
                float(x.split('_')[-1].replace('.pth', ''))
            ),
            reverse=True
        )
    
        if model_files:
            latest_model_path = os.path.join(d_model_dir, model_files[0])

            pretrained_weights = torch.load(latest_model_path)
        
            model.branch_DDFB.load_state_dict(pretrained_weights)

            for param in model.branch_DDFB.parameters():
                param.requires_grad = False
        else:
            print("warning2")
    else:
        print("warning")

    emotion_model = LNet().branch_EFB.to(device)
    
    emotion_optimizer = torch.optim.Adam(emotion_model.parameters(), lr=0.0001, 
                                     betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    emotion_loss_func = nn.CrossEntropyLoss().cuda(0)
    best_emotion_acc = 49.99
    patience = 10
    no_improve_epochs = 0
    emotion_epochs = 50

    for epoch in range(1, emotion_epochs + 1):
        emotion_model.train()
        train_loss = 0
        correct = 0
        total = 0
    
        loop = tqdm(enumerate(trainLoader2), total=len(trainLoader2))
        for batch_idx, (videoData,audioData, label) in loop:
            if torch.cuda.is_available():
                videoData,audioData, label = videoData.cuda(0),audioData.cuda(0), label.cuda(0)

            _ , _ , outputs = emotion_model(videoData,audioData)
            loss = emotion_loss_func(outputs, label.long())

            emotion_optimizer.zero_grad()
            loss.backward()
            emotion_optimizer.step()
        
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data).cpu().sum()
        
            loop.set_description(f'Emotion Train Epoch [{epoch}/{emotion_epochs}]')
            loop.set_postfix(loss=train_loss/(batch_idx+1), acc=100.0*correct/total)

        emotion_model.eval()
        val_loss = 0
        correct = 0
        total = 0
    
        with torch.no_grad():
            for batch_idx, (videoData,audioData, label) in enumerate(devLoader2):
                if torch.cuda.is_available():
                    videoData,audioData, label = videoData.cuda(0),audioData.cuda(0), label.cuda(0)

                _, _ , outputs = emotion_model(videoData,audioData)
                loss = emotion_loss_func(outputs, label.long())
            
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += predicted.eq(label.data).cpu().sum()
    
        current_acc = 100.0 * correct / total

        if current_acc > best_emotion_acc:
            best_emotion_acc = current_acc
            no_improve_epochs = 0
            emotion_save_dir = ""
            if not os.path.exists(emotion_save_dir):
                os.makedirs(emotion_save_dir)

            timestamp = time.strftime('%m_%d__%H_%M', time.localtime())
            save_path = f"{emotion_save_dir}/emo_{numkfold}_{timestamp}_{best_emotion_acc:.4f}.pth"
            torch.save(emotion_model.state_dict(), save_path)
        else:
            no_improve_epochs += 1
    
        if no_improve_epochs >= patience:
            break
    
        if best_emotion_acc > 60.0:
            break


    if best_emotion_acc > 0.0:
        emotion_model_dir = ""
        model_files = [f for f in os.listdir(emotion_model_dir) if f.endswith('.pth')]

        model_files.sort(
            key=lambda x: (
                int(x.split('_')[1]),
                float(x.split('_')[-1].replace('.pth', ''))
            ),
            reverse=True
        )
    
        if model_files:
            latest_model_path = os.path.join(emotion_model_dir, model_files[0])

            pretrained_weights = torch.load(latest_model_path)

            model.branch_EFB.load_state_dict(pretrained_weights)

            for param in model.branch_EFB.parameters():
                param.requires_grad = False
        else:
            print("1")
    else:
        print("warning")

    
    return model


def plot_confusion_matrix(y_true, y_pred, labels_name, savename,title=None, thresh=0.6, axis_labels=None):
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()

    if title is not None:
        plt.title(title)

    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = classes
    plt.xticks(num_local, ['Normal','Depression'])
    plt.yticks(num_local, ['Normal','Depression'],rotation=90,va='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if cm[i][j] * 100 > 0:
                plt.text(j, i, format(cm[i][j] * 100 , '0.2f') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")


    plt.savefig(savename, format='png')
    plt.clf()

class AffectnetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset):
        print('initial balance sampler ...')

        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)

        expression_count = [0] * 63
        for idx in self.indices:
            label = dataset.label[idx]
            expression_count[int(label)] += 1

        self.weights = torch.zeros(self.num_samples)
        for idx in self.indices:
            label = dataset.label[idx]
            self.weights[idx] = 1. / expression_count[int(label)]

        print('initial balance sampler OK...')


    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))


    def __len__(self):
        return self.num_samples


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 0.5 * (cos(min((current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps),1) * math.pi) + 1)
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def print_gradients(model, model_name):
    print(f"\nGradients for {model_name}:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  {name}: grad mean = {param.grad.abs().mean().item():.6f}")
        else:
            print(f"  {name}: No gradient")
            
