import numpy as np
from re import T
import torch
import logging
from kfoldLoader import MyDataLoader 
from kfoldLoader_emotion import MyDataLoader as MyDataLoader2
from torch.utils.data import DataLoader
import math
from torch.optim.lr_scheduler import LambdaLR,MultiStepLR
from math import cos
from tqdm import tqdm
import torch.nn as nn
import time
import os
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from model import LNet
from pretrain import *

lr = 0.00001
# lr =0.0001
# lr =0.00005
epochSize = 300
warmupEpoch = 0
testRows = 1
schedule = 'consine'
classes = ['Normal','Depression']
ps = []
rs = []
f1s = []
totals = []

total_pre = []
total_label = []

tim = time.strftime('%m_%d__%H_%M', time.localtime())
filepath = 'the log file path'+str(tim)
savePath1 = "the model file path"+str(tim)

if not os.path.exists(filepath):
        os.makedirs(filepath)

logging.basicConfig(level=logging.NOTSET,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=filepath+'/'+'the log file name.log',
                    filemode='w')


def train(VideoPath, AudioPath,X_train,X_test,labelPath,numkfold,VideoPath2, mdnAudioPath2, X_train2,X_test2,labelPath2):
    mytop = 0
    topacc = 80
    top_p=0
    top_r=0
    top_f1=0
    top_pre=[]
    top_label=[]

    trainSet = MyDataLoader(VideoPath, AudioPath,X_train,labelPath,  "train")
    trainLoader = DataLoader(trainSet, batch_size=16, shuffle=True)
    devSet = MyDataLoader(VideoPath, AudioPath,X_test,labelPath,  "dev")
    devLoader = DataLoader(devSet, batch_size=4, shuffle=False)
    print("trainLoader finish", len(trainLoader), len(devLoader))
    
    trainSet2 = MyDataLoader2(VideoPath2, mdnAudioPath2, X_train2, labelPath2,  "train")
    trainLoader2 = DataLoader(trainSet2, batch_size=16, shuffle=True)
    devSet2 = MyDataLoader2(VideoPath2, mdnAudioPath2, X_test2, labelPath2,  "dev")
    devLoader2 = DataLoader(devSet2, batch_size=4, shuffle=False)
    print("trainLoader2 finish", len(trainLoader2), len(devLoader2))
    
    if torch.cuda.is_available():
        model = LNet().to(device)
    print(f"Model is on: {next(model.parameters()).device}")

    model=pretrain(model,trainLoader,trainLoader2,devLoader,devLoader2,numkfold)

    lossFunc = nn.CrossEntropyLoss().cuda(0)
    print(lr)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr,
                                    betas=(0.9,0.999),
                                    eps=1e-8,
                                    weight_decay=0,
                                    amsgrad=False
                                    )

    train_steps = len(trainLoader)*epochSize
    warmup_steps = len(trainLoader)*warmupEpoch
    target_steps = len(trainLoader)*epochSize
    
    if schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=train_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=target_steps)

    
    logging.info('The {}  fold training begins！！'.format(numkfold))
    savePath=str(savePath1)+'/'+str(numkfold)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
        
    for epoch in range(1, epochSize):
        
        loop = tqdm(enumerate(trainLoader), total=len(trainLoader))
        traloss_one = 0
        correct = 0
        total = 0
        lable1 = []
        pre1 = []
        
        last_train_acc=65
        
        model.train()
        
        # print(f"Model is on: {next(model.parameters()).device}")
        
        for batch_idx, (videoData, audioData, label) in loop:
            if torch.cuda.is_available():
                videoData, audioData, label = videoData.cuda(0), audioData.cuda(0), label.cuda(0)
            # print(f"Device for videoData: {videoData.device}")
            #########################################################################################
            output, loss = model(videoData, audioData , 16 , label)
            #########################################################################################
            #########################################################################################
            traLoss = lossFunc(output, label.long())
            
            Loss=traLoss+loss
            
            optimizer.zero_grad()
            
            Loss.backward()
            

            # print_gradients(model, "model")
            
            
            optimizer.step()
            scheduler.step()
            #########################################################################################
            #########################################################################################
            traloss_one += traLoss
            # traloss_one += contrastive_loss
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data).cpu().sum()
            
            #########################################################################################
            current_train_acc = 100.0 * correct / total
            # if scheduler.getflag= :
            #     last_train_acc=current_train_acc
            # current_lr = scheduler.step(epoch, train_acc=current_train_acc,dev_acc=None)
            
            #########################################################################################
            
            loop.set_description(f'Train Epoch [{epoch}/{epochSize}]')
            # loop.set_postfix(loss = traloss_one/(batch_idx+1), acc=100.0*correct/total,current_lr=current_lr )
            loop.set_postfix(loss = traloss_one/(batch_idx+1), acc=100.0*correct/total)
           
        
        logging.info('EpochSize: {}, Train batch: {}, Loss:{}, Acc:{}%'.format(epoch, batch_idx+1, traloss_one/len(trainLoader), 100.0*correct/total))
        #########################################################################################
        # last_train_acc=scheduler.get()
        # scheduler.stop(last_train_acc)
        #########################################################################################
        
        if epoch-warmupEpoch >=0 and epoch % testRows == 0:
            train_num = 0
            correct = 0
            total = 0
            dictt, labelDict = {},{}
            
            
            label2=[]
            pre2 = []
            
            model.eval()
            
            # print(f"Model is on: {next(model.parameters()).device}")
            
            print("*******dev********")
            loop = tqdm(enumerate(devLoader), total=len(devLoader))
            with torch.no_grad():
                loss_one = 0
                for batch_idx, (videoData, audioData, label) in loop:
                    if torch.cuda.is_available():
                        videoData, audioData,label = videoData.cuda(0), audioData.cuda(0),label.cuda(0)
                    # print(f"Device for videoData: {videoData.device}")
                    devOutput,  Loss = model(videoData, audioData, 4)
                    #######################################################################################################################
                    loss = lossFunc(devOutput, label.long())
                    loss_one += loss
                    train_num+=label.size(0)
                    
                    _, predicted = torch.max(devOutput.data, 1)
                    total += label.size(0)
                    correct += predicted.eq(label.data).cpu().sum()
                    
                    label2.append(label.data)
                    pre2.append(predicted)
                    
                    lable1 += label.data.tolist()
                    pre1 += predicted.tolist()
            
            
            acc = 100.0*correct/total
            lable1 = np.array(lable1)
            pre1 = np.array(pre1)

            # current_lr = scheduler.step(epoch, train_acc=current_train_acc, dev_acc=acc)

            # p = precision_score(lable1, pre1, average='weighted')
            # r = recall_score(lable1, pre1, average='weighted')
            # f1score = f1_score(lable1, pre1, average='weighted')
            p = precision_score(lable1, pre1)
            r = recall_score(lable1, pre1)
            f1score = f1_score(lable1, pre1)
            logging.info('precision:{}'.format(p))
            logging.info('recall:{}'.format(r))
            logging.info('f1:{}'.format(f1score))

            logging.debug('Dev epoch:{}, Loss:{}, Acc:{}%'.format(epoch,loss_one/len(devLoader), acc))
            loop.set_description(f'__Dev Epoch [{epoch}/{epochSize}]')
            loop.set_postfix(loss=loss, acc=acc)
            print('Dev epoch:{}, Loss:{},Acc:{}%'.format(epoch,loss_one/len(devLoader),acc))
            if acc> mytop:
                mytop = max(acc,mytop)
                top_p = p
                top_r = r
                top_f1 = f1score
                top_pre = pre2
                top_label = label2
                
            if acc > topacc:
                topacc = max(acc, topacc)
                checkpoint = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch, 'scheduler':scheduler.state_dict()}
                torch.save(checkpoint, savePath+'/'+"mdn+tcn"+'_'+str(epoch)+'_'+ str(acc)+'_'+ str(p)+'_'+str(r)+'_'+str(f1score)+'.pth')
                
        lo=traloss_one/len(trainLoader)
        if lo<0.01:
            logging.info(f"Early stopping at epoch {epoch}. Moving to the next fold.")
            break
        if epoch>150:
            logging.info(f"Early stopping at epoch {epoch}. Moving to the next fold.")
            break       
            
    top_pre = torch.cat(top_pre,axis=0).cpu()
    top_label=torch.cat(top_label,axis=0).cpu()
    
    totals.append(mytop)
    ps.append(top_p)
    rs.append(top_r)
    f1s.append(top_f1)
    logging.info('topacc:'.format(mytop))
    logging.info('')
    
    print("train end")
    
    return top_label,top_pre

def count(string):
    dig = sum(1 for char in string if char.isdigit())
    return dig

if __name__ == '__main__':
    import random
    from sklearn.model_selection import KFold,StratifiedKFold
    seed = 2222
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    ##########################################################################
    tcn2 = ""
    mdnAudioPath2 = ""
    labelPath2 = ""
    
    Y2 = []
    kf2 = StratifiedKFold(n_splits=10, shuffle=True, random_state = 42)
    X2 = os.listdir(tcn2)
    X2 = [f for f in os.listdir(tcn2) if f.endswith('.npy')]
    X2.sort(key=lambda x: int(x.split('_')[0]))
    X2 = np.array(X2)
    
    to_remove2 = []
    for idx2, i2 in enumerate(X2):
        base_filename2 = str(i2.split('.npy')[0]) + ".csv"
        full_path2 = os.path.join(labelPath2, base_filename2)
        
        if not os.path.exists(full_path2):
            print(f" {full_path2} ")
            to_remove2.append(idx2)
            continue
        
        file_csv2 = pd.read_csv(full_path2)
        bdi2 = int(file_csv2.columns[0])
        Y2.append(bdi2)
    X2 = [i2 for j2, i2 in enumerate(X2) if j2 not in to_remove2]
    X2 = np.array(X2)
    Y2 = np.array(Y2)
    
    train_test_splits2 = kf2.split(X2, Y2)
    train_index2, test_index2 = next(train_test_splits2)
    X_train2, X_test2 = X2[train_index2], X2[test_index2]
    Y_train2, Y_test2 = Y2[train_index2], Y2[test_index2]
    
    ##########################################################################
    tcn = ""
    mdnAudioPath = "e"
    labelPath = ""
    
    
    Y = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state = 42)
#     66x
    X = os.listdir(tcn)
    X.sort(key = lambda x : int(x.split(".")[0]))
    X = np.array(X)
    
    for i in X:
        file_csv = pd.read_csv(os.path.join(labelPath,(str(i.split('.npy')[0])+"_Depression.csv")))
        bdi = int(file_csv.columns[0])
        Y.append(bdi)
        
    numkfold  = 0
    ##########################################################################
    ##########################################################################
    for train_index, test_index in kf.split(X,Y):
        X_train, X_test = X[train_index], X[test_index] 
        numkfold +=1
        logging.info('{}  :{}'.format(numkfold, X_train))
        logging.info('{}  :{}'.format(numkfold, X_test))
        total_label_0,total_pre_0 = train(tcn, mdnAudioPath,X_train,X_test,labelPath,numkfold, tcn2, mdnAudioPath2, X_train2,X_test2,labelPath2)
        total_pre.append(total_pre_0)
        total_label.append(total_label_0)
        # break
        
    total_pre = torch.cat(total_pre,axis=0).cpu().numpy()
    total_label=torch.cat(total_label,axis=0).cpu().numpy()
    np.save(filepath+"/total_pre.npy",total_pre)
    np.save(filepath+"/total_label.npy",total_label)
    
    # plot_confusion_matrix(total_label,total_pre,[0,1],savename=filepath+'/confusion_matrix.png')
    
    logging.info('The accuracy is：{}'.format(totals))
    logging.info('The average accuracy is：{}'.format(sum(totals)/len(totals)))
    logging.info('The precision is：{}'.format(ps))
    logging.info('The average precision is：{}'.format(sum(ps)/len(ps)))
    logging.info('The recall is：{}'.format(rs))
    logging.info('The average recall is：{}'.format(sum(rs)/len(rs)))
    logging.info('The f1 is：{}'.format(f1s))
    logging.info('The average f1 is：{}'.format(sum(f1s)/len(f1s)))
    print(totals)
    print(sum(totals)/len(totals))    