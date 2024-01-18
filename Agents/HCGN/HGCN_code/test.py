import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import copy
import torch
import joblib
import random
import json
import math
import sys
import argparse
import numpy as np
import torch.nn as nn
import time as sys_time
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.model_selection import KFold 
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index as ci
from sklearn.model_selection import StratifiedKFold
from HGCN.HGCN_code.mae_model import fusion_model_mae_2
from HGCN.HGCN_code.util import Logger, get_patients_information,get_all_ci,get_val_ci,adjust_learning_rate
from HGCN.HGCN_code.mae_utils import generate_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def HGCN():
    
    def _neg_partial_log(prediction, T, E):
    
        current_batch_len = len(prediction)
        R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_matrix_train[i, j] = T[j] >= T[i]
    
        train_R = torch.FloatTensor(R_matrix_train)
        train_R = train_R.cuda()
    
        train_ystatus = torch.tensor(np.array(E),dtype=torch.float).to(device)
    
        theta = prediction.reshape(-1)
    
        exp_theta = torch.exp(theta)
        loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)
    
        return loss_nn 
    def prediction(all_data,v_model,val_id,patient_and_time,patient_sur_type,args):
        v_model.eval()
        # print(val_id)
        lbl_pred_all = None
        status_all = []
        survtime_all = []
        val_pre_time = {}
        val_pre_time_img = {}
        val_pre_time_rna = {}
        val_pre_time_cli = {}
        iter = 0
        
        with torch.no_grad():
            for i_batch, id in enumerate(val_id):
    
                graph = all_data[id].to(device)
                if args.train_use_type != None:
                    use_type_eopch = args.train_use_type
                else:
                    use_type_eopch = graph.data_type
                out_pre,out_fea,out_att,_ = v_model(graph,args.train_use_type,use_type_eopch,mix=args.mix)
                lbl_pred = out_pre[0]
    
                survtime_all.append(patient_and_time[id])
                status_all.append(patient_sur_type[id])
    
                val_pre_time[id] = lbl_pred.cpu().detach().numpy()[0]
    
                if iter == 0 or lbl_pred_all == None:
                    lbl_pred_all = lbl_pred
                else:
                    lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])
    
                iter += 1
                
                if 'img' in use_type_eopch:
                    val_pre_time_img[id] = out_pre[1][use_type_eopch.index('img')].cpu().detach().numpy()
                if 'rna' in use_type_eopch:
                    val_pre_time_rna[id] = out_pre[1][use_type_eopch.index('rna')].cpu().detach().numpy()            
                if 'cli' in use_type_eopch:
                    val_pre_time_cli[id] = out_pre[1][use_type_eopch.index('cli')].cpu().detach().numpy()            
                
        survtime_all = np.asarray(survtime_all)
        status_all = np.asarray(status_all)
    #     print(lbl_pred_all,survtime_all,status_all)
        loss_surv = _neg_partial_log(lbl_pred_all, survtime_all, status_all)
        loss = loss_surv
    
        val_ci_ = get_val_ci(val_pre_time,patient_and_time,patient_sur_type)
        # print(val_pre_time)
        # print(patient_and_time)
        # ordered_time, ordered_pred_time, ordered_observed=[],[],[]
        # ordered_time, ordered_pred_time, ordered_observed=[],[],[]
        # for x in patient_and_time:
        #     ordered_time.append(patient_and_time[x])
        #     ordered_pred_time.append(val_pre_time[x]*-1)
        #     ordered_observed.append(patient_sur_type[x])
        # print(ordered_time, ordered_pred_time, ordered_observed)
        ordered_time, ordered_pred_time, ordered_observed=[],[],[]
        for x in val_pre_time:
            ordered_time.append(patient_and_time[x])
            ordered_pred_time.append(val_pre_time[x]*-1)
            ordered_observed.append(patient_sur_type[x])
    #     print(len(ordered_time), len(ordered_pred_time), len(ordered_observed))
        # print("F",ordered_time, ordered_pred_time, ordered_observed)
        val_ci_img_ = 0 
        val_ci_rna_ = 0 
        val_ci_cli_ = 0
    
        if 'img' in args.train_use_type :
            val_ci_img_ = get_val_ci(val_pre_time_img,patient_and_time,patient_sur_type)
        if 'rna' in args.train_use_type :
            val_ci_rna_ = get_val_ci(val_pre_time_rna,patient_and_time,patient_sur_type)
        if 'cli' in args.train_use_type :
            val_ci_cli_ = get_val_ci(val_pre_time_cli,patient_and_time,patient_sur_type)
        return loss.item(), val_ci_, val_ci_img_, val_ci_rna_, val_ci_cli_,ordered_time, ordered_pred_time, ordered_observed
    
    def main(args): 
        start_seed = args.start_seed
        cancer_type = 'luad'
        repeat_num = args.repeat_num
        drop_out_ratio = args.drop_out_ratio
        lr = args.lr
        epochs = args.epochs
        batch_size = args.batch_size
        details = args.details
        fusion_model = args.fusion_model
        format_of_coxloss = args.format_of_coxloss
        if_adjust_lr = args.if_adjust_lr
        
    
        label = "{} {} lr_{} {}_coxloss".format(cancer_type, details, lr,format_of_coxloss) 
        
        if args.add_mse_loss_of_mae:
            label = label + " {}*mae_loss".format(args.mse_loss_of_mae_factor)
    
        if args.img_cox_loss_factor != 1:
            label = label + " img_ft_{}".format(args.img_cox_loss_factor)
        if args.rna_cox_loss_factor != 1:
            label = label + " rna_ft_{}".format(args.rna_cox_loss_factor)    
        if args.cli_cox_loss_factor != 1:
            label = label + " cli_ft_{}".format(args.cli_cox_loss_factor)    
        if args.mix:
            label = label + " mix"
        if args.train_use_type != None:
            label = label + ' use_'
            for x in args.train_use_type:
                label = label + x
            
        
        # print(label)                                                                                  
    
    
      
        if cancer_type == 'lihc': 
            patients = joblib.load('your path')
            sur_and_time = joblib.load('your path')
            all_data=joblib.load('your path')        
            seed_fit_split = joblib.load('your path')
        elif cancer_type == 'lusc': 
            patients = joblib.load('your path')
            sur_and_time = joblib.load('your path')
            all_data=joblib.load('your path')     
            seed_fit_split = joblib.load('your path')
        elif cancer_type == 'esca': 
            patients = joblib.load('your path')
            sur_and_time = joblib.load('your path')
            all_data=joblib.load('your path')    
            seed_fit_split = joblib.load('your path')
        elif cancer_type == 'luad': 
            patients = joblib.load('./HGCN/HGCN_code/LUAD/luad_patients.pkl')
            sur_and_time = joblib.load('./HGCN/HGCN_code/LUAD/luad_sur_and_time.pkl')
            all_data=joblib.load('./HGCN/HGCN_code/LUAD/luad_data.pkl')     
            seed_fit_split = joblib.load('./HGCN/HGCN_code/LUAD/luad_split.pkl')
        elif cancer_type == 'ucec': 
            patients = joblib.load('your path')
            sur_and_time = joblib.load('your path')
            all_data=joblib.load('your path')             
            seed_fit_split = joblib.load('your path')
        elif cancer_type == 'kirc': 
            patients = joblib.load('your path')
            sur_and_time = joblib.load('your path')
            all_data=joblib.load('your path')             
            seed_fit_split = joblib.load('your path')
    
        patient_sur_type, patient_and_time, kf_label = get_patients_information(patients,sur_and_time)
    
        model = fusion_model_mae_2(in_feats=1024,
                                   n_hidden=args.n_hidden,
                                   out_classes=args.out_classes,
                                   dropout=drop_out_ratio,
                                   train_type_num = len(args.train_use_type)
                                          ).to(device)
        model.load_state_dict(torch.load('./HGCN/HGCN_code/LUAD/model2024-01-11luad  lr_3e-05 multi_coxloss 5*mae_loss img_ft_5 cli_ft_5 mix use_imgrnacli_4_5.pth'))
    
        model.eval() 
    
        # print(patients)
        test_data = patients[:10]
        # test_data = ['TCGA-49-4501','TCGA-49-6742','TCGA-75-7030','TCGA-86-6562']
        
    
        t_test_loss,test_ci,_,_,_,ordered_time, ordered_pred_time, ordered_observed = prediction(all_data,model,test_data,patient_and_time,patient_sur_type,args)
        return (test_ci,ordered_pred_time)
    
    def get_params():
        parser = argparse.ArgumentParser()
        parser.add_argument("--cancer_type", type=str, default="lihc", help="Cancer type")
        parser.add_argument("--img_cox_loss_factor", type=float, default=5, help="img_cox_loss_factor")
        parser.add_argument("--rna_cox_loss_factor", type=float, default=1, help="rna_cox_loss_factor")
        parser.add_argument("--cli_cox_loss_factor", type=float, default=5, help="cli_cox_loss_factor")
        parser.add_argument("--train_use_type", type=list, default=['img','rna','cli'], help='train_use_type,Please keep the relative order of img, rna, cli')
        parser.add_argument("--format_of_coxloss", type=str, default="multi", help="format_of_coxloss:multi,one")
        parser.add_argument("--add_mse_loss_of_mae", action='store_true', default=True, help="add_mse_loss_of_mae")
        parser.add_argument("--mse_loss_of_mae_factor", type=float, default=5, help="mae_loss_factor")
        parser.add_argument("--start_seed", type=int, default=0, help="start_seed")
        parser.add_argument("--repeat_num", type=int, default=5, help="Number of repetitions of the experiment")
        parser.add_argument("--fusion_model", type=str, default="fusion_model_mae_2", help="")
        parser.add_argument("--drop_out_ratio", type=float, default=0.5, help="Drop_out_ratio")
        parser.add_argument("--lr", type=float, default=0.00003, help="Learning rate of model training")
        parser.add_argument("--epochs", type=int, default=60, help="Cycle times of model training")
        parser.add_argument("--batch_size", type=int, default=32, help="Data volume of model training once")
        parser.add_argument("--n_hidden", type=int, default=512, help="Model middle dimension")    
        parser.add_argument("--out_classes", type=int, default=512, help="Model out dimension")
        parser.add_argument("--mix", action='store_true', default=True, help="mix mae")
        parser.add_argument("--if_adjust_lr", action='store_true', default=True, help="if_adjust_lr")
        parser.add_argument("--adjust_lr_ratio", type=float, default=0.5, help="adjust_lr_ratio")
        parser.add_argument("--if_fit_split", action='store_true', default=False, help="fixed division/random division")
        parser.add_argument("--details", type=str, default='', help="Experimental details")
        parser.add_argument("--test_data", type=list, default=[], help="Str of case ids")
        args, _ = parser.parse_known_args()
        return args
    
    
    
    try:
        args=get_params()
        ci,pred = main(args)
        # return args
    except Exception as exception:
        raise
    return ci,pred
   