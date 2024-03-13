import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch import optim
from sklearn.metrics import auc
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from tqdm import tqdm

def evaluation(preds, topk):
    sort = np.argsort(-preds, axis=1)[:, :topk]
    hr_arr = np.zeros(shape=[sort.shape[0]])
    ndcg_arr = np.zeros(shape=[sort.shape[0]])
    mrr_arr = np.zeros(shape=[sort.shape[0]])
    rows = np.where(sort==99)[0]
    cols = np.where(sort==99)[1]
    hr_arr[rows] = 1.0
    ndcg_arr[rows] = np.log(2) / np.log(cols + 2.0)
    mrr_arr[rows] = 1.0 / (cols + 1.0)
    return hr_arr.tolist(), ndcg_arr.tolist(), mrr_arr.tolist()

def test_process(model, train_loader_1,train_loader_2, p_mu_a, p_logv_a, p_mu_b, p_logv_b, feed_data, is_cuda, topK,  mode='val'):
    all_hr1_list = []
    all_ndcg1_list = []
    all_mrr1_list = []
    all_hr2_list = []
    all_ndcg2_list = []
    all_mrr2_list = []
    fts1 = feed_data['fts1']
    fts2 = feed_data['fts2']

    if mode == 'val':
        movie_nega = feed_data['movie_nega']
        movie_test = feed_data['movie_vali']
        book_nega = feed_data['book_nega']
        book_test = feed_data['book_vali']
    elif mode=='test':
        movie_nega = feed_data['movie_nega']
        movie_test = feed_data['movie_test']
        book_nega = feed_data['book_nega']
        book_test = feed_data['book_test']
    else:
        raise Exception

    for batch_idx, (data1,data2) in enumerate(zip(train_loader_1,train_loader_2)):
        data1 = data1.reshape([-1])
        data2 = data2.reshape([-1])
        val_user_arr_1 = data1.numpy()
        val_user_arr_2 = data2.numpy()
        v_item1 = fts1[val_user_arr_1]
        v_item2 = fts2[val_user_arr_1]
        if is_cuda:
            v_user_1 = torch.LongTensor(val_user_arr_1).cuda()
            v_user_2 = torch.LongTensor(val_user_arr_2).cuda()
            v_item1 = torch.FloatTensor(v_item1).cuda()
            v_item2 = torch.FloatTensor(v_item2).cuda()
        else:
            v_user_1 = torch.LongTensor(val_user_arr_1)
            v_user_2 = torch.LongTensor(val_user_arr_2)
            v_item1 = torch.FloatTensor(v_item1)
            v_item2 = torch.FloatTensor(v_item2)

        res = model.forward2(v_user_1,v_user_2, v_item1, v_item2, p_mu_a, p_logv_a, p_mu_b, p_logv_b)
        y1 = res[0] 
        y2 = res[1] 
        if is_cuda:
            y1 = y1.detach().cpu().numpy()
            y2 = y2.detach().cpu().numpy()
        else:
            y1 = y1.detach().numpy()
            y2 = y2.detach().numpy()


        nega_vali1 = np.array([movie_nega[ele] + [movie_test[ele]] for ele in val_user_arr_2])
        nega_vali2 = np.array([book_nega[ele] + [book_test[ele]] for ele in val_user_arr_1])
        pred1 = np.stack([y1[xx][nega_vali1[xx]] for xx in range(nega_vali1.shape[0])])
        pred2 = np.stack([y2[xx][nega_vali2[xx]] for xx in range(nega_vali2.shape[0])])
        hr1_list, ndcg1_list, mrr1_list = evaluation(pred1, topK)
        hr2_list, ndcg2_list, mrr2_list = evaluation(pred2, topK)
        all_hr1_list += hr1_list
        all_ndcg1_list += ndcg1_list
        all_mrr1_list += mrr1_list
        all_hr2_list += hr2_list
        all_ndcg2_list += ndcg2_list
        all_mrr2_list += mrr2_list

    avg_hr1 = np.mean(all_hr1_list)
    avg_ndcg1 = np.mean(all_ndcg1_list)
    avg_mrr1 = np.mean(all_mrr1_list)
    avg_hr2 = np.mean(all_hr2_list)
    avg_ndcg2 = np.mean(all_ndcg2_list)
    avg_mrr2 = np.mean(all_mrr2_list)

    return avg_hr1, avg_ndcg1, avg_mrr1, avg_hr2, avg_ndcg2, avg_mrr2


