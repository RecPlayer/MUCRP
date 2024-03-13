from __future__ import print_function
import matplotlib
from numpy.core.fromnumeric import trace
matplotlib.use('Agg')
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
import time
import os
import numpy as np
from Dataset_CDR import Dataset
from model import MUCRP,GMMPrior
import time
from utils import *
from evaluation import *
from scipy.sparse import csr_matrix

method_name = 'MUCRP'
topK_list = [10, 20]

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--batch', type=int, default=256, help='batch size.')
parser.add_argument('--emb_size', type=int, default=128, help='embed size.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--log', type=str, default='logs/{}'.format(method_name), help='log directory')
parser.add_argument('--K', type=int, default=30, help='d epoch')
parser.add_argument('--t_percent', type=float, default=1.0, help='target percent')
parser.add_argument('--s_percent', type=float, default=1.0, help='source percent')
parser.add_argument('--dataset', type=str, default='amazon', help='amazon')
parser.add_argument('--opk', type=float, default=0.6, help='overlapping ratio')
parser.add_argument('--beta',type=float,default=0.2, help='regularization loss')
parser.add_argument('--lam_vl',type=float,default=0.7, help='local alignment loss ratio')
parser.add_argument('--lam_vg',type=float,default=1.0, help='global alignment loss ratio')
parser.add_argument('--lam_vc',type=float,default=0.3, help='cycle loss ratio')
parser.add_argument('--pos-weight', type=float, default=1.0, help='weight for positive samples')

args = parser.parse_args()

args.cuda = torch.cuda.is_available()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    log = os.path.join(args.log, '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.emb_size, args.weight_decay, args.beta,
                                                     args.lam_vl, args.lam_vg, args.lam_vc, args.opk, args.K))
    if os.path.isdir(log):
        print("%s already exist. are you sure to override? Ok, I'll wait for 5 seconds. Ctrl-C to abort." % log)
        time.sleep(5)
        os.system('rm -rf %s/' % log)

    os.makedirs(log)
    print("made the log directory", log)

    print('preparing data...')

    dataset = Dataset(args.batch, dataset=args.dataset)

    NUM_USER = dataset.num_user
    NUM_MOVIE = dataset.num_movie
    NUM_BOOK = dataset.num_book

    print('Preparing the training data......')

    row, col = dataset.get_part_train_indices('movie', args.s_percent)
    values = np.ones(row.shape[0])
    user_x = csr_matrix((values, (row, col)), shape=(NUM_USER, NUM_MOVIE)).toarray()
    row, col = dataset.get_part_train_indices('book', args.t_percent)
    values = np.ones(row.shape[0])
    user_y = csr_matrix((values, (row, col)), shape=(NUM_USER, NUM_BOOK)).toarray()

    print('Preparing the training data over......')

    user_id = np.arange(NUM_USER).reshape([NUM_USER, 1])

    per = (1.0+args.opk)/2.0
    user_id_1 = np.arange(int(per*NUM_USER)).reshape([int(per*NUM_USER),1])
    user_id_2 = np.arange(int((1-per)*NUM_USER),NUM_USER).reshape([NUM_USER-int((1-per)*NUM_USER),1])
    # user_id_3 = np.arange(int((1-args.opk)/2.0*NUM_USER)).reshape([int((1-args.opk)/2.0*NUM_USER),1])
    # user_id_4 = np.arange(NUM_USER-int((1-args.opk)/2.0*NUM_USER),NUM_USER).reshape([int((1-args.opk)/2.0*NUM_USER),1])
    
    user_x = torch.FloatTensor(user_x)
    user_y = torch.FloatTensor(user_y)

    
    train_loader_1 = torch.utils.data.DataLoader(torch.from_numpy(user_id_1),
                                                     batch_size=args.batch,
                                                     shuffle=True)
    train_loader_2 = torch.utils.data.DataLoader(torch.from_numpy(user_id_2),
                                                     batch_size=args.batch,
                                                     shuffle=True)
                            
    

    pos_weight = torch.FloatTensor([args.pos_weight])

    if args.cuda:
        pos_weight = pos_weight.cuda()

    model = MUCRP(NUM_USER=NUM_USER, NUM_MOVIE=NUM_MOVIE, NUM_BOOK=NUM_BOOK,
                 EMBED_SIZE=args.emb_size, dropout=args.dropout)
    gmm_prior_a = GMMPrior(data_size=[args.K, args.emb_size])
    gmm_prior_b = GMMPrior(data_size=[args.K, args.emb_size])



    # optimizer_g = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.Adam(list(model.parameters()) + list(gmm_prior_a.parameters()) + list(gmm_prior_b.parameters()), lr=args.lr,weight_decay=args.weight_decay, betas=(0.5, 0.999))
    
    BCEWL = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    LL2 = torch.nn.MSELoss(reduction='sum')
    BCEL = torch.nn.BCEWithLogitsLoss(reduction='none')

    if args.cuda:
        model = model.cuda()
        gmm_prior_a = gmm_prior_a.cuda()
        gmm_prior_b = gmm_prior_b.cuda()


    # prepare data fot test process
    movie_vali, movie_test, movie_nega = dataset.movie_vali, dataset.movie_test, dataset.movie_nega
    book_vali, book_test, book_nega = dataset.book_vali, dataset.book_test, dataset.book_nega
    feed_data = {}
    feed_data['fts1'] = user_x
    feed_data['fts2'] = user_y
    feed_data['movie_vali'] = movie_vali
    feed_data['book_vali'] = book_vali
    feed_data['movie_test'] = movie_test
    feed_data['book_test'] = book_test
    feed_data['movie_nega'] = movie_nega
    feed_data['book_nega'] = book_nega


    loss_list = []
    local_loss_list = []
    global_loss_list = []


    epoch_time_list = []
    for epoch in range(args.epochs):
        model.train()
        gmm_prior_a.train()
        gmm_prior_b.train()

        
        batch_loss_list = []
        batch_local_loss_list = []
        batch_global_loss_list = []


        epoch_time = 0.0
        for batch_idx, (data1,data2) in enumerate(zip(train_loader_1,train_loader_2)):
            
            data1 = data1.reshape([-1])
            data2 = data2.reshape([-1])
            list_1 = data1.numpy().tolist()
            list_2 = data2.numpy().tolist()
            common_set = set(list_1) & set(list_2)

            in_com_1 = []
            in_com_2 = []
            out_com_1 = []
            out_com_2 = []
            for idx,user in enumerate(list_1):
                if user in common_set:
                    in_com_1.append(idx)
                else:
                    out_com_1.append(idx)
            for idx,user in enumerate(list_2):
                if user in common_set:
                    in_com_2.append(idx)
                else:
                    out_com_2.append(idx)
            
            in_com_1 = torch.from_numpy(np.array(in_com_1))
            in_com_2 = torch.from_numpy(np.array(in_com_2))
            out_com_1 = torch.from_numpy(np.array(out_com_1))
            out_com_2 = torch.from_numpy(np.array(out_com_2))
            

            optimizer.zero_grad()

            if args.cuda:
                batch_user_1 = data1.cuda()
                batch_user_2 = data2.cuda()
                batch_user_x = user_x[data1].cuda()
                batch_user_x2y = user_y[data1].cuda()
                batch_user_y = user_y[data2].cuda()
                batch_user_y2x = user_x[data2].cuda()
                in_com_1 = in_com_1.cuda()
                in_com_2 = in_com_2.cuda()
                out_com_1 = out_com_1.cuda()
                out_com_2 = out_com_2.cuda()

            else:
                batch_user_1 = data1
                batch_user_2 = data2
                batch_user_x = user_x[data1]
                batch_user_x2y = user_y[data1]
                batch_user_y = user_y[data2]
                batch_user_y2x = user_x[data2]

            
            p_mu_a, p_logv_a = gmm_prior_a()
            p_mu_b, p_logv_b = gmm_prior_b()


            time1 = time.time()
            pred_x, pred_y, pred_x2y, pred_y2x, zx, zy, gmm_reg_loss_a, gmm_reg_loss_b, feature_x, feature_y, align_global_loss,z_x_cyc_loss, z_y_cyc_loss = model.forward(batch_user_1, batch_user_2, batch_user_x, batch_user_y, p_mu_a, p_logv_a, p_mu_b, p_logv_b)
            time2 = time.time()
            epoch_time += time2 - time1

            loss_x = BCEWL(pred_x, batch_user_x).sum() + args.beta * gmm_reg_loss_a
            loss_y = BCEWL(pred_y, batch_user_y).sum() + args.beta * gmm_reg_loss_b

            
            loss_x2y = 0
            loss_y2x = 0
            loss_local = 0

            if len(common_set) >0:
                loss_x2y = BCEWL(pred_x2y[in_com_1], batch_user_x2y[in_com_1]).sum()
                loss_y2x = BCEWL(pred_y2x[in_com_2], batch_user_y2x[in_com_2]).sum()
                loss_local = torch.norm(zx[in_com_1]-zy[in_com_2])

            loss_global = align_global_loss
            # loss_cycle = z_x_cyc_loss + z_y_cyc_loss + loss_x2y + loss_y2x
            loss_cycle = loss_x2y + loss_y2x
            loss = loss_x + loss_y + args.lam_vl* loss_local + args.lam_vg  * loss_global + args.lam_vc * loss_cycle 
            
            loss.backward()
            optimizer.step()
            

            batch_loss_list.append(loss.item())
            if len(common_set) >0:
              batch_local_loss_list.append(loss_local.item())
            batch_global_loss_list.append(loss_global.item())

        epoch_time_list.append(epoch_time)

        epoch_loss = np.mean(batch_loss_list)
        epoch_local_loss = np.mean(batch_local_loss_list)
        epoch_global_loss = np.mean(batch_global_loss_list)

        loss_list.append(epoch_loss)
        local_loss_list.append(epoch_local_loss)
        global_loss_list.append(epoch_global_loss)
        
        print('epoch:{}, loss:{:.4f}'.format(epoch,epoch_loss))



if __name__ == "__main__":
    print(args)
    main()
    print(args)