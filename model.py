from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F

def log_normal(x, m, v):
    """
    Computes the elem-wise log probability of a Gaussian and then sum over the
    last dim. Basically we're assuming all dims are batch dims except for the
    last dim.
    Args:
        x: tensor: (batch, ..., dim): Observation
        m: tensor: (batch, ..., dim): Mean
        v: tensor: (batch, ..., dim): Variance
    Return:
        kl: tensor: (batch1, batch2, ...): log probability of each sample. Note
            that the summation dimension (dim=-1) is not kept
    """
    # print("q_m", m.size())
    # print("q_v", v.size())
    const = -0.5 * x.size(-1) * torch.log(2 * torch.tensor(np.pi))
    # print(const.size())
    log_det = -0.5 * torch.sum(torch.log(v), dim=-1)
    # print("log_det", log_det.size())
    log_exp = -0.5 * torch.sum((x - m) ** 2 / v, dim=-1)
    log_prob = const + log_det + log_exp
    return log_prob


def log_normal_mixture(z, m, v):
    """
    Computes log probability of a uniformly-weighted Gaussian mixture.
    Args:
        z: tensor: (batch, dim): Observations
        m: tensor: (batch, mix, dim): Mixture means
        v: tensor: (batch, mix, dim): Mixture variances
    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    z = z.unsqueeze(1)
    log_probs = log_normal(z, m, v)
    log_prob = log_mean_exp(log_probs, 1)

    return log_prob


def log_mean_exp(x, dim):
    """
    Compute the log(mean(exp(x), dim)) in a numerically stable manner
    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which mean is computed
    Return:
        _: tensor: (...): log(mean(exp(x), dim))
    """
    return log_sum_exp(x, dim) - np.log(x.size(dim))


def log_sum_exp(x, dim=0):
    """
    Compute the log(sum(exp(x), dim)) in a numerically stable manner
    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which sum is computed
    Return:
        _: tensor: (...): log(sum(exp(x), dim))
    """
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().sum(dim)).log()


def KL_divergence_gmm(z_given_x, q_m, q_logv, p_m, p_logv):
    """
    Computes the Evidence Lower Bound, KL and, Reconstruction costs
    Returns:
        kld: tensor: (): ELBO KL divergence to prior
    """
    # Compute the mixture of Gaussian prior
    p_m = p_m.unsqueeze(0)
    p_v = torch.exp(p_logv.unsqueeze(0))
    q_v = torch.exp(q_logv)

    # terms for KL divergence
    log_q_phi = log_normal(z_given_x, q_m, q_v)
    # print("log_q_phi", log_q_phi.size())
    log_p_theta = log_normal_mixture(z_given_x, p_m, p_v)
    # print("log_p_theta", log_p_theta.size())
    kl = log_q_phi - log_p_theta
    # print("kl", kl.size())

    kld = torch.sum(kl)
    return kld

def sum_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor):
    """
    Returns the matrix of "x_i + y_j".
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :return: [R, C, D] sum matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    return x_col + y_row


def prod_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor):
    """
    Returns the matrix of "x_i * y_j".
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :return: [R, C, D] sum matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    return x_col * y_row


def distance_tensor(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :param p:
    :return: [R, C, D] distance matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.abs(x_col - y_row) ** p
    return distance


def distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :param p:
    :return: [R, C] distance matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)
    return distance


def distance_gmm(mu_src: torch.Tensor, mu_dst: torch.Tensor, logvar_src: torch.Tensor, logvar_dst: torch.Tensor):
    """
    Calculate a Wasserstein distance matrix between the gmm distributions with diagonal variances
    :param mu_src: [R, D] matrix, the means of R Gaussian distributions
    :param mu_dst: [C, D] matrix, the means of C Gaussian distributions
    :param logvar_src: [R, D] matrix, the log(variance) of R Gaussian distributions
    :param logvar_dst: [C, D] matrix, the log(variance) of C Gaussian distributions
    :return: [R, C] distance matrix
    """
    std_src = torch.exp(0.5 * logvar_src)
    std_dst = torch.exp(0.5 * logvar_dst)
    distance_mean = distance_matrix(mu_src, mu_dst, p=2)
    distance_var = distance_matrix(std_src, std_dst, p=2)
    # distance_var = torch.sum(sum_matrix(std_src, std_dst) - 2 * (prod_matrix(std_src, std_dst) ** 0.5), 2)
    return distance_mean + distance_var + 1e-6


# def tensor_gmm(mu_src: torch.Tensor, mu_dst: torch.Tensor, logvar_src: torch.Tensor, logvar_dst: torch.Tensor):
#     """
#     Calculate a Wasserstein distance matrix between the gmm distributions with diagonal variances
#     :param mu_src: [R, D] matrix, the means of R Gaussian distributions
#     :param mu_dst: [C, D] matrix, the means of C Gaussian distributions
#     :param logvar_src: [R, D] matrix, the log(variance) of R Gaussian distributions
#     :param logvar_dst: [C, D] matrix, the log(variance) of C Gaussian distributions
#     :return: [R, C, D] distance tensor
#     """
#     std_src = torch.exp(0.5 * logvar_src)
#     std_dst = torch.exp(0.5 * logvar_dst)
#     distance_mean = distance_tensor(mu_src, mu_dst, p=2)
#     distance_var = sum_matrix(std_src, std_dst) - 2 * (prod_matrix(std_src, std_dst) ** 0.5)
#     return distance_mean + distance_var


def cost_mat(cost_s: torch.Tensor, cost_t: torch.Tensor, tran: torch.Tensor) -> torch.Tensor:
    """
    Implement cost_mat for Gromov-Wasserstein discrepancy (GWD)

    Suppose the loss function in GWD is |a-b|^2 = a^2 - 2ab + b^2. We have:

    f1(a) = a^2,
    f2(b) = b^2,
    h1(a) = a,
    h2(b) = 2b

    When the loss function can be represented in the following format: loss(a, b) = f1(a) + f2(b) - h1(a)h2(b), we have

    cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
    cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T

    Args:
        cost_s: (ns, ns) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        cost_t: (nt, nt) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        tran: (ns, nt) matrix (torch tensor), representing the optimal transport from source to target domain.
    Returns:
        cost: (ns, nt) matrix (torch tensor), representing the cost matrix conditioned on current optimal transport
    """
    f1_st = torch.sum(cost_s ** 2, dim=1, keepdim=True) / cost_s.size(0)
    f2_st = torch.sum(cost_t ** 2, dim=1, keepdim=True) / cost_t.size(0)
    tmp = torch.sum(sum_matrix(f1_st, f2_st), dim=2)
    cost = tmp - 2 * cost_s @ tran @ torch.t(cost_t)
    return cost


def fgw_discrepancy(mu, z_mu, logvar, z_logvar, device, beta):
    cost_posterior = distance_gmm(mu, mu, logvar, logvar)
    cost_prior = distance_gmm(z_mu, z_mu, z_logvar, z_logvar)
    cost_pp = distance_gmm(mu, z_mu, logvar, z_logvar)

    ns = cost_posterior.size(0)
    nt = cost_prior.size(0)
    p_s = torch.ones(ns, 1) / ns
    p_t = torch.ones(nt, 1) / nt
    tran = torch.ones(ns, nt) / (ns * nt)
    p_s = p_s.to(device)
    p_t = p_t.to(device)
    tran = tran.to(device)
    dual = (torch.ones(ns, 1) / ns).to(device)
    for m in range(10):
        cost = beta * cost_mat(cost_posterior, cost_prior, tran) + (1 - beta) * cost_pp
        kernel = torch.exp(-cost / torch.max(torch.abs(cost))) * tran
        b = p_t / (torch.t(kernel) @ dual)
        # dual = p_s / (kernel @ b)
        for i in range(5):
            dual = p_s / (kernel @ b)
            b = p_t / (torch.t(kernel) @ dual)
        tran = (dual @ torch.t(b)) * kernel
    if torch.isnan(tran).sum() > 0:
        tran = (torch.ones(ns, nt) / (ns * nt)).to(device)

    cost = beta * cost_mat(cost_posterior, cost_prior, tran.detach().data) + (1 - beta) * cost_pp
    d_fgw = (cost * tran.detach().data).sum()
    return d_fgw

class GMMPrior(nn.Module):
    def __init__(self, data_size: list):
        super(GMMPrior, self).__init__()
        self.data_size = data_size
        self.number_components = data_size[0]
        self.output_size = data_size[1]
        self.mu = nn.Parameter(torch.randn(data_size), requires_grad=True)
        self.logvar = nn.Parameter(torch.randn(data_size), requires_grad=True)

    def forward(self):
        return self.mu, self.logvar

class MUCRP(nn.Module):
    def __init__(self, NUM_USER, NUM_MOVIE, NUM_BOOK,  EMBED_SIZE, dropout, is_sparse=False):
        super(MUCRP, self).__init__()
        self.NUM_MOVIE = NUM_MOVIE
        self.NUM_BOOK = NUM_BOOK
        self.NUM_USER = NUM_USER
        self.emb_size = EMBED_SIZE
        self.beta = 0.1

        self.user_embeddings = nn.Embedding(self.NUM_USER, EMBED_SIZE, sparse=is_sparse)
        self.user_embeddings.weight.data = torch.from_numpy(np.random.normal(0, 0.01, size=[self.NUM_USER, EMBED_SIZE])).float()

        self.z_dim = EMBED_SIZE
        self.fc1_a = nn.Linear(self.NUM_MOVIE, 256)
        self.fc21_a = nn.Linear(256, self.z_dim)
        self.fc22_a = nn.Linear(256, self.z_dim)
        self.fc3_a = nn.Linear(self.z_dim, 256)
        self.fc4_a = nn.Linear(256, self.NUM_MOVIE)

        self.fc1_b = nn.Linear(self.NUM_BOOK, 256)
        self.fc21_b = nn.Linear(256, self.z_dim)
        self.fc22_b = nn.Linear(256, self.z_dim)
        self.fc3_b = nn.Linear(self.z_dim, 256)
        self.fc4_b = nn.Linear(256, self.NUM_BOOK)

        self.dropout = nn.Dropout(dropout)
    #### domain a
    def encode_a(self, x):
        h1 = F.relu(self.fc1_a(x))
        return self.fc21_a(h1), self.fc22_a(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode_a(self, z):
        h3 = F.relu(self.fc3_a(z))
        return self.fc4_a(h3)

    def forward_a(self, x):
        mu, logvar = self.encode_a(x.view(-1, self.NUM_MOVIE))
        z = self.reparameterize(mu, logvar)
        return self.decode_a(z), z, mu, logvar

    #### domain b
    def encode_b(self, x):
        h1 = F.relu(self.fc1_b(x))
        return self.fc21_b(h1), self.fc22_b(h1)

    def decode_b(self, z):
        h3 = F.relu(self.fc3_b(z))
        return self.fc4_b(h3)

    def forward_b(self, x):
        mu, logvar = self.encode_b(x.view(-1, self.NUM_BOOK))
        z = self.reparameterize(mu, logvar)
        return self.decode_b(z), z, mu, logvar
    
    

    def forward(self, user_1, user_2, batch_user_x, batch_user_y, p_mu_a, p_logv_a, p_mu_b, p_logv_b):
        preds_x, z_x, mu_a, logvar_a = self.forward_a(self.dropout(batch_user_x))
        preds_y, z_y, mu_b, logvar_b = self.forward_b(self.dropout(batch_user_y))

        gmm_reg_loss_a = KL_divergence_gmm(z_x, mu_a, logvar_a, p_mu_a, p_logv_a)
        gmm_reg_loss_b = KL_divergence_gmm(z_y, mu_b, logvar_b, p_mu_b, p_logv_b)
        
        
        align_global_loss = fgw_discrepancy(mu_b, mu_a, logvar_b, logvar_a, z_y.device, self.beta)

        preds_x2y = self.decode_b(z_x)
        preds_y2x = self.decode_a(z_y)

        _,feature_x_r,_,_ = self.forward_b(preds_x2y)
        _,feature_y_r,_,_ = self.forward_a(preds_y2x)
        
        
        cyc_loss_x = torch.norm(z_x-feature_x_r,p=2)
        cyc_loss_y = torch.norm(z_y-feature_y_r,p=2)


        return preds_x, preds_y, preds_x2y, preds_y2x, z_x, z_y, gmm_reg_loss_a, gmm_reg_loss_b, feature_x_r, feature_y_r, align_global_loss, cyc_loss_x, cyc_loss_y
