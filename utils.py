#author:zhaochao time:2021/5/18

import torch as t
import torch.nn.functional as F
import numpy as np
import  random
import torch.nn as nn



class Center_alignment_loss(nn.Module):
    def __init__(self,src_class):
        super(Center_alignment_loss, self).__init__()

        self.n_class=src_class
        self.MSELoss = nn.MSELoss()  # (x-y)^2
        self.MSELoss = self.MSELoss.cuda()




    def forward(self, feature,label,domain_index):
        domian_index_1 = (domain_index == 0)
        domian_index_2 = (domain_index == 1)
        domian_index_3 = (domain_index == 2)

        label1 = label[domian_index_1]
        label2 = label[domian_index_2]
        label3 = label[domian_index_3]

        feature1 = feature[domian_index_1]
        feature2 = feature[domian_index_2]
        feature3 = feature[domian_index_3]

        s1, d = feature1.shape
        s2, d = feature2.shape
        s3, d = feature3.shape

        ones1 = t.ones_like(label1, dtype=t.float)
        ones2 = t.ones_like(label2, dtype=t.float)
        ones3 = t.ones_like(label3, dtype=t.float)

        zeros = t.zeros(self.n_class)

        zeros = zeros.cuda()

        n_classes1 = zeros.scatter_add(0, label1, ones1)
        n_classes2 = zeros.scatter_add(0, label2, ones2)
        n_classes3 = zeros.scatter_add(0, label3, ones3)

        # image number cannot be 0, when calculating centroids
        s_ones1 = t.ones_like(n_classes1)
        s_ones2 = t.ones_like(n_classes2)
        s_ones3 = t.ones_like(n_classes3)

        n_classes1 = t.max(n_classes1, s_ones1)
        n_classes2 = t.max(n_classes2, s_ones2)
        n_classes3 = t.max(n_classes3, s_ones3)

        # calculating centroids, sum and divide
        zeros = t.zeros(self.n_class, d)

        zeros = zeros.cuda()

        s_sum_feature1 = zeros.scatter_add(0, t.transpose(label1.repeat(d, 1), 1, 0), feature1)
        s_sum_feature2 = zeros.scatter_add(0, t.transpose(label2.repeat(d, 1), 1, 0), feature2)
        s_sum_feature3 = zeros.scatter_add(0, t.transpose(label3.repeat(d, 1), 1, 0), feature3)

        current_s_centroid1 = t.div(s_sum_feature1, n_classes1.view(self.n_class, 1))
        current_s_centroid2 = t.div(s_sum_feature2, n_classes2.view(self.n_class, 1))
        current_s_centroid3 = t.div(s_sum_feature3, n_classes3.view(self.n_class, 1))

        semantic_loss = self.MSELoss(current_s_centroid1, current_s_centroid2) + \
                        self.MSELoss(current_s_centroid1, current_s_centroid3) + \
                        self.MSELoss(current_s_centroid2, current_s_centroid3)


        return semantic_loss


        return

class Center_loss(nn.Module):
    def __init__(self,src_class):
        super(Center_loss, self).__init__()

        self.n_class=src_class
        self.MSELoss = nn.MSELoss()  # (x-y)^2
        self.MSELoss = self.MSELoss.cuda()




    def forward(self, s_feature,s_labels):


        n, d = s_feature.shape

        # get labels


        # image number in each class
        ones = t.ones_like(s_labels, dtype=t.float)
        zeros = t.zeros(self.n_class)

        zeros = zeros.cuda()

        s_n_classes = zeros.scatter_add(0, s_labels, ones)


        # image number cannot be 0, when calculating centroids
        ones = t.ones_like(s_n_classes)
        s_n_classes = t.max(s_n_classes, ones)


        # calculating centroids, sum and divide
        zeros = t.zeros(self.n_class, d)

        zeros = zeros.cuda()
        s_sum_feature = zeros.scatter_add(0, t.transpose(s_labels.repeat(d, 1), 1, 0), s_feature)

        s_centroid = t.div(s_sum_feature, s_n_classes.view(self.n_class, 1))


        # calculating inter distance

        temp = t.zeros((n, d)).cuda()

        for i in range(n):
            temp[i] = s_centroid[s_labels[i]]



        # intra_loss = t.norm(temp-s_feature, p=1, dim=0).sum()
        # intra_loss = intra_loss / (d * n)

        intra_loss = self.MSELoss(temp, s_feature)


        return intra_loss


def open_classify(features,centroids,delta,unseen_token_id,th):

    logits = euclidean_metric(features, centroids)


    probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1)

    euc_dis = t.norm(features - centroids[preds], 2, 1).view(-1)

    preds[euc_dis >= th*delta[preds]] = unseen_token_id

    return preds


def open_classify1(predict_label,features,centroids,delta,unseen_token_id,th):
    # logits = euclidean_metric(features, centroids)
    # probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1)
    euc_dis = t.norm(features - centroids[predict_label], 2, 1).view(-1)
    predict_label[euc_dis >= th*delta[predict_label]] = unseen_token_id
    return predict_label



def open_Mclassify(features,centroids,delta,unseen_token_id,th,labels,src_feature):

    logits = Mahalanobis_metric(features, centroids,labels,src_feature)


    probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1)

    euc_dis = t.norm(features - centroids[preds], 2, 1).view(-1)

    preds[euc_dis >= th*delta[preds]] = unseen_token_id

    return preds



def euclidean_metric(a, b):


    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)## 第2维增加一个维度
    b = b.unsqueeze(0).expand(n, m, -1)

    logits = -((a - b) ** 2).sum(dim=2)

    return logits


def Mahalanobis_metric(test_feature, mean, train_label,train_feature):

    batch_size=test_feature.shape[0]

    cl_num=mean.shape[0]

    mahalaonbis_dis=t.zeros((batch_size,cl_num)).long().cuda()


    for cl in range(cl_num):
        label_index = (train_label == cl)
        sour = train_feature[label_index]

        center = mean[cl].detach().cpu().numpy()

        S = np.cov(sour.detach().cpu().numpy(), rowvar=False)

        S_s = np.identity(256) * 0.00000000001

        inverse_cov = np.linalg.inv(S + S_s)

        zero_f = test_feature.detach().cpu().numpy() - center

        zero_f = t.tensor(t.from_numpy(zero_f), dtype=t.float32)

        inverse_cov = t.tensor(t.from_numpy(inverse_cov), dtype=t.float32)

        te_mahalaonbis_dis = t.mm(t.mm(zero_f, inverse_cov), zero_f.t()).diag()

        mahalaonbis_dis[:,cl]=te_mahalaonbis_dis

    mahalaonbis_dis=t.tensor(mahalaonbis_dis,dtype=t.float32).cuda()



    return mahalaonbis_dis



class BoundaryLoss(nn.Module):

    def __init__(self, num_labels=10, feat_dim=2):
        super(BoundaryLoss, self).__init__()
        self.num_labels = num_labels
        self.feat_dim = feat_dim
        self.delta = nn.Parameter(t.randn(num_labels).cuda())
        nn.init.normal_(self.delta)

    def forward(self, pooled_output, centroids, labels):

        logits = euclidean_metric(pooled_output, centroids)


        probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1)

        delta = F.softplus(self.delta)

        c = centroids[labels]
        d = delta[labels]
        x = pooled_output

        euc_dis = t.norm(x - c, 2, 1).view(-1)
        pos_mask = (euc_dis > d).type(t.cuda.FloatTensor)
        neg_mask = (euc_dis < d).type(t.cuda.FloatTensor)

        pos_loss = (euc_dis - d) * pos_mask
        neg_loss = (d - euc_dis) * neg_mask
        loss = pos_loss.mean() + neg_loss.mean()

        return loss, delta


class MBoundaryLoss(nn.Module):

    def __init__(self, num_labels=10, feat_dim=2):
        super(MBoundaryLoss, self).__init__()
        self.num_labels = num_labels
        self.feat_dim = feat_dim
        self.delta = nn.Parameter(t.randn(num_labels).cuda())
        nn.init.normal_(self.delta)

    def forward(self, pooled_output, centroids, labels,src_feature):

        logits = Mahalanobis_metric(pooled_output, centroids,labels,src_feature)


        probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1)

        delta = F.softplus(self.delta)

        c = centroids[labels]
        d = delta[labels]
        x = pooled_output

        euc_dis = t.norm(x - c, 2, 1).view(-1)
        pos_mask = (euc_dis > d).type(t.cuda.FloatTensor)
        neg_mask = (euc_dis < d).type(t.cuda.FloatTensor)

        pos_loss = (euc_dis - d) * pos_mask
        neg_loss = (d - euc_dis) * neg_mask
        loss = pos_loss.mean() + neg_loss.mean()

        return loss, delta






class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin = None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin = margin, p = 2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = t.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = t.norm(anchor - pos, 2, dim = 1).view(-1)
            an_dist = t.norm(anchor - neg, 2, dim = 1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = t.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = t.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx


class BatchHardTripletSelector(object):
    '''
    a selector to generate hard batch embeddings from the embedded batch
    '''
    def __init__(self, *args, **kwargs):
        super(BatchHardTripletSelector, self).__init__()

    def __call__(self, embeds, labels):
        dist_mtx = pdist_torch(embeds, embeds).detach().cpu().numpy()# 计算距离
        labels = labels.contiguous().cpu().numpy().reshape((-1, 1))# 断开连接，深拷贝
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)#返回对角线索引
        lb_eqs = labels == labels.T
        lb_eqs[dia_inds] = False
        dist_same = dist_mtx.copy()
        dist_same[lb_eqs == False] = -np.inf #负正无穷大的浮点表示
        pos_idxs = np.argmax(dist_same, axis = 1)
        dist_diff = dist_mtx.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs == True] = np.inf
        neg_idxs = np.argmin(dist_diff, axis = 1)
        pos = embeds[pos_idxs].contiguous().view(num, -1)
        neg = embeds[neg_idxs].contiguous().view(num, -1)
        return embeds, pos, neg


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    t.manual_seed(seed)  # cpu
    t.cuda.manual_seed_all(seed)  # 并行gpu
    t.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    t.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速




def cal_sim(x1, x2, metric='cosine'):
    # x = x1.clone()
    if len(x1.shape) != 2:
        x1 = x1.reshape(-1, x1.shape[-1])
    if len(x2.shape) != 2:
        x2 = x2.reshape(-1, x2.shape[-1])

    if metric == 'cosine':
        sim = (F.cosine_similarity(x1, x2) + 1) / 2
    else:
        sim = F.pairwise_distance(x1, x2) / t.norm(x2, dim=1)
    return sim




def crit_contrast(feats, probs, s_ctds, t_ctds, lambd=1e-3):
    batch_num = feats.shape[0]
    class_num = s_ctds.shape[0]
    probs = F.softmax(probs, dim=-1)
    max_probs, preds = probs.max(1, keepdim=True)
    # print(probs.shape, max_probs.shape)
    select_index = t.nonzero(max_probs.squeeze() >= 0.3).squeeze(1)
    select_index = select_index.cpu().tolist()

    # todo: calculate margins
    # dist_ctds = cal_cossim(to_np(s_ctds), to_np(t_ctds))
    dist_ctds = cal_sim(s_ctds, t_ctds)
    # print('dist_ctds', dist_ctds.shape)

    M = np.ones(class_num)
    for i in range(class_num):
        # M[i] = np.sum(dist_ctds[i, :]) - dist_ctds[i, i]
        M[i] = dist_ctds.mean() - dist_ctds[i]
        M[i] /= class_num - 1
    # print('M', M)

    # todo: calculate D_k between known samples to its source centroid &
    # todo: calculate D_u distances between unknown samples to all source centroids
    D_k, n_k = 0, 1e-5
    D_u, n_u = 0, 1e-5
    for i in select_index:
        class_id = preds[i][0]
        if class_id < class_num:
            # D_k += F.pairwise_distance(feats[i, :], s_ctds[class_id]).squeeze()
            # print(feats.shape, i)
            D_k += cal_sim(feats[i, :], s_ctds[class_id, :])
            # print('D_k', D_k)
            n_k += 1
        else:
            # todo: judge if unknown sample in the radius region of known centroid
            rp_feats = feats[i, :].unsqueeze(0).repeat(class_num, 1)

            # dist_known = F.pairwise_distance(rp_feats, s_ctds)
            dist_known = cal_sim(rp_feats, s_ctds)
            # print('dist_known', len(dist_known), dist_known)

            M_mean = M.mean()
            outliers = dist_known < M_mean
            dist_margin = (dist_known - M_mean) * outliers.float()
            D_u += dist_margin.sum()

    loss = D_k / n_k  # - D_u / n_u
    return loss.mean() * lambd



def CrossEntropyLoss(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()

    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label * t.log(predict_prob + epsilon)
    return t.sum(instance_level_weight * ce * class_level_weight) / float(N)

def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-20):
    N, C = predict_prob.size()

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    mask = predict_prob.ge(0.000001)  # 逐元素比较
    mask_out = t.masked_select(predict_prob, mask)#


    entropy =-mask_out * t.log(mask_out)


#
    return t.sum(instance_level_weight * entropy * class_level_weight) / float(N)



def BCELossForMultiClassification(label, predict_prob, class_level_weight=None, instance_level_weight=None,
                                  epsilon=1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()

    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'


    bce = -label * t.log(predict_prob + epsilon) - (1.0 - label) * t.log(1.0 - predict_prob + epsilon)

    return t.sum(instance_level_weight * bce * class_level_weight) / float(N)




class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (t.device('cuda')
                  if features.is_cuda
                  else t.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = t.eye(batch_size, dtype=t.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = t.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = t.cat(t.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = t.div(
            t.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = t.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = t.scatter(
            t.ones_like(mask),
            1,
            t.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = t.exp(logits) * logits_mask
        log_prob = logits - t.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss