import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from manu_data import part2attr_np, part2attr_dict
from manu_data_flickr import part2attr_np as part2attr_np_flickr
from collections import defaultdict


# ##################Loss for matching text-image###################
def get_pt_version():
    raw_version = torch.__version__
    raw_version = raw_version.split('.')
    # raw_version = [int(raw_version[i])*pow(10, len(raw_version)-i-1) for i in range(len(raw_version))]
    raw_version = [int(raw_version[i]) * pow(10, 3 - i - 1) for i in range(2)]
    version = sum(raw_version)
    return version


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def func_attention_original(query, context, segs, opt):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw=17*17)
    mask: batch_size x sourceL
    """
    PT_VERSION = get_pt_version()
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x ndf x sourceL
    context = context.view(batch_size, -1, sourceL)
    seg = nn.Upsample(size=(ih, iw), mode='nearest')(segs)
    seg = seg.view(batch_size, opt.label_nc, sourceL)
    # [8, 19, 289 (17 * 17)]
    max_seg = torch.zeros(batch_size, sourceL)
    if opt.dataset_mode == 'landscape' or opt.dataset_mode == 'traffic':
        for i in range(opt.label_nc):
            nonzero = torch.nonzero(seg[:, i, :])
            max_seg[nonzero[:, 0], nonzero[:, 1]] = i
        max_seg = max_seg.reshape(-1).int()  # [batch_size * sourceL]
        part2attr_mask = torch.from_numpy(part2attr_np_flickr[max_seg, :]).cuda()  # [batch_size * sourceL, queryL]
    else:
        for i in range(opt.label_nc):
            nonzero = torch.nonzero(seg[:, i, :])
            max_seg[nonzero[:, 0], nonzero[:, 1]] = i + 1
        max_seg = max_seg.reshape(-1).int()  # [batch_size * sourceL]
        part2attr_mask = torch.from_numpy(part2attr_np[max_seg, :]).cuda()  # [batch_size * sourceL, queryL]
    part2attr_mask = part2attr_mask.float()
    part2attr_mask = part2attr_mask.view(batch_size, sourceL, queryL)
    part2attr_mask = torch.transpose(part2attr_mask, 1, 2).contiguous()     # [bs, queryL, sourceL]

    part2attr_mask = part2attr_mask.view(batch_size * queryL, sourceL).float()
    if PT_VERSION >= 120:
        attn = part2attr_mask.data.masked_fill(part2attr_mask.bool().bitwise_not(), -float('inf'))
    else:
        attn = part2attr_mask.data.masked_fill(1 - part2attr_mask.byte(), -float('inf'))
    attn = nn.Softmax(dim=1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    attn[torch.isnan(attn)] = 0
    attnT = torch.transpose(attn, 1, 2).contiguous()    # [bs, sourceL, queryL]
    # (batch x ndf x sourceL)(batch x sourceL x queryL)--> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)
    return weightedContext, attn.view(batch_size, -1, ih, iw)



def func_attention(query, context):
    """query: batch_size * ndf * queryL [bs, 256, 11]
       context: batch_size * queryL * 8 * 4 [bs, 11 * 256, 8, 4]
    """
    batch_size, ndf, queryL = query.size(0), query.size(1), query.size(2)
    ih, iw = context.size(2), context.size(3)
    context = context.view(batch_size, queryL, ndf, ih * iw)    # [batch_size, 11, 256, 8 * 4]

    # --> [batch_size, sourceL, queryL]
    global_attn = torch.zeros([batch_size, queryL, ih * iw]).cuda()
    global_weicontext = torch.zeros([batch_size, ndf, queryL]).cuda()
    for i in range(queryL):
        context_i = context[:, i, :, :]     # [bs, ndf, ih * iw]
        context_iT = torch.transpose(context_i, 1, 2)
        region_mul = torch.bmm(context_iT, query[:, :, i].unsqueeze(-1)).squeeze(-1)     # [bs, ih * iw]
        region_attn = region_mul / (region_mul.sum(dim=1).unsqueeze(-1).repeat(1, ih * iw) + (1e-8))
        global_attn[:, i, :] = region_attn
        weightedContext = torch.bmm(context_i, region_attn.unsqueeze(-1)).squeeze(-1)       # [bs, ndf, 1]
        global_weicontext[:, :, i] = weightedContext
    return global_weicontext, global_attn.view(batch_size, queryL, ih, iw)


def words_loss(img_features, segs, att_embs, labels, words_num, class_ids, correct, opt, attr_relist, total_m):
    """
        att_embs(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17 [8, 256, 17, 17]
        segs[-1]: [batch_size, P_NUM (19), 256, 128]
    """
    PT_VERSION = get_pt_version()
    batchSize = segs.shape[0]
    # multi_grained = True, default.
    simi = torch.zeros(batchSize, batchSize).cuda()
    for index in range(len(img_features)):
        masks = []
        attn_maps = []
        similarities = []
        context = img_features[index]
        att_emb_ = att_embs[index]
        for i in range(batchSize):
            if class_ids is not None:
                mask = (class_ids == class_ids[i]).astype(np.uint8)
                mask[i] = 0
                masks.append(mask.reshape((1, -1)))
            word = att_emb_[i, :, :words_num].unsqueeze(0).contiguous()
            word = word.repeat(batchSize, 1, 1)
            weiContext, attn = func_attention(word, context)

            attn_maps.append(attn[i].unsqueeze(0).contiguous())
            word = word.transpose(1, 2).contiguous()
            weiContext = weiContext.transpose(1, 2).contiguous()
            word = word.view(batchSize * words_num, -1)
            weiContext = weiContext.view(batchSize * words_num, -1)
            row_sim = cosine_similarity(word, weiContext)
            # --> batch_size x words_num
            row_sim = row_sim.view(batchSize, words_num)
            row_sim.mul_(opt.smooth_gamma2).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)
            similarities.append(row_sim)
        simi += torch.cat(similarities, 1)
    # mean.
    similarities = simi / opt.num_grained
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        masks = torch.BoolTensor(masks).cuda()


    similarities = similarities * opt.smooth_gamma3
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    if opt.dataset_mode == 'landscape' or opt.dataset_mode == 'traffic':
        # 多标签的检索训练方式，比如同一个Batch中出现不止一个相同的attr: blue-sky，那么对应image中也有多个属于正确答案
        dd = defaultdict(list)
        attr_relist = attr_relist.cpu().numpy().tolist()
        for k, va in [(v, i) for i, v in enumerate(attr_relist)]:
            dd[k].append(va)
        m = len(dd)
        if m < batchSize:
            new_simi = torch.zeros([len(attr_relist), m]).cuda()  # N * m
            newest_simi = torch.zeros([m, m]).cuda()  # m * m
            for column in range(m):
                value = list(dd.values())[column]  # [0], [1, 3], [2]
                sum = torch.zeros([len(attr_relist), 1]).cuda()
                for i in value:
                    sum[:, 0] += similarities[:, i]
                sum = sum / len(value)
                new_simi[:, column] = sum[:, 0]
            for line in range(m):
                value = list(dd.values())[line]  # [0], [1, 3], [2]
                sum = torch.zeros([1, m]).cuda()
                for i in value:
                    sum[0, :] += new_simi[i, :]
                sum = sum / len(value)
                newest_simi[line, :] = sum[0, :]
            newest_simi1 = newest_simi.transpose(0, 1)
        elif m == batchSize:
            newest_simi = similarities
            newest_simi1 = similarities.transpose(0, 1)
        labels = Variable(torch.LongTensor(range(m))).cuda()
        loss0 = nn.CrossEntropyLoss()(newest_simi, labels)
        loss1 = nn.CrossEntropyLoss()(newest_simi1, labels)
        _, predicted = torch.max(newest_simi, 1)
        _, predicted1 = torch.max(newest_simi1, 1)
        correct += ((predicted == labels).sum().cpu().item() + (predicted1 == labels).sum().cpu().item())
        total_m += m
    elif opt.dataset_mode == 'vip':
        similarities1 = similarities.transpose(0, 1)
        criterion = nn.CrossEntropyLoss()
        loss0 = criterion(similarities, labels[:batchSize])
        loss1 = criterion(similarities1, labels[:batchSize])
        _, predicted = torch.max(similarities, 1)
        _, predicted1 = torch.max(similarities1, 1)
        correct += ((predicted == labels).sum().cpu().item() + (predicted1 == labels).sum().cpu().item())
        total_m += batchSize
    return loss0, loss1, attn_maps, correct, total_m


