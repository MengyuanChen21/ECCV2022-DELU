import numpy as np
import torch
import utils.wsad_utils as utils
from scipy.signal import savgol_filter
import pdb
import pandas as pd
import options

args = options.parser.parse_args()


def filter_segments(segment_predict, vn):
    ambilist = './Thumos14reduced-Annotations/Ambiguous_test.txt'
    try:
        ambilist = list(open(ambilist, "r"))
        ambilist = [a.strip("\n").split(" ") for a in ambilist]
    except:
        ambilist = []
    ind = np.zeros(np.shape(segment_predict)[0])
    for i in range(np.shape(segment_predict)[0]):
        # s[j], e[j], np.max(seg)+0.7*c_s[c],c]
        for a in ambilist:
            if a[0] == vn:
                gt = range(
                    int(round(float(a[2]) * 25 / 16)), int(round(float(a[3]) * 25 / 16))
                )
                pd = range(int(segment_predict[i][0]), int(segment_predict[i][1]))
                IoU = float(len(set(gt).intersection(set(pd)))) / float(
                    len(set(gt).union(set(pd)))
                )
                if IoU > 0:
                    ind[i] = 1
    s = [
        segment_predict[i, :]
        for i in range(np.shape(segment_predict)[0])
        if ind[i] == 0
    ]
    return np.array(s)


def smooth(v, order=2, lens=200):
    l = min(lens, len(v))
    l = l - (1 - l % 2)
    if len(v) <= order:
        return v
    return savgol_filter(v, l, order)


def get_topk_mean(x, k, axis=0):
    return np.mean(np.sort(x, axis=axis)[-int(k):, :], axis=0)


def get_cls_score(element_cls, dim=-2, rat=20, ind=None):
    topk_val, _ = torch.topk(element_cls,
                             k=max(1, int(element_cls.shape[-2] // rat)),
                             dim=-2)
    instance_logits = torch.mean(topk_val, dim=-2)
    pred_vid_score = torch.softmax(
        instance_logits, dim=-1)[..., :-1].squeeze().data.cpu().numpy()
    return pred_vid_score


def _get_vid_score(pred):
    # pred : (n, class)
    if args is None:
        k = 8
        topk_mean = self.get_topk_mean(pred, k)
        # ind = topk_mean > -50
        return pred, topk_mean

    win_size = int(args.topk)
    split_list = [i * win_size for i in range(1, int(pred.shape[0] // win_size))]
    splits = np.split(pred, split_list, axis=0)

    tops = []
    # select the avg over topk2 segments in each window
    for each_split in splits:
        top_mean = get_topk_mean(each_split, args.topk2)
        tops.append(top_mean)
    tops = np.array(tops)
    c_s = np.max(tops, axis=0)
    return pred, c_s


def __vector_minmax_norm(vector, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        max_val = np.max(vector)
        min_val = np.min(vector)

    delta = max_val - min_val
    # delta[delta <= 0] = 1
    ret = (vector - min_val) / delta

    return ret


@torch.no_grad()
def multiple_threshold_hamnet(vid_name, data_dict):
    elem = data_dict['cas']
    element_atn = data_dict['attn']
    act_thresh_cas = np.arange(0.1, 0.9, 10)
    element_logits = elem * element_atn
    # element_logits = data_dict['cas_supp']
    pred_vid_score = get_cls_score(element_logits, rat=10)
    score_np = pred_vid_score.copy()
    # score_np[score_np < 0.2] = 0
    # score_np[score_np >= 0.2] = 1
    cas_supp = element_logits[..., :-1]
    cas_supp_atn = element_atn

    pred = np.where(pred_vid_score >= 0.2)[0]

    # NOTE: threshold
    act_thresh = np.linspace(0.1, 0.9, 10)
    # act_thresh = np.linspace(0.2,0.4,10)

    prediction = None
    if len(pred) == 0:
        pred = np.array([np.argmax(pred_vid_score)])
    cas_pred = cas_supp[0].cpu().numpy()[:, pred]
    num_segments = cas_pred.shape[0]
    cas_pred = np.reshape(cas_pred, (num_segments, -1, 1))

    cas_pred_atn = cas_supp_atn[0].cpu().numpy()[:, [0]]

    cas_pred_atn = np.reshape(cas_pred_atn, (num_segments, -1, 1))

    proposal_dict = {}

    for i in range(len(act_thresh)):
        cas_temp = cas_pred.copy()
        cas_temp_atn = cas_pred_atn.copy()
        seg_list = []
        for c in range(len(pred)):
            pos = np.where(cas_temp_atn[:, 0, 0] > act_thresh[i])
            seg_list.append(pos)

        proposals = utils.get_proposal_oic_2(seg_list,
                                             cas_temp,
                                             pred_vid_score,
                                             pred,
                                             args.scale,
                                             num_segments,
                                             args.feature_fps,
                                             num_segments,
                                             gamma=args.gamma_oic)

        for j in range(len(proposals)):
            try:
                class_id = proposals[j][0][0]

                if class_id not in proposal_dict.keys():
                    proposal_dict[class_id] = []

                proposal_dict[class_id] += proposals[j]
            except IndexError:
                logger.error(f"Index error")
    final_proposals = []
    for class_id in proposal_dict.keys():
        final_proposals.append(
            utils.soft_nms(proposal_dict[class_id], 0.7, sigma=0.3))
    # self.final_res["results"][vid_name[0]] = utils.result2json(
    # final_proposals, class_dict)

    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    # [c_pred[i], c_score, t_start, t_end]
    segment_predict = []
    for i in range(len(final_proposals)):
        for j in range(len(final_proposals[i])):
            [c_pred, c_score, t_start, t_end] = final_proposals[i][j]
            segment_predict.append([t_start, t_end, c_score, c_pred])

    segment_predict = np.array(segment_predict)
    segment_predict = filter_segments(segment_predict, vid_name.decode())

    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    for i in range(np.shape(segment_predict)[0]):
        video_lst.append(vid_name.decode())
        t_start_lst.append(segment_predict[i, 0])
        t_end_lst.append(segment_predict[i, 1])
        score_lst.append(segment_predict[i, 2])
        label_lst.append(segment_predict[i, 3])
    prediction = pd.DataFrame(
        {
            "video-id": video_lst,
            "t-start": t_start_lst,
            "t-end": t_end_lst,
            "label": label_lst,
            "score": score_lst,
        }
    )
    return prediction


@torch.no_grad()
def multiple_threshold_hamnet_modified(vid_name, data_dict, labels):
    elem = data_dict['cas']
    element_atn = data_dict['attn']
    element_logits = elem * element_atn
    pred_vid_score = get_cls_score(element_logits, rat=10)
    cas_supp = element_logits[..., :-1]
    cas_supp_atn = element_atn

    # pred = np.where(pred_vid_score >= 0.2)[0]
    pred = np.where(pred_vid_score >= 0.01)[0]

    # NOTE: threshold
    act_thresh = np.linspace(0.1, 0.9, 10)

    if len(pred) == 0:
        pred = np.array([np.argmax(pred_vid_score)])

    pred_one_hot = np.zeros_like(labels)
    pred_one_hot[pred] = 1

    # if pred_one_hot[0] == 0 and pred_vid_score[0] >= 0.15:
    #     pred = np.concatenate((pred, np.array([0])))
    #     pred_one_hot[0] = 1
    # if pred_one_hot[5] == 0 and pred_vid_score[5] >= 0.17:
    #     pred = np.concatenate((pred, np.array([5])))
    #     pred_one_hot[5] = 1
    # if pred_one_hot[6] == 0 and pred_vid_score[6] >= 0.14:
    #     pred = np.concatenate((pred, np.array([6])))
    #     pred_one_hot[6] = 1

    correct_pred = pred_one_hot * labels

    cas_pred = cas_supp[0].cpu().numpy()[:, pred]
    num_segments = cas_pred.shape[0]
    cas_pred = np.reshape(cas_pred, (num_segments, -1, 1))

    cas_pred_atn = cas_supp_atn[0].cpu().numpy()[:, [0]]

    cas_pred_atn = np.reshape(cas_pred_atn, (num_segments, -1, 1))

    proposal_dict = {}

    for i in range(len(act_thresh)):
        cas_temp = cas_pred.copy()
        cas_temp_atn = cas_pred_atn.copy()
        seg_list = []
        for c in range(len(pred)):
            pos = np.where(cas_temp_atn[:, 0, 0] > act_thresh[i])
            seg_list.append(pos)

        proposals = utils.get_proposal_oic_2(seg_list,
                                             cas_temp,
                                             pred_vid_score,
                                             pred,
                                             args.scale,
                                             num_segments,
                                             args.feature_fps,
                                             num_segments,
                                             gamma=args.gamma_oic)

        for j in range(len(proposals)):
            class_id = proposals[j][0][0]
            if class_id not in proposal_dict.keys():
                proposal_dict[class_id] = []
            proposal_dict[class_id] += proposals[j]

    final_proposals = []
    for class_id in proposal_dict.keys():
        final_proposals.append(
            utils.soft_nms(proposal_dict[class_id], 0.7, sigma=0.3))

    segment_predict = []
    for i in range(len(final_proposals)):
        for j in range(len(final_proposals[i])):
            [c_pred, c_score, t_start, t_end] = final_proposals[i][j]
            segment_predict.append([t_start, t_end, c_score, c_pred])

    segment_predict = np.array(segment_predict)
    segment_predict = filter_segments(segment_predict, vid_name.decode())

    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    for i in range(np.shape(segment_predict)[0]):
        video_lst.append(vid_name.decode())
        t_start_lst.append(segment_predict[i, 0])
        t_end_lst.append(segment_predict[i, 1])
        score_lst.append(segment_predict[i, 2])
        label_lst.append(segment_predict[i, 3])
    prediction = pd.DataFrame(
        {
            "video-id": video_lst,
            "t-start": t_start_lst,
            "t-end": t_end_lst,
            "label": label_lst,
            "score": score_lst,
        }
    )
    return prediction, correct_pred, pred_one_hot
