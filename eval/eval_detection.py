# This code is originally from the official ActivityNet repo
# https://github.com/activitynet/ActivityNet
# Small modification from ActivityNet Code
from __future__ import print_function
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import savgol_filter,medfilt
import sys
import scipy.io as sio
import os
from eval.utils_eval import get_blocked_videos
from eval.utils_eval import interpolated_prec_rec
from eval.utils_eval import segment_iou
import pdb

def str2ind(categoryname, classlist):
    return [i for i in range(len(classlist)) if categoryname == classlist[i]][0]


def strlist2indlist(strlist, classlist):
    return [str2ind(s, classlist) for s in strlist]



def sigmoid(x, eps=1e-10):
    return 1/(1+np.exp(-x) + eps)



def smooth(v, order=2,lens=200):
    # return v
    l = min(lens, len(v))
    l = l - (1 - l % 2)
    if len(v) <= order:
        return v
    return savgol_filter(v, l, order)

def smooth_medfilt(v,lens=200):
    l = min(lens, len(v))
    l = l - (1 - l % 2)
    if len(v) <= lens:
        return v
    return medfilt(v,l)

def filter_segments(segment_predict, videonames, ambilist):
    ind = np.zeros(np.shape(segment_predict)[0])
    for i in range(np.shape(segment_predict)[0]):
        vn = videonames[int(segment_predict[i, 0])]
        for a in ambilist:
            if a[0] == vn:
                gt = range(
                    int(round(float(a[2]) * 25 / 16)), int(round(float(a[3]) * 25 / 16))
                )
                gt = range(
                    int(round(float(a[2]) * 25 / 16)), int(round(float(a[3]) * 25 / 16))
                )
                pd = range(int(segment_predict[i][1]), int(segment_predict[i][2]))
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

def moving_smooth(y,box_size):
    assert box_size%2==1, 'The bosx size should be ood'
    box=np.ones(box_size)/box_size
    y=np.array([y[0]]*(box_size//2)+y.tolist()+[y[-1]]*(box_size//2))
    y_smooth=np.convolve(y,box,mode='valid')
    return y_smooth

def gaussian_smooth(score,sigma=30):
    # r = score.shape[0] //39
    # if r%2==0:
    #     r+=1
    r = 125
    if r > score.shape[0] // 2:
        r = score.shape[0] // 2 - 1
    if r % 2 == 0:
        r += 1
    gaussian_temp=np.ones(r*2-1)
    for i in range(r*2-1):
        gaussian_temp[i]=np.exp(-(i-r)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    new_score=score
    for i in range(r,score.shape[0]-r):
        new_score[i]=np.dot(score[i-r:i+r-1],gaussian_temp)
    return new_score

def min_max_norm(p):
    min_p=np.min(p)
    max_p=np.max(p)
    return (p-min_p)/(max_p-min_p)

class ANETdetection(object):

    def __init__(
        self,
        annotation_path='./Thumos14reduced-Annotations',
        tiou_thresholds=np.array([0.1, 0.3, 0.5]),
        args=None,
        subset="test",
        # verbose=False
        verbose=True
    ):
        self.subset = subset
        self.args = args
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.ap = None
        self.annotation_path = os.path.join(args.path_dataset,annotation_path)
        self.prediction = None
        self._import_ground_truth(self.annotation_path)

        #####

    def _import_ground_truth(self, annotation_path):
        gtsegments = np.load(annotation_path + "/segments.npy", allow_pickle=True)
        gtlabels = np.load(annotation_path + "/labels.npy", allow_pickle=True)
        videoname = np.load(annotation_path + "/videoname.npy", allow_pickle=True)
        videoname = np.array([i.decode("utf8") for i in videoname])
        subset = np.load(annotation_path + "/subset.npy", allow_pickle=True)
        subset = np.array([s.decode("utf-8") for s in subset])
        classlist = np.load(annotation_path + "/classlist.npy", allow_pickle=True)
        classlist = np.array([c.decode("utf-8") for c in classlist])
        duration = np.load(annotation_path + "/duration.npy", allow_pickle=True)
        ambilist = annotation_path + "/Ambiguous_test.txt"

        try:
            ambilist = list(open(ambilist, "r"))
            ambilist = [a.strip("\n").split(" ") for a in ambilist]
        except:
            ambilist = []

        self.ambilist = ambilist
        self.classlist = classlist

        subset_ind = (subset == self.subset)
        gtsegments = gtsegments[subset_ind]
        gtlabels = gtlabels[subset_ind]
        videoname = videoname[subset_ind]
        duration = duration[subset_ind]

        self.idx_to_take = [i for i, s in enumerate(gtsegments)
                            if len(s) >0 ]

        gtsegments = gtsegments[self.idx_to_take]
        gtlabels = gtlabels[self.idx_to_take]
        videoname = videoname[self.idx_to_take]

        self.videoname = videoname
        # which categories have temporal labels ?
        templabelcategories = sorted(list(set([l for gtl in gtlabels for l in gtl])))

        # # the number index for those categories.
        templabelidx = []
        for t in templabelcategories:
            templabelidx.append(str2ind(t, classlist))

        self.templabelidx = templabelidx

        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []

        for i in range(len(gtsegments)):
            for j in range(len(gtsegments[i])):
                video_lst.append(str(videoname[i]))
                t_start_lst.append(round(gtsegments[i][j][0]*25/16))
                t_end_lst.append(round(gtsegments[i][j][1]*25/16))
                label_lst.append(str2ind(gtlabels[i][j], self.classlist))
        ground_truth = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
            }
        )
        self.ground_truth = ground_truth
        self.activity_index = {i:templabelidx[i] for i in range(len(templabelidx))}

    def get_topk_mean(self, x, k, axis=0):
        return np.mean(np.sort(x, axis=axis)[-int(k):, :], axis=0)


    def _get_vid_score(self, pred):
        # pred : (n, class)
        if self.args is None:
            k = 8
            topk_mean = self.get_topk_mean(pred, k)
            # ind = topk_mean > -50
            return pred, topk_mean

        win_size = int(self.args.topk)
        split_list = [i*win_size for i in range(1, int(pred.shape[0]//win_size))]
        splits = np.split(pred, split_list, axis=0)

        tops = []
        #select the avg over topk2 segments in each window
        for each_split in splits:
            top_mean = self.get_topk_mean(each_split, self.args.topk2)
            tops.append(top_mean)
        tops = np.array(tops)
        c_s = np.max(tops, axis=0)
        return pred, c_s

    def _get_vid_score_1(self, p):
        pp = - p; [pp[:,i].sort() for i in range(np.shape(pp)[1])]; pp=-pp
        if int(np.shape(pp)[0]/8)>0:
            c_s = np.mean(pp[:int(np.shape(pp)[0]/8),:],axis=0)
        else:
            c_s = np.mean(pp[:np.shape(pp)[0],:],axis=0)
        return p,c_s

    def _get_att_topk_mean(self,p,att_logits,k):
        args_topk = np.argsort(att_logits, axis=0)[-k:]
        topk_mean = 1 / (1 + np.exp(-np.mean(att_logits[args_topk], axis=0))) * 1 / (
                    1 + np.exp(-np.mean(p[args_topk], axis=0)))
        return topk_mean

    def _get_vid_score_2(self,p,att_logits):
        if self.args is None:
            k=8
            topk_mean=self._get_att_topk_mean(p,att_logits,k)
            return p,topk_mean
        win_size = int(self.args.topk)
        split_list = [i * win_size for i in range(1, int(p.shape[0] // win_size))]
        p_splits = np.split(p, split_list, axis=0)
        att_splits=np.split(att_logits,split_list,axis=0)

        tops=[]
        for p_s,a_s in zip(p_splits,att_splits):
            top_mean=self._get_att_topk_mean(p_s,a_s,self.args.topk2)
            tops.append(top_mean)
        tops = np.array(tops)
        c_s = np.max(tops, axis=0)
        return p, c_s

    def OIC_Cofidence(self,s,e,cls_pred,c_s,_lambda=0.25):
        for i in range(len(s)):
            seg=cls_pred[s[i]:e[i]]
            inner_score=np.mean(seg)
            proposal_len=e[i]-s[i]
            outer_s=max(0,int(s[i]-proposal_len*_lambda))
            outer_e=min(cls_pred.shape[0],int(e[i]-proposal_len*_lambda))

            front_outer_score=np.mean(cls_pred[outer_s:s[i]])
            back_outer_score=np.mean(cls_pred[e[i]:outer_e])

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            print("Warning: No predictions of label '%s' were provdied." % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby("label")
        prediction_by_label = self.prediction.groupby("label")

        results = Parallel(n_jobs=3)(
            delayed(compute_average_precision_detection)(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(
                    drop=True
                ),
                prediction=self._get_predictions_with_label(
                    prediction_by_label, label_name, cidx
                ),
                tiou_thresholds=self.tiou_thresholds,
            )
            for label_name, cidx in self.activity_index.items()
        )

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:, cidx] = results[i]

        return ap

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        if self.verbose:
            # print("[INIT] Loaded annotations from {} subset.".format(self.subset))
            nr_gt = len(self.ground_truth)
            print("\tNumber of ground truth instances: {}".format(nr_gt))
            nr_pred = len(self.prediction)
            print("\tNumber of predictions: {}".format(nr_pred))
            # print("\tFixed threshold for tiou score: {}".format(self.tiou_thresholds))

        self.ap = self.wrapper_compute_average_precision()
        # print(self.ap)
        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        # if self.verbose:
            # print ('[RESULTS] Performance on ActivityNet detection task.')
            # for k in range(len(self.tiou_thresholds)):
            #     print("Detection map @ %f = %f" % (self.tiou_thresholds[k], self.mAP[k]))
            # print("Average-mAP: {}\n".format(self.mAP))
        return self.mAP


    def save_info(self, fname):
        import pickle
        Dat = {
            "prediction": self.prediction,
            "gt": self.ground_truth
        }
        with open(fname, 'wb') as fp:
            pickle.dump(Dat, fp)



def compute_average_precision_detection(
    ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)
    ):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction["score"].values.argsort()[::-1]  #idx from high to low
    prediction = prediction.loc[sort_idx].reset_index(drop=True) #value from high to low

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby("video-id")

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred["video-id"])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(
            this_pred[["t-start", "t-end"]].values, this_gt[["t-start", "t-end"]].values
        )
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]["index"]] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]["index"]] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(
            precision_cumsum[tidx, :], recall_cumsum[tidx, :]
        )

    return ap
