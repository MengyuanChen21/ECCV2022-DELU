from __future__ import print_function
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
import os
import numpy as np
import pandas as pd

def str2ind(categoryname, classlist):
    return [i for i in range(len(classlist)) if categoryname == classlist[i]][0]


def strlist2indlist(strlist, classlist):
    return [str2ind(s, classlist) for s in strlist]

def filter_segments(segment_predict, videonames, ambilist):
    ind = np.zeros(np.shape(segment_predict)[0])
    for i in range(np.shape(segment_predict)[0]):
        vn = videonames[int(segment_predict[i, 0])]
        for a in ambilist:
            if a[0] == vn:
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

def generate_single_ground_truth_file(annotation_path,args,subset,verbose,output_annotation_path):
    '''the content have to be stored:
    1. idx_to_take
    2. videoname
    3. ambilist
    4. ground_truth
    5. activity_index
    '''

    gtsegments = np.load(annotation_path + "/segments.npy", allow_pickle=True)
    gtlabels = np.load(annotation_path + "/labels.npy", allow_pickle=True)
    videoname = np.load(annotation_path + "/videoname.npy", allow_pickle=True)
    videoname = np.array([i.decode("utf8") for i in videoname])
    gt_subset = np.load(annotation_path + "/subset.npy", allow_pickle=True)
    gt_subset = np.array([s.decode("utf-8") for s in gt_subset])
    classlist = np.load(annotation_path + "/classlist.npy", allow_pickle=True)
    classlist = np.array([c.decode("utf-8") for c in classlist])
    duration = np.load(annotation_path + "/duration.npy", allow_pickle=True)
    ambilist = annotation_path + "/Ambiguous_test.txt"

    try:
        ambilist = list(open(ambilist, "r"))
        ambilist = [a.strip("\n").split(" ") for a in ambilist]
    except:
        ambilist = []

    subset_ind = (subset == gt_subset)
    gtsegments = gtsegments[subset_ind]
    gtlabels = gtlabels[subset_ind]
    videoname = videoname[subset_ind]
    duration = duration[subset_ind]

    idx_to_take = [i for i, s in enumerate(gtsegments)
                       if len(s) > 0]

    gtsegments = gtsegments[idx_to_take]
    gtlabels = gtlabels[idx_to_take]
    videoname = videoname[idx_to_take]



    # which categories have temporal labels ?
    templabelcategories = sorted(list(set([l for gtl in gtlabels for l in gtl])))

    # # the number index for those categories.
    templabelidx = []
    for t in templabelcategories:
        templabelidx.append(str2ind(t, classlist))


    video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []

    for i in range(len(gtsegments)):
        for j in range(len(gtsegments[i])):
            video_lst.append(str(videoname[i]))
            t_start_lst.append(round(gtsegments[i][j][0] * 25 / 16))
            t_end_lst.append(round(gtsegments[i][j][1] * 25 / 16))
            label_lst.append(str2ind(gtlabels[i][j], classlist))
    ground_truth = pd.DataFrame(
        {
            "video-id": video_lst,
            "t-start": t_start_lst,
            "t-end": t_end_lst,
            "label": label_lst,
        }
    )
    activity_index = {i: templabelidx[i] for i in range(len(templabelidx))}

    # to store all these things into a single pkl file
    stored_content={'idx_to_take':idx_to_take,'videoname':videoname,
                    'ambilist':ambilist,'ground_truth':ground_truth,'activity_index':activity_index}
    # store in the target path
    np.save(output_annotation_path,stored_content)


