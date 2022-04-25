from collections import Counter
import numpy as np
from sklearn.metrics import f1_score

from yf_kazan_test.category_tree import Category
from yf_kazan_test.util import print_md

def hF1_score(true, pred):
    hF1_weighted = 0
    
    true_paths = np.array([cat.path for cat in map(Category, true)])
    pred_paths = np.array([cat.path for cat in map(Category, pred)])
    
    weights = {id: count / len(true) for id, count in Counter(true).most_common()}   
    
    for label in map(Category, set(true)):
        p_idx = np.where((pred_paths == label.path) & (pred_paths != 0))
        t_idx = np.where((true_paths == label.path) & (true_paths != 0))
        pt_idx = np.where((pred_paths == true_paths) & (pred_paths == label.path) & (pred_paths != 0))
    
        if len(p_idx[0]) and len(t_idx[0]) and len(pt_idx[0]):
            hP = len(pt_idx[0]) / len(p_idx[0])
            hR = len(pt_idx[0]) / len(t_idx[0])
            hF1 = 2*hP*hR / (hP+hR)
            hF1_weighted += hF1 * weights[label.id]
    
    return round(hF1_weighted, 6)

def match_rate(true, pred):
    perfect_match = len([1 for t, p in zip(true, pred) if t == p])

    true_path = np.array([cat.path for cat in map(Category, true)])
    pred_path = np.array([cat.path for cat in map(Category, pred)])
    no_match = len(np.where(((true_path != pred_path) | ((true_path == 0) & (pred_path == 0))).all(axis=1))[0])
    total = len(true)

    return {
        "perfect_match": perfect_match*100 / total, 
        "partial_match": (total-perfect_match-no_match)*100 / total,
        "no_match" : no_match*100 / total
    }

def prediction_score(true, pred):
    return {
        "match_rate": match_rate(true, pred),
        "basic_f1": f1_score(true, pred, average="weighted"),
        "hierarchical_f1": hF1_score(true, pred)
    }

def print_score(true, pred):
    score = prediction_score(true, pred)

    print("perfect match: {:.1f}%, partial match: {:.1f}%, no match: {:.1f}%".format(
        score["match_rate"]["perfect_match"],
        score["match_rate"]["partial_match"],
        score["match_rate"]["no_match"]
    ))

    print("basic F1: {:.4f}".format(score["basic_f1"]))
    print_md("<pre>hierarchical F1: <b>{:.4f}</b></pre>".format(score["hierarchical_f1"]))