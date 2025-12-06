import numpy as np


def calculate_metrics(predictions, ground_truth):
    precision = len(set(predictions) & set(ground_truth)) / len(predictions)

    recall = len(set(predictions) & set(ground_truth)) / len(ground_truth)

    dcg = sum([1 / np.log2(i + 2) for i, item in enumerate(predictions) if item in ground_truth])
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(predictions), len(ground_truth)))])
    ndcg = dcg / idcg if idcg > 0 else 0

    return {'precision': precision, 'recall': recall, 'ndcg': ndcg}