from dataloader_frame import DataloaderFrame
from agent_frame import AgentFrame
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score, precision_recall_curve
import os
import json
import numpy as np


class ExperimentFrame:
    def __init__(self, model_name, llm_name, dataset_name):
        self.model_name = model_name
        self.llm_name = llm_name
        self.dataset_name = dataset_name
        self.dataloader_frame = DataloaderFrame(dataset_name)
        self.agent_frame = AgentFrame(model_name, llm_name)
        self.output_dir = './logs'
        self.output_file = os.path.join(self.output_dir, f"{self.model_name}_{self.llm_name}_{self.dataset_name}.json")
        
    def run(self):
        if os.path.exists(self.output_file):
            result = json.load(open(self.output_file, "r"))
            preds = result['preds']
            labels = result['labels']
            from_user_idx = result['from_user_idx']
            processed_user = set(from_user_idx)
        else:
            preds = []
            labels = []
            from_user_idx = []
            processed_user = set()

        while self.dataloader_frame.has_next():
            print(f"Processing {self.dataloader_frame.progress()}")
            uid, user = self.dataloader_frame.next_user()
            if uid in processed_user:
                continue
            try:
                user_preds, user_labels = self.agent_frame.run(user)
                assert len(user_preds) == len(user_labels)
            except Exception as e:
                print(f"Something wrong with user {uid} : {e}")
                continue
            preds.extend(user_preds)
            labels.extend(user_labels)
            from_user_idx.extend([uid] * len(user_preds))
            self.save(preds, labels, from_user_idx)
            if (uid+1) % 100 == 0:
                self.evaluate(preds, labels)
        self.evaluate(preds, labels)

    def save(self, preds, labels, from_user_idx):
        with open(self.output_file, "w") as f:
            json.dump({
                "preds": preds,
                "labels": labels,
                "from_user_idx": from_user_idx
            }, f)
    
    def evaluate(self, preds, labels):
        auc, mrr, ndcg = evaluate(preds, labels)
        print(f"AUC: {auc}, MRR: {mrr}, NDCG: {ndcg}")
        return auc, mrr, ndcg

def evaluate(preds, labels):
    """
    preds: score of each news, list[list[float]]
    labels: ground truth of each news in 0/1, list[list[int]]
    return: metrics(AUC, MRR, NDCG)
    """
    # Flatten preds and labels for AUC calculation
    flat_preds = [item for sublist in preds for item in sublist]
    flat_labels = [item for sublist in labels for item in sublist]
    auc = roc_auc_score(flat_labels, flat_preds)
    # Calculate MRR and NDCG
    mrr_scores = []
    ndcg_scores = []
    for i in range(len(labels)):
        # Ensure there are positive labels before calculating MRR for a user
        if sum(labels[i]) > 0:
            # Sort predictions and labels by prediction score
            sorted_indices = np.argsort(preds[i])[::-1]
            sorted_labels = np.array(labels[i])[sorted_indices]
            # Calculate MRR
            for rank, label in enumerate(sorted_labels):
                if label == 1:
                    mrr_scores.append(1 / (rank + 1))
                    break 
        # Calculate NDCG
        # ndcg_score expects 2D arrays: (n_samples, n_labels)
        # Here, we calculate NDCG for each user (sample) separately and average
        # Ensure true_relevance and scores are 2D for ndcg_score
        true_relevance = np.asarray([labels[i]])
        scores = np.asarray([preds[i]])
        ndcg_scores.append(ndcg_score(true_relevance, scores))
    mrr = np.mean(mrr_scores) if mrr_scores else 0
    ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
    return auc, mrr, ndcg

