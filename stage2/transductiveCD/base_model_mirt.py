import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from joblib import Parallel, delayed

def top_k_concepts(top_k, q_matrix, tmp_set):
    arr = np.array(tmp_set[:, 1], dtype=int)
    counts = np.sum(q_matrix[np.array(tmp_set[:, 1], dtype=int), :], axis=0)
    return np.argsort(counts).tolist()[:-top_k - 1:-1]

class base_CD(nn.Module):
    def __init__(self):
        super(base_CD, self).__init__()

    def train(self, learning_rate, extractor, inter_func, dataloader, device):
        loss_func = nn.BCELoss()
        optimizer = optim.Adam([{'params': extractor.parameters(),
                                 'lr': learning_rate}])
        extractor.train()
        inter_func.train()
        epoch_loss = []
        for batch_data in dataloader:
            student_id, exercise_id, q_mask, r = batch_data
            student_id: torch.Tensor = student_id.to(device)
            exercise_id: torch.Tensor = exercise_id.to(device)
            q_mask: torch.Tensor = q_mask.to(device)
            r: torch.Tensor = r.to(device)
            _ = extractor.extract(student_id, exercise_id, 'train')
            student_ts, diff_ts, disc_ts, knowledge_ts = _[:4]
            pred_r = inter_func.compute(student_ts, diff_ts, disc_ts, knowledge_ts, q_mask)
            if len(_) > 4:
                extra_loss = _[4].get('extra_loss', 0)
            else:
                extra_loss = 0
            loss = loss_func(pred_r, r) + extra_loss 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss.mean()))
        print("average train loss: %.6f" % (float(np.mean(epoch_loss))))
        wandb.log({"train_loss": float(np.mean(epoch_loss))})

    def eval(self, extractor, inter_func, dataloader, device):
        loss_func = nn.BCELoss()
        epoch_loss, auc_list, acc_list, rmse_list = [], [], [], []
        y_pred, y_true = [], []
        for batch_data in dataloader:
            extractor.eval()
            inter_func.eval()
            with torch.no_grad():
                student_id, exercise_id, q_mask, r = batch_data
                student_id: torch.Tensor = student_id.to(device)
                exercise_id: torch.Tensor = exercise_id.to(device)
                q_mask: torch.Tensor = q_mask.to(device)
                r: torch.Tensor = r.to(device)
                _ = extractor.extract(student_id, exercise_id, 'test')
                student_ts, diff_ts, disc_ts, knowledge_ts = _[:4]
                pred_r = inter_func.compute(student_ts, diff_ts, disc_ts, knowledge_ts, q_mask)
                if len(_) > 4:
                    extra_loss = _[4].get('extra_loss', 0)
                else:
                    extra_loss = 0
                l = loss_func(pred_r, r) + extra_loss
                y_pred.extend(pred_r.detach().cpu().tolist())
                y_true.extend(r.tolist())
                epoch_loss.append(float(l.mean()))

        auc_list.append(roc_auc_score(y_true, y_pred))
        acc_list.append(accuracy_score(y_true, np.array(y_pred) >= 0.5))
        mse=mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        print(mse,rmse)
        rmse_list.append(rmse)
        stu_mas = inter_func.transform(extractor.update("student"), extractor.update("knowledge"))
        loss_value = float(np.mean(epoch_loss))
        auc_value = round(float(np.mean(auc_list)) * 100, 2)
        acc_value = round(float(np.mean(acc_list)) * 100, 2)
        rmse_value = round(float(np.mean(rmse_list)), 4)
        print("average test loss: %.6f" % (loss_value))
        test_dict = {
            "test_loss": loss_value,
            "auc": auc_value,
            "acc": acc_value,
            "rmse":rmse_value
        }
        print("auc: %.2f , acc: %.2f , rmse: %.4f" % (auc_value, acc_value, rmse_value))
        wandb.log(test_dict)
