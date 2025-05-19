import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from utils import degree_of_agreement

class base_CD(nn.Module):
    def __init__(self):
        super(base_CD, self).__init__()

    def train(self, learning_rate, extractor, inter_func, dataloader, device):
        loss_func = nn.BCELoss()
        optimizer = optim.Adam([{'params': extractor.parameters(),
                                 'lr': learning_rate},
                                {'params': inter_func.parameters(),
                                 'lr': learning_rate}])
        extractor.train()
        inter_func.train()
        epoch_loss = []

        with tqdm(dataloader, desc="Training", unit="batch") as pbar:
            for batch_data in pbar:
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
                pbar.set_postfix(loss=float(np.mean(epoch_loss)))

        print("average train loss: %.6f" % (float(np.mean(epoch_loss))))
        wandb.log({"train_loss": float(np.mean(epoch_loss))})

    def eval(self, extractor, inter_func, dataloader, device, doa_list):
        loss_func = nn.BCELoss()
        epoch_loss, auc_list, acc_list = [], [], []
        y_pred, y_true = [], []

        with tqdm(dataloader, desc="Evaluating", unit="batch") as pbar:
            for batch_data in pbar:
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
                    pbar.set_postfix(loss=float(np.mean(epoch_loss)))

        auc_list.append(roc_auc_score(y_true, y_pred))
        acc_list.append(accuracy_score(y_true, np.array(y_pred) >= 0.5))
        stu_mas = inter_func.transform(extractor.update("student"), extractor.update("knowledge"))

        loss_value = float(np.mean(epoch_loss))
        auc_value = round(float(np.mean(auc_list)) * 100, 2)
        acc_value = round(float(np.mean(acc_list)) * 100, 2)
        doa_value = round(degree_of_agreement(stu_mas.detach().cpu().numpy(), doa_list) * 100, 2)

        print("average test loss: %.6f" % (loss_value))
        test_dict = {
            "test_loss": loss_value,
            "auc": auc_value,
            "acc": acc_value,
            "doa": doa_value
        }
        print("auc: %.2f , acc: %.2f , doa: %.2f" % (auc_value, acc_value, doa_value))
        wandb.log(test_dict)
