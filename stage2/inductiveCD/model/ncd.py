from extract.default import extract_fea
from interact.ncd import NCD_IF
from base_model import base_CD

class ncd_model(base_CD):
    def __init__(self, dataset, **kwargs):
        super(ncd_model, self).__init__()
        self.dataset = dataset
        _, _, _, _, tr_know_n, to_know_n = dataset.get_num()
        _, q_matrix = dataset.get_q_matrix()
        q_matrix=q_matrix[:,tr_know_n:]
        self.know_n = to_know_n-tr_know_n
        self.doa_dict = {
            "r_matrix": dataset.doa_data(),
            "data": dataset.get_response(),
            "q_matrix": q_matrix,
            "know_n": 0
        }
       
        self.extractor = extract_fea(dataset, self.know_n, kwargs['dtype'], kwargs['device'])
        self.interfunc = NCD_IF(self.know_n,kwargs['device'], kwargs['dtype'])

    def train_model(self, batch_size, epoch_num, lr, device):
        train_loader, test_loader=self.dataset.get_orginal_iter(batch_size)
        for i in range(epoch_num):
            print("[epoch %d]:" % (i))
            self.train(lr, self.extractor, self.interfunc, train_loader, device)
            self.eval(self.extractor, self.interfunc, test_loader, device, self.doa_dict)
