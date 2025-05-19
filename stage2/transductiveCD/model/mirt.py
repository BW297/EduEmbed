from extract.default import extract_fea
from interact.mirt import MIRT_IF
from base_model_mirt import base_CD
class mirt_model(base_CD):
    def __init__(self, dataset, **kwargs):
        super(mirt_model, self).__init__()
        self.dataset = dataset
        self.extractor = extract_fea(dataset, kwargs['latent_dim'], kwargs['dtype'], kwargs['device'],kwargs['lamda'],kwargs['alpha'])
        self.interfunc = MIRT_IF(kwargs['latent_dim'],kwargs['device'], kwargs['dtype'])

    def train_model(self, batch_size, epoch_num, lr, device):
        train_loader, test_loader=self.dataset.get_orginal_iter(batch_size)
        for i in range(epoch_num):
            print("[epoch %d]:" % (i))
            self.train(lr, self.extractor, self.interfunc, train_loader, device)
            self.eval(self.extractor, self.interfunc, test_loader, device)
