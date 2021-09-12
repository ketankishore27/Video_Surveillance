from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import glob
from torchvision.models import resnet50
import torchmetrics
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import AdvancedProfiler
pl.seed_everything=42

transformations_train = transforms.Compose([transforms.Resize(224), 
                                            transforms.CenterCrop(224), 
                                            transforms.RandomVerticalFlip(p=0.5), 
                                            transforms.RandomHorizontalFlip(p=0.5), 
                                            transforms.RandomRotation(degrees=15), 
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

class FireImageDataset(Dataset):
    
    def __init__(self, img_dir, transform=None):
        
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.img_path = self.get_image_paths()

    def get_image_paths(self):
        path_fire = [i for i in glob.glob(self.img_dir+'fire\*')]
        path_non_fire = [i for i in glob.glob(self.img_dir+'non_fire\*')]
        path = path_fire + path_non_fire
        return path
        
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        image = Image.open(img_path).convert('RGB')
        label = img_path.split('\\')[0].split('/')[-1]
        if self.transform:
            image = self.transform(image)
         
        if label == 'fire':
            label = [1]
        elif label == 'non_fire':
            label = [0]
        else:
            raise Exception
            
        return image, torch.FloatTensor(label)


class FireImageDataLoader(pl.LightningDataModule):
    
    def __init__(self, img_dir, batch_size = 32, num_workers=8, pin_memory=True):
        super().__init__()
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = False
        self.dataset = FireImageDataset(img_dir = self.img_dir, transform=transformations_train)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, 
                                                                             lengths=[int(self.dataset.__len__() * 0.9), int(self.dataset.__len__() * 0.1)])

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory = self.pin_memory, shuffle = True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory = self.pin_memory, shuffle = False)


class ResnetModule(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.base_model = resnet50(pretrained=True, progress=True)
        for params in self.base_model.parameters():
            params.requires_grad=False
        self.base_model.fc = nn.Sequential(
                    nn.Linear(in_features=self.base_model.fc.in_features, out_features=256),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=256, out_features=32),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=32, out_features=1)         
        )

    def forward(self, x):
        return self.base_model(x)

class ResnetClassifier(pl.LightningModule):
    
    def __init__(self, learning_rate=0.0001, path_pretrain=None, training=True):
        super().__init__()
                 
        self.learning_rate = learning_rate
        self.classifier = ResnetModule()
        self.accuracy = torchmetrics.Accuracy()
        self.F1 = torchmetrics.F1(num_classes = 2)
        self.recall = torchmetrics.Recall(average='micro', num_classes=2)
        self.criterion = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()
        
    def forward(self, x):
        return torch.sigmoid(self.classifier(x))
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def metrics_logger_custom(self, predictions, target, prefix):
        prefix = prefix + '/'
        outputs = dict()
        preds = predictions[:, 0]
        target = target[:, 0].int()
        outputs[prefix + '/accuracy'] = torchmetrics.functional.accuracy(preds, target)
        outputs[prefix + '/precision'] = torchmetrics.functional.precision(preds, target)
        outputs[prefix + '/recall'] = torchmetrics.functional.recall(preds, target)
        outputs[prefix + '/f1'] = torchmetrics.functional.f1(preds, target, num_classes=1)
        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        outputs = self.metrics_logger_custom(predictions = preds, target = y, prefix = 'train')
        self.log('train_loss/step', loss)
        return {'loss': loss, 'metrics' : outputs}
    
    def training_epoch_end(self, outputs):
        for metric in outputs[0]['metrics'].keys():
            self.log(metric, torch.tensor([x['metrics'][metric] for x in outputs]).mean())
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        outputs = self.metrics_logger_custom(predictions = preds, target = y, prefix = 'val')
        outputs['validation_loss'] = loss
        return outputs
    
    def validation_epoch_end(self, outputs):
        for metric in outputs[0].keys():
            self.log(metric, torch.tensor([x[metric] for x in outputs]).mean())

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb_logger = WandbLogger(project='VideoSurveillance_Classiier', log_model='all')
    checkpoint_callback = ModelCheckpoint(monitor="validation_loss")
    early_stop_callback = EarlyStopping(monitor='validation_loss', min_delta=1e-4, patience=5,verbose=True, mode='min')
    model = ResnetClassifier().to(device)
    data_module = FireImageDataLoader(img_dir = './Fire-Detection/', batch_size = 16, num_workers = 8, pin_memory = True)
    trainer = pl.Trainer(gpus = 1, precision=32, logger=wandb_logger, \
        callbacks=[early_stop_callback, checkpoint_callback], progress_bar_refresh_rate=True)

    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()
