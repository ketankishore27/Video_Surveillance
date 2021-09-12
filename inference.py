import torch, cv2, copy, numpy as np, pytorch_lightning as pl, torch.nn as nn, torchmetrics, torch
from torchvision.models import resnet50
from torchvision.transforms import transforms

transformations_test = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224),
                                           transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

model = torch.hub.load('ultralytics/yolov5', 'yolov5l').cuda()

model_custom = torch.hub.load('ultralytics/yolov5', 'custom', path='best_from_internet.pt').cuda()

objectMapping = {model.names[i]:i for i in range(len(model.names))}
reverseObjectMapping = {str(i):model.names[i] for i in range(len(model.names))}
colormappingBounding = {
    "person": (255,215,0),
    "car": 	(165,42,42),
    "motorcycle": (139,69,19),
    "bus": 	(255,140,0),
    "truck": (0,255,255),
} 

def load_classifier():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResnetClassifier().to(device)
    model = model.load_from_checkpoint('fire_classifier.ckpt')
    return model.eval()

def image_loader(image):
    image = transformations_test(image).float().unsqueeze(0)
    return image

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

writer = None
cap = cv2.VideoCapture("videos/sample7.mp4")
model_classifier = load_classifier()
#print(cap.isOpened())
while cap.isOpened():
    imagedump=[]
    ret,frame=cap.read()
    if not ret:
        break
    imageDuplicate = copy.deepcopy(frame)
    results = model.forward(frame) if frame is not None else None
    result_custom = model_custom.forward(frame) if frame is not None else None
    with torch.no_grad():
        result_classifier = model_classifier(image_loader(frame)).item()
    #results_fire = model_fire.forward(frame)  if frame is not None else None
    if frame.any()==None:
        print("none")
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
    for detection in results.xywh[0].cpu().detach().numpy():
        classId = int(detection[-1])
        confidenceScore = round(detection[4] * 100, 2)
        if classId in [objectMapping['person'], objectMapping['car'], objectMapping['motorcycle'], 
                       objectMapping['bus'], objectMapping['truck']] and confidenceScore > 60:
            (centerX, centerY, width, height) = detection[:4]
            startx = int(centerX - (width/2))
            starty = int(centerY - (height/2))
            endx = int(startx + width)
            endy = int(starty + height)
            imageDuplicate = np.array(imageDuplicate)
            imageDuplicate = cv2.rectangle(img = imageDuplicate, pt1 = (startx, starty), pt2 = (endx, endy), color = colormappingBounding[reverseObjectMapping[str(classId)]], thickness = 2)
            cv2.putText(imageDuplicate, reverseObjectMapping[str(classId)]+ " "+str(confidenceScore), (startx, starty-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 2)

    for detection in result_custom.xywh[0].cpu().detach().numpy():
        classId = int(detection[-1])
        confidenceScore = round(detection[4] * 100, 2)
        if confidenceScore > 25:
            (centerX, centerY, width, height) = detection[:4]
            startx = int(centerX - (width/2))
            starty = int(centerY - (height/2))
            endx = int(startx + width)
            endy = int(starty + height)
            cv2.rectangle(img = imageDuplicate, pt1 = (startx, starty), pt2 = (endx, endy), color = (0,0,255), thickness = 2)
            #cv2.putText(imageDuplicate, "Gun " + str(confidenceScore), (startx, starty-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 2)
            cv2.putText(imageDuplicate, "Fire " + str(confidenceScore), (startx, starty-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 2)

    cv2.putText(imageDuplicate, "Fire_Possibility: " + str(round(result_classifier*100)), (850, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("video",imageDuplicate)
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("SampleOutput.avi", fourcc, 15, (imageDuplicate.shape[1], imageDuplicate.shape[0]), True)
    writer.write(imageDuplicate)
    
    #print(".", sep = "", end = "")
            
cap.release()
cv2.destroyAllWindows()