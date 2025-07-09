import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchreid import models

class ReIDExtractor:
    def __init__(self, model_path):
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')
            
        self.model = models.build_model('osnet_x0_25', num_classes=1041, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        with torch.no_grad():
            img_t = self.transform(img).unsqueeze(0).to(self.device)
            return self.model(img_t).squeeze(0).cpu().numpy()

def cosine_similarity(f1, f2):
    return F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
