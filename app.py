import torch                          
import gradio as gr                   
from torchvision import transforms   
from PIL import Image                 
import torch.nn as nn                 
from torchvision import models 


class OCTModel(nn.Module):
    """
        CNN Model for OCT Classification
    """
    def __init__(self, num_classes=4):
        """
        Initializes the pretrained ResNet-18 model and replaces
        the final classification layer.

        Args:
            num_classes (int): Number of OCT output classes.
        """
        super(OCTModel, self).__init__() # Utalizes the nn.Module constructor 

        self.model = models.resnet18(weights=None)
 
        in_features = self.model.fc.in_features #Number of features 

        self.model.fc = nn.Sequential( #replace the final layer
            nn.Dropout(0.5), 
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): Input image batch.

        Returns:
            Tensor: Output logits for each class.
        """
        return self.model(x)
    

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]

DESCRIPTIONS = {                                    
    "CNV":    "Choroidal Neovascularization — The abnormal growth of new blood vessels within the choroid.",
    "DME":    "Diabetic Macular Edema — Diabetes symptom that causes swelling in the macula, the central part of the retina responsible for sharp, detailed vision.",
    "DRUSEN": "Drusen — Small, yellow deposits that accumulate under the retina in the macula, associated with AMD.",
    "NORMAL": "Normal retina — no signs of disease.",
}


########################################################################################## Processing
#Standard ImageNet stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

########################################################################################################## MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
                                                                     

model = OCTModel().to(device)         
model.load_state_dict(torch.load("best_model.pth",map_location=device))

model.eval()

#############################################################################################################
def predict(image: Image.Image):
    """
    Runs inference on an uploaded OCT image.

    Args:
        image (Image.Image): image uploaded by the user.

    Returns:
        tuple: (results, description) 
            - results (dict): probabilities.
            - description (str):  explanation of the top prediction.
    """

    img = image.convert("RGB")                        

    tensor = transform(img)                            
    tensor = tensor.unsqueeze(0)                       
    tensor = tensor.to(device)                         

    with torch.no_grad():                              
        logits = model(tensor)                         
        probs = torch.softmax(logits, dim=1)           
        probs = probs.squeeze()                        
        probs = probs.cpu().tolist()                  

    results = {}                                       
    for cls, prob in zip(CLASSES, probs):              
        results[cls] = round(prob, 4)                  

    top_class = CLASSES[probs.index(max(probs))]       
    description = DESCRIPTIONS[top_class]              

    return results, description                        

###############################################################################################


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload OCT scan"),
    outputs=[
        gr.Label(num_top_classes=4, label="Diagnosis probabilities"),
        gr.Textbox(label="Descrp"),
    ],
    title="OCT Retinal Scan Classifier",
    description=(
        "Upload a retinal OCT scan to classify it into one of four categories: "
        "CNV, DME, DRUSEN, or NORMAL. "
        "Model: ResNet-18"
    ),
    examples=None,
    allow_flagging="never",
)
 
if __name__ == "__main__":
    demo.launch()