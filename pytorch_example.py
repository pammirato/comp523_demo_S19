import torch
import torchvision.models as models
from torchvision import transforms as T
import numpy as np
from PIL import Image #might want to use this instead of numpy
#TODO use PIL images instead of numpy



class DummyNetwork(torch.nn.Module):


    def __init__(self, scene_image_transform=None, target_image_transform=None,
                 device=None):
        super(DummyNetwork,self).__init__()
        vgg = models.vgg16_bn(pretrained=False)
        self.backbone = torch.nn.Sequential(*list(vgg.features.children())[:-1])
        self.scene_image_transform = scene_image_transform
        self.target_image_transform = target_image_transform
        self.device = device


    def forward(self,scene_image,target_images):
    
        #assumes images are already pytorch Tensors 
        #run the images through the backbone model
        scene_features = self.backbone(scene_image)
        target_features = self.backbone(target_images)

        #in the real model we would do more computation for object detection

        #now just output something silly for testing purposes
        num_rows = scene_image.shape[0]
        num_cols = scene_image.shape[1]
        boxes = [[0,0,num_cols//4, num_rows//4],
            [num_cols//2,num_rows//2,num_cols-1,num_rows-1]]
        scores = [.8,.4]
        return boxes,scores

    def do_object_detection(self,scene_image,target_images):
        '''
            scene_image: a single Height x Width x Channel (numpy) image
            target_images: a list of H x W x C  (numpy) images

            Channel is RGB, so should be 3
            
        '''
        #convert the images to the format the object deteciton model needs
        if not(self.scene_image_transform is None):
            scene_image = self.scene_image_transform(scene_image)
        if not(self.target_image_transform is None):
            transformed_target_images = []
            for target_image in target_images: 
                transformed_target_images.append(self.target_image_transform(target_image))

        #concatenate the target images to a single matrix
        transformed_target_images = torch.stack(transformed_target_images,dim=0)
        #add a batch dimension to the scene image
        scene_image = scene_image.unsqueeze(0)
        #put the images on the gpu if available
        scene_image = scene_image.to(self.device).float()
        transformed_target_images = transformed_target_images.to(self.device).float()

        boxes, scores = self(scene_image,transformed_target_images)
        return boxes,scores






def main():

    model_parameters_filename = './model.pth'   

    #get gpu if available, otherwise use CPU
    #uncomment second line if you want to try to use a gpu
    device = torch.device("cpu")
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    #image transforms for converting from numpy to pytorch format
    scene_transform = T.Compose([
                                  T.ToTensor(), 
                                  T.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225]),
                                ]) 
    target_transform = T.Compose([
                                  T.ToTensor(), 
                                  T.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225]),
                                ]) 
 
    #init the model object
    model = DummyNetwork(
                            scene_image_transform=scene_transform,
                            target_image_transform=target_transform,
                            device=device, 
                        )
    #load the model parameters
    model.load_state_dict(torch.load(model_parameters_filename).state_dict())
    model = model.to(device)

    #make blank images
    scene_image = np.zeros((200,200,3))
    #target images must be the same size (for now)
    target_images = []
    target_images.append(np.zeros((100,100,3)))
    target_images.append(np.zeros((100,100,3)))

    boxes,scores = model.do_object_detection(scene_image,target_images)
    print(boxes, scores)



if __name__ == "__main__":
    main()

