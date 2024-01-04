# import pandas as pd
# import torch, torchvision
# import torch.nn as nn
# from torchvision import transforms
# from PIL import Image
# from torch.utils.data import Dataset
# from ctran import ctranspath


# mean = (0.485, 0.456, 0.406)
# std = (0.229, 0.224, 0.225)
# trnsfrms_val = transforms.Compose(
#     [
#         transforms.Resize((224,224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean = mean, std = std)
#     ]
# )
# class roi_dataset(Dataset):
#     def __init__(self, img_csv,
#                  ):
#         super().__init__()
#         self.transform = trnsfrms_val

#         self.images_lst = img_csv

#     def __len__(self):
#         return len(self.images_lst)

#     def __getitem__(self, idx):
#         path = self.images_lst.filename[idx]
#         image = Image.open(path).convert('RGB')
#         image = self.transform(image)


#         return image

# img_csv=pd.read_csv(r'./test_list.csv')
# test_datat=roi_dataset(img_csv)
# database_loader = torch.utils.data.DataLoader(test_datat, batch_size=1, shuffle=False)

# model = ctranspath()
# model.head = nn.Identity()
# td = torch.load(r'./ctranspath.pth')
# model.load_state_dict(td['model'], strict=False)


# model.eval()
# with torch.no_grad():
#     for batch in database_loader:
#         features = model(batch)
#         features = features.cpu().numpy()
#         print(features)


# #
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from ctran import ctranspath
import openslide

# mean = (0.485, 0.456, 0.406)
# std = (0.229, 0.224, 0.225)
# trnsfrms_val = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=mean, std=std)
# ])

# class SVSDataset(Dataset):
#     def __init__(self, img_csv):
#         super().__init__()
#         self.transform = trnsfrms_val
#         self.images_df = img_csv

#     def __len__(self):
#         return len(self.images_df)

#     def __getitem__(self, idx):
#         svs_path = self.images_df.filename[idx]
#         image = self.convert_svs_to_image(svs_path)
#         image = self.transform(image)
#         return image

#     def convert_svs_to_image(self, svs_path):
#         slide = openslide.OpenSlide(svs_path)
#         slide_image = slide.read_region((0, 0), 0, slide.level_dimensions[0])

#         if slide_image.mode != 'RGB':
#             slide_image = slide_image.convert('RGB')

#         # Resize the image to the target size
#         slide_image = slide_image.resize((224, 224), Image.Resampling.LANCZOS)
#         return slide_image

# # Load SVS file paths from CSV
# img_csv = pd.read_csv(r'./test_list.csv')

# # Create SVS dataset
# svs_dataset = SVSDataset(img_csv)

# # Create DataLoader
# database_loader = torch.utils.data.DataLoader(svs_dataset, batch_size=1, shuffle=False)

# # Load the model
# model = ctranspath()
# model.head = nn.Identity()
# td = torch.load(r'./ctranspath.pth')
# model.load_state_dict(td['model'], strict=False)

# # Set the model to evaluation mode
# model.eval()

# # Run the dataset through the model
# with torch.no_grad():
#     for batch in database_loader:
#         features = model(batch)
#         features = features.cpu().numpy()
#         print(features)

# import os
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image
# from ctran import ctranspath
# import openslide

# mean = (0.485, 0.456, 0.406)
# std = (0.229, 0.224, 0.225)
# trnsfrms_val = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=mean, std=std)
# ])

# class roi_dataset(Dataset):
#     def __init__(self, folder_path):
#         super().__init__()
#         self.transform = trnsfrms_val
#         self.image_paths = self.get_image_paths(folder_path)

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]

#         # Convert SVS to PNG and get PIL image
#         image = convert_svs_to_png(image_path)

#         # Transform PIL image
#         image = self.transform(image)

#         return image

#     def get_image_paths(self, folder_path):
#         image_paths = []
#         for root, dirs, files in os.walk(folder_path):
#             for file in files:
#                 if file.endswith(".svs"):
#                     svs_path = os.path.join(root, file)
#                     image_paths.append(svs_path)
#         return image_paths

# def convert_svs_to_png(svs_path):
#     print(svs_path)
#     slide = openslide.OpenSlide(svs_path)
#     slide_image = slide.read_region((0, 0), 0, slide.level_dimensions[0])

#     if slide_image.mode != 'RGB':
#         slide_image = slide_image.convert('RGB')

#     # Resize the image to the target size
#     slide_image = slide_image.resize((224, 224),Image.Resampling.LANCZOS)

#     return slide_image

# # Example usage:
# folder_path = '../Slides'  # Replace with the actual path to your Slides folder
# test_datat = roi_dataset(folder_path)
# database_loader = torch.utils.data.DataLoader(test_datat, batch_size=1, shuffle=False)

# model = ctranspath()
# model.head = nn.Identity()
# td = torch.load(r'./ctranspath.pth')
# model.load_state_dict(td['model'], strict=False)

# model.eval()
# with torch.no_grad():
#     for batch in database_loader:
#         features = model(batch)
#         features = features.cpu().numpy()
#         print(features)






import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ctran import ctranspath
import openslide

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

class roi_dataset(Dataset):
    def __init__(self, folder_path):
        super().__init__()
        self.transform = trnsfrms_val
        self.image_paths = self.get_image_paths(folder_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Convert SVS to PNG and get PIL image
        image = convert_svs_to_png(image_path)

        # Transform PIL image
        image = self.transform(image)

        return image, image_path

    def get_image_paths(self, folder_path):
        image_paths = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".svs"):
                    svs_path = os.path.join(root, file)
                    image_paths.append(svs_path)
        return image_paths

def convert_svs_to_png(svs_path):
    print(svs_path)
    slide = openslide.OpenSlide(svs_path)
    slide_image = slide.read_region((0, 0), 0, slide.level_dimensions[0])

    if slide_image.mode != 'RGB':
        slide_image = slide_image.convert('RGB')

    # Resize the image to the target size
    slide_image = slide_image.resize((224, 224), Image.Resampling.LANCZOS)

    return slide_image

# Example usage:
folder_path = '../Slides'  # Replace with the actual path to your Slides folder
test_datat = roi_dataset(folder_path)
database_loader = torch.utils.data.DataLoader(test_datat, batch_size=1, shuffle=False)

model = ctranspath()
model.head = nn.Identity()
td = torch.load(r'./ctranspath.pth')
model.load_state_dict(td['model'], strict=False)

output_file_path = 'features.txt'

model.eval()

with torch.no_grad():
    for batch, svs_paths in database_loader:
        features = model(batch)
        features = features.cpu().numpy().squeeze()
        output_file = open(output_file_path, 'w')
        # Write svs path and features to the text 
        # print(f'SVS Path: {svs_paths}\nFeatures: {",".join(map(str, features))}\n', file=output_file)
        print(svs_paths,features)
        # output_file.write("YES")
        output_file.write(f'SVS Path: {svs_paths[0]}\nFeatures: {",".join(map(str, features))}\n\n')
        output_file.close()
        











