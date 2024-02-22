import torch
from torchvision import transforms
from PIL import Image
from torchvision.models.resnet import Bottleneck, ResNet
import openslide
import os
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
    }
    # pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    pretrained_url = "{}/{}".format(URL_PREFIX, model_zoo_registry.get(key))

    return pretrained_url

def resnet50(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model

def convert_svs_to_png(svs_path, output_png_path):
    # slide = openslide.OpenSlide(svs_path)
    # image = slide.read_region((0, 0), 0, slide.level_dimensions[0])
    # image = image.convert("RGB")
    # image.save(output_png_path)
    print("Inside")
    slide = openslide.OpenSlide(svs_path)
    slide_image = slide.read_region((0, 0), 0, slide.level_dimensions[0])

    if slide_image.mode != 'RGB':
        slide_image = slide_image.convert('RGB')

    # Resize the image to the target size
    slide_image = slide_image.resize((224, 224), Image.Resampling.LANCZOS)
    
    slide_image.save(output_png_path)
    # return slide_image

if __name__ == "__main__":
    # Convert SVS to PNG
    # svs_path = "../Slides1/TCGA-49-4501/TCGA-49-4501-01Z-00-DX3.b6c2cc84-1c94-4816-92e7-8cf4446ac9ac.svs"
    # output_png_path = "image_check.png"
    # # convert_svs_to_png(svs_path, output_png_path)

    # # Load the PNG file
    # image = Image.open("./1.png").convert('RGB')
    # print("Loaded")
    # # Define the image transformation
    # preprocess = transforms.Compose([
    #     transforms.Resize((256, 256)),  # Resize to match the expected input size
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    # # Apply the transformation to the image
    # input_tensor = preprocess(image)
    # input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # # Initialize the ResNet50 model using BT pre-trained weights
    # model = resnet50(pretrained=True, progress=False, key="BT")

    # # Set the model to evaluation mode
    # model.eval()

    # # Forward pass to extract features
    # with torch.no_grad():
    #     features = model(input_batch)
    # print(features)
    # # Do something with the extracted features
    # print("Extracted features shape:", features.shape)
    
    import os
    import torch
    from torchvision import transforms
    from PIL import Image
    from torchvision.models.resnet import Bottleneck, ResNet
    import pandas as pd
    import openslide
    
    # Initialize ResNet50 model
    model = resnet50(pretrained=True, progress=False, key="BT").to(device)
    model.eval()
    
    # Define the image transformation
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Assume you have a root folder containing TCGA_X folders
    root_folder = "../Slides_new"
    
    features_dict = {}

    # Loop through TCGA_X folders
    for tcga_folder in os.listdir(root_folder):
        tcga_path = os.path.join(root_folder, tcga_folder)
        
        if os.path.isdir(tcga_path):
            # Collect SVS files in the TCGA_X folder
            svs_files = [f for f in os.listdir(tcga_path) if f.endswith('.svs')]
            
            # Extract features for each SVS file
            features_list = []
            for svs_file in svs_files:
                svs_path = os.path.join(tcga_path, svs_file)
    
                # Open the SVS file directly without saving to PNG
                slide = openslide.OpenSlide(svs_path)
                image = slide.read_region((0, 0), 0, slide.level_dimensions[0])
                image = image.convert("RGB")
    
                # Extract features
                input_tensor = preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = model(input_tensor)
    
                # Append the features to the list
                features_flat = features.view(1, -1).numpy()
                print(features_flat.flatten())
                features_list.append(features_flat.flatten())
    
            # Convert features list to a Pandas DataFrame
            df = pd.DataFrame(data=[features_list], columns=[f"Feature_{i+1}" for i in range(len(features_list))])
            df1 = pd.DataFrame(data=[tcga_folder], columns=["Case_ID"])
            # df1['Case_ID'] = tcga_folder
            print(df)
            # Save the DataFrame to a dictionary using the TCGA_X folder as the key
            features_dict[tcga_folder] = df
    
    # Combine all DataFrames into a single DataFrame
    result_df = pd.concat(features_dict.values(), keys=features_dict.keys(), names=['Case_ID'])
    # result_df.columns.values[0] = 'Case_ID'
    
    print(result_df['Feature_1'])
    print("CHECK",result_df.columns)
    
    # Optionally, save the DataFrame to a CSV file
    result_df.to_csv("output_dataframe.csv")
    result_df.to_pickle('output_dataframe.pkl')
    feature_list_new = result_df.apply(
        lambda row: torch.tensor(row.dropna().tolist(), dtype=torch.float32),
        axis=1
    ).tolist()
    
    print(feature_list_new)
    result_df['feature_list'] = feature_list_new
    
    
    
    with open('../survival_status.pkl', 'rb') as f:
        loaded_dictionary = pickle.load(f)
        
        
    df_Truths = pd.read_pickle("../ground_Truths.pkl")
    df_Truths = df_Truths.drop_duplicates()
    print(df_Truths.columns)
    df_Truths['days_to_last_follow_up_or_death'] = pd.to_numeric(df_Truths['days_to_last_follow_up_or_death'], errors='coerce')
    result_df = result_df.merge(df_Truths, how='left', left_on='Case_ID', right_on='case_submitter_id')
    print(df_Truths.columns)
    print(result_df.columns)
    print(df1)
    sta = []
    for i in df1['Case_ID']:
        if i in loaded_dictionary:
            sta.append(int(loaded_dictionary[i]))
    # Extract specific rows based on index or condition
    # rows_to_extract = [0, 2]  # Replace with the index of the rows you want to extract
    
    # Modify the DataFrame in-place
    # df_Truths = df_Truths.iloc[rows_to_extract]
    # df_Truths['days_to_last_follow_up_or_death'] = pd.to_numeric(df_Truths['days_to_last_follow_up_or_death'], errors='coerce')

    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    from lifelines.utils import concordance_index
    
    # Assuming your dataframe is named df and contains a column 'features' with variable-sized arrays
    # Also, assuming 'target' is the column you want to predict
    
    # Convert the 'features' column to a list of tensors
    features = result_df['feature_list'].apply(lambda x: torch.tensor(x, dtype=torch.float32)).tolist()
    
    # Assuming all sequences have the same feature dimension (e.g., 512)
    input_size = len(features[0][0])
    
    # Pad sequences to the maximum length
    max_length = max(len(seq) for seq in features)
    padded_features = [torch.nn.functional.pad(seq, (0, 0, 0, max_length - len(seq))) for seq in features]
    
    # Stack the padded sequences
    padded_features = torch.stack(padded_features)
    
    # Assuming 'target' is the column you want to predict
    target = torch.tensor(result_df['days_to_last_follow_up_or_death'].values, dtype=torch.float32)
    padded_features = padded_features.to(device)
    target = target.to(device)
    # Split the data into training and validation sets
    padded_features_train, padded_features_val, target_train, target_val = train_test_split(
        padded_features, target, test_size=0.4, random_state=42
    )
    
    # Define a simple LSTM-based regression model
    class RegressionModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RegressionModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            return self.fc(h_n[-1])
    
    # Instantiate the model
    hidden_size = 64  # Adjust as needed
    output_size = 1   # Regression output size (adjust as needed)
    model = RegressionModel(input_size, hidden_size, output_size)
    model = model.to(device)
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Define DataLoader for training
    train_dataset = torch.utils.data.TensorDataset(padded_features_train, target_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        for batch_features, batch_target in train_loader:
            # Zero the gradients
            batch_features, batch_target = batch_features.to(device), batch_target.to(device)
            optimizer.zero_grad()
    
            # Forward pass
            predictions = model(batch_features)
    
            # Compute the loss
            loss = criterion(predictions.squeeze(), batch_target)
    
            # Backward pass
            loss.backward()
    
            # Update the parameters
            optimizer.step()
    
        # Print the loss for monitoring
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
    
    # Validation
    with torch.no_grad():
        predictions_val = model(padded_features_val)
        predictions_val = predictions_val.squeeze()
        c_index_val = concordance_index(target_val, -predictions_val,sta)
        print(f'Validation C-index: {c_index_val}')
        # Now, 'result_df' contains the desired structure for regression
        # print(result_df)
    torch.save(model, "SSL.pt")
