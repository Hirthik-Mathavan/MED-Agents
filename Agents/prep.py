import os
import shutil

# # Specify the root folder containing subfolders with .svs files
root_folder = "./Slides"

# # Specify the destination folder for the organized folders
destination_folder = "./Slides_new"

# Iterate through each subfolder
for subfolder in os.listdir(root_folder):
    subfolder_path = os.path.join(root_folder, subfolder)

    # Check if it's a directory
    if os.path.isdir(subfolder_path):
        # Iterate through .svs files in the subfolder
        for svs_file in os.listdir(subfolder_path):
            if svs_file.endswith(".svs"):
                # Get the first 12 characters of the file name
                folder_name = svs_file[:12]

                # Create a folder path based on the first 12 characters in the destination folder
                folder_path = os.path.join(destination_folder, folder_name)

                # If the folder doesn't exist in the destination folder, create it
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                # Move the .svs file to the folder in the destination folder
                src_path = os.path.join(subfolder_path, svs_file)
                dest_path = os.path.join(folder_path, svs_file)
                shutil.move(src_path, dest_path)

# import pandas as pd


# # df = pd.read_csv('./features2.csv')
# # df['SVS Path'] = df['SVS Path'].str.split('/').str[-1].str[:-3]
# df_t1 = pd.read_csv("./TransPath/clinical.tsv",sep='\t')
# # df_t1 = df.merge(df_t, how='left', left_on='SVS Path', right_on='case_submitter_id')/
# df_t1['days_to_last_follow_up_or_death'] = df_t1['days_to_last_follow_up'].where(df_t1['vital_status'] == 'Alive', df_t1['days_to_death'])
# df_t1 = df_t1[['case_submitter_id','days_to_last_follow_up_or_death']]
# # df_t1.to_csv('ground_Truths.csv', index=False)
# df_t1.to_pickle('ground_Truths.pkl')
# max_files = 0
# s = ''
# # Iterate through each subfolder in the destination folder
# for folder_name in os.listdir(destination_folder):
#     folder_path = os.path.join(destination_folder, folder_name)
    
#     # Count the number of files in the folder
#     num_files = len(os.listdir(folder_path))
#     if num_files > max_files:
#         s = folder_path
#     # Update the maximum number of files if necessary
#         max_files = max(max_files, num_files)

# print("Maximum number of files in any folder:", max_files,s)