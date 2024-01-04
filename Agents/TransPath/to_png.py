import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()

import cv2

def convert_svs_files_to_png_opencv(input_folder, output_folder, target_size=(512, 512)):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".svs"):
                svs_path = os.path.join(root, file)
                folder_name = os.path.basename(root)
                converted_png_path = convert_svs_to_png_opencv(svs_path, output_folder, folder_name, target_size)
                print(f"SVS converted to PNG: {converted_png_path}")

def convert_svs_to_png_opencv(svs_path, output_folder, folder_name, target_size):
    # Read SVS image using OpenCV
    print("YES")
    image = cv2.imread(svs_path)
    
    # Resize the image to the target size
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Save as PNG in the corresponding output folder
    png_filename = f"{folder_name}_{os.path.splitext(os.path.basename(svs_path))[0]}.png"
    png_path = os.path.join(output_folder, png_filename)
    cv2.imwrite(png_path, image)

    return png_path

# Example usage:
input_folder = '../Slides'  # Replace with the actual path to your Slides folder
output_folder = '../png_files'
target_size = (512, 512)  # Adjust the target size as needed
convert_svs_files_to_png_opencv(input_folder, output_folder, target_size)


# import openslide
# from PIL import Image
# import os

# def convert_svs_files_to_png(input_folder, output_folder, target_size=(512, 512)):
#     for root, dirs, files in os.walk(input_folder):
#         for file in files:
#             if file.endswith(".svs"):
#                 svs_path = os.path.join(root, file)
#                 folder_name = os.path.basename(root)
#                 converted_png_path = convert_svs_to_png(svs_path, output_folder, folder_name, target_size)
#                 print(f"SVS converted to PNG: {converted_png_path}")

# def convert_svs_to_png(svs_path, output_folder, folder_name, target_size):
#     slide = openslide.OpenSlide(svs_path)
#     slide_image = slide.read_region((0, 0), 0, slide.level_dimensions[0])

#     if slide_image.mode != 'RGBA':
#         slide_image = slide_image.convert('RGBA')

#     # Resize the image to the target size
#     slide_image = slide_image.resize(target_size, Image.ANTIALIAS)

#     png_filename = f"{folder_name}_{os.path.splitext(os.path.basename(svs_path))[0]}.png"
#     png_path = os.path.join(output_folder, png_filename)
#     slide_image.save(png_path, format='PNG')

#     return png_path

# Example usage:
# input_folder = '../Slides'  # Replace with the actual path to your Slides folder
# output_folder = '../png_files'
# target_size = (512, 512)  # Adjust the target size as needed
# convert_svs_files_to_png(input_folder, output_folder, target_size)


# import openslide
# from PIL import Image
# import os

# def convert_svs_files_to_png(input_folder, output_folder):
#     for root, dirs, files in os.walk(input_folder):
#         for file in files:
#             if file.endswith(".svs"):
#                 svs_path = os.path.join(root, file)
#                 print(svs_path)
#                 folder_name = os.path.basename(root)
#                 converted_png_path = convert_svs_to_png(svs_path, output_folder, folder_name)
#                 print(f"SVS converted to PNG: {converted_png_path}")

# def convert_svs_to_png(svs_path, output_folder, folder_name):
#     print(svs_path,output_folder,folder_name)
#     slide = openslide.OpenSlide(svs_path)
#     slide_image = slide.read_region((0, 0), 0, slide.level_dimensions[0])

#     if slide_image.mode != 'RGBA':
#         slide_image = slide_image.convert('RGBA')

#     png_filename = f"{folder_name}_{os.path.splitext(os.path.basename(svs_path))[0]}.png"
#     png_path = os.path.join(output_folder, png_filename)
#     slide_image.save(png_path, format='PNG')

#     return png_path

# # Example usage:
# input_folder = '../Slides'  # Replace with the actual path to your Slides folder
# output_folder = '../png_files'
# convert_svs_files_to_png(input_folder, output_folder)


# import os
# import openslide
# from PIL import Image
# from concurrent.futures import ThreadPoolExecutor

# def convert_svs_files_to_png(input_folder, output_folder, target_size=(512, 512), max_workers=None):
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = []
#         for root, dirs, files in os.walk(input_folder):
#             for file in files:
#                 if file.endswith(".svs"):
#                     svs_path = os.path.join(root, file)
#                     folder_name = os.path.basename(root)
#                     futures.append(
#                         executor.submit(convert_svs_to_png, svs_path, output_folder, folder_name, target_size)
#                     )

#         # Wait for all conversions to complete
#         for future in futures:
#             converted_png_path = future.result()
#             print(f"SVS converted to PNG: {converted_png_path}")

# def convert_svs_to_png(svs_path, output_folder, folder_name, target_size):
#     slide = openslide.OpenSlide(svs_path)
#     slide_image = slide.read_region((0, 0), 0, slide.level_dimensions[0])

#     if slide_image.mode != 'RGBA':
#         slide_image = slide_image.convert('RGBA')

#     # Resize the image to the target size
#     slide_image = slide_image.resize(target_size, Image.ANTIALIAS)

#     png_filename = f"{folder_name}_{os.path.splitext(os.path.basename(svs_path))[0]}.png"
#     png_path = os.path.join(output_folder, png_filename)
#     slide_image.save(png_path, format='PNG')

#     return png_path

# # Example usage:
# input_folder = '../Slides'  # Replace with the actual path to your Slides folder
# output_folder = '../png_files'
# target_size = (512, 512)  # Adjust the target size as needed
# max_workers = None  # Adjust the number of parallel workers (None for automatic)

# convert_svs_files_to_png(input_folder, output_folder, target_size, max_workers)
