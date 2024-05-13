import os
import zipfile
import pandas as pd

def extract_files(zip_path, filenames, output_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for filename in filenames:
            new_filename = f"ortoRgb/{filename}"

            try:
                zip_ref.extract(new_filename, output_folder)
            except KeyError:
                print(f"File '{new_filename}' not found in the zipfile.")

# Specify the path to your zipfile
#zip_path = 'home/circ/Data/SpatialEcology_Lab/Siewert/ortoRgb.zip'
zip_path = 'ortoRgb.zip'


# Specify the list of filenames you want to extract

###################
#CHANGE DIRECTORY!#
###################
names = pd.read_csv('/home/nadjaflechner/palsa_seg/data_prep/filenames_to_use.csv', header=None, names=['files'])
filenames_to_extract = names.files.tolist()

# Specify the output folder where the extracted files will be saved
output_folder = '/home/nadjaflechner/filtered_tifs'

# Create the output folder if it doesn't exist
#os.makedirs(output_folder, exist_ok=True)

# Call the function to extract the files
extract_files(zip_path, filenames_to_extract, output_folder)
