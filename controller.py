import os
from GnssTool import  EphemerisManager, gnssTool, parser
from GnssTool import  *

parent_directory = "".join(os.path.split(os.getcwd())[:-1])
root_directory = os.getcwd()
inputs_directory = os.path.join(root_directory,'data','inputs')
outputs_directory= os.path.join(root_directory,'data','outputs')
ephemeris_data_directory = os.path.join(root_directory,'data','outputs', 'ephemeris')

def get_folders(path:str) -> list[str]:
    # Check if the path exists
    if not os.path.exists(path):
        print("The specified path does not exist.")
        return

    # Check if the path is a directory
    if not os.path.isdir(path):
        print("The specified path is not a directory.")
        return

    # Get the list of all files and directories in the specified directory
    contents = os.listdir(path)

    # Filter out only directories from the list
    folders = [item for item in contents if os.path.isdir(os.path.join(path, item))]

    # Print the list of folders
    return folders

def get_files(directory:str) -> list[str]:
    # Check if the directory exists
    if not os.path.exists(directory):
        print("Directory does not exist.")
        return

    # Check if the given directory is a directory
    if not os.path.isdir(directory):
        print("Given path is not a directory.")
        return

    # Get the list of all files in the directory
    return os.listdir(directory)
    print(files)
    

if __name__ == '__main__':
    print(os.getcwd())
    folders = get_folders(inputs_directory)
    for folder in folders:   
        folder_path_in=os.path.join(inputs_directory,folder)
        folder_path_out=os.path.join(outputs_directory,folder)
        files = get_files(folder_path_in)
        for index, filename in enumerate(files):
            if filename.endswith('.txt'):
                file_path = os.path.join(inputs_directory,folder,filename)
                parser.parse_log_file(folder,folder_path_in,filename)
                gt = gnssTool(file_path,ephemeris_data_directory)
                gt.calculate_ECEF_to_file(folder_path_out)
                gt.calculate_LLA(folder_path_out)
                gt.create_kml(folder_path_out)

        
        