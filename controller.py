import os
from GnssTool import  EphemerisManager, gnssTool, parser
from GnssTool import  *

parent_directory = "".join(os.path.split(os.getcwd())[:-1])
"""parent_directory: Computes the parent directory of the current working directory.
"""
root_directory = os.getcwd()
"""root_directory: Retrieves the current working directory.
"""
inputs_directory = os.path.join(root_directory,'data','inputs')
"""inputs_directory: Constructs the path to the directory where input GNSS data files are located.
"""
outputs_directory= os.path.join(root_directory,'data','outputs')
"""outputs_directory: Constructs the path to the directory where output files will be saved.
"""
ephemeris_data_directory = os.path.join(root_directory,'data','outputs', 'ephemeris')
"""ephemeris_data_directory: Constructs the path to the directory where ephemeris data is stored.
"""
def get_folders(path:str) -> list[str]:
    """
    Description: 
        Retrieves a list of folders (subdirectories) within the specified directory path.
    Parameters:
        path: The path of the directory from which to retrieve folders.
    Returns:
        A list of folder names (strings) within the specified directory.
    """
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
    """
    Description: 
        Retrieves a list of files within the specified directory path.
    Parameters:
        directory: The path of the directory from which to retrieve files.
    Returns:
        A list of file names (strings) within the specified directory.
    """
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
    """
    The script checks the current working directory and retrieves a list of folders within the inputs_directory.
    For each folder, it processes the GNSS data files (with .txt extension) by parsing them and performing calculations using the gnssTool class.
    The parsed data is then saved to output files in the corresponding output directory.
    """
folders = get_folders(inputs_directory)

# Process each folder
for folder in folders:
    folder_path_in = os.path.join(inputs_directory, folder)
    folder_path_out = os.path.join(outputs_directory, folder)
    
    # Get the list of files in the input folder
    files = get_files(folder_path_in)
    
    # Process each file
    for filename in files:
        if filename.endswith('.txt'):
            file_path = os.path.join(inputs_directory, folder, filename)
            
            # Parse the log file
            parser.parse_log_file(folder, folder_path_in, filename)
            
            # Initialize the gnssTool instance
            gt = gnssTool(file_path, ephemeris_data_directory)
            
            # Perform calculations and generate output files
            gt.calculate_ECEF_to_file(folder_path_out)
            gt.calculate_LLA(folder_path_out)
            gt.create_kml(folder_path_out)

        
        