import os

parent_directory = "".join(os.path.split(os.getcwd())[:-1])
inputs_directory = os.path.join(parent_directory,'data','inputs')
outputs_directory= os.path.join(parent_directory,'data','outputs')
ephemeris_data_directory = os.path.join(outputs_directory, 'ephemeris')

WEEKSEC = 604800
LIGHTSPEED = 2.99792458e8