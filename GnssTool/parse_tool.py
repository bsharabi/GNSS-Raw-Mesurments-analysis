import pandas as pd
import csv
import math
from datetime import datetime
import os

parent_directory = os.getcwd()
inputs_directory = os.path.join(parent_directory, 'data', 'inputs')
outputs_directory = os.path.join(parent_directory, 'data', 'outputs')


class parser:
    """
    A class to parse NMEA data and log files.

    Attributes:
        None
    """

    @staticmethod
    def nmea_to_csv(nmea_data: list, filename: str) -> None:
        """
        Converts NMEA data to CSV format and writes it to a file.

        Args:
            nmea_data (list): The NMEA data to convert.
            filename (str): The name of the output CSV file.

        Returns:
            None
        """
        # Open the output file in write mode
        with open(filename, 'wt') as output_file:
            # Create a csv writer object for the output file
            writer = csv.writer(output_file, delimiter=',', lineterminator='\n')
            # Write the header line to the csv file
            writer.writerow(['date_and_time', 'lat', 'lon', 'speed'])
            # Iterate over all the rows in the NMEA data
            for row in nmea_data:
                # Skip all lines that do not start with $GPRMC
                if row[0].startswith('$GNRMC'):
                    # Fetch the values from the row's columns
                    time = row[1]
                    warning = row[2]
                    lat = row[3]
                    lat_direction = row[4]
                    lon = row[5]
                    lon_direction = row[6]
                    speed = row[7]
                    date = row[9]
                    # Skip rows with warning 'V'
                    if warning == 'V':
                        continue
                    # Merge the time and date columns into one Python datetime object
                    date_and_time = datetime.strptime(date + ' ' + time, '%d%m%y %H%M%S.%f')
                    date_and_time = date_and_time.strftime('%y-%m-%d %H:%M:%S.%f')[:-3]
                    # Convert lat and lon values to decimal degree format
                    lat = round(math.floor(float(lat) / 100) + (float(lat) % 100) / 60, 6)
                    if lat_direction == 'S':
                        lat = lat * -1
                    lon = round(math.floor(float(lon) / 100) + (float(lon) % 100) / 60, 6)
                    if lon_direction == 'W':
                        lon = lon * -1
                    # Convert speed from knots to km/h
                    speed = float(speed) * 1.15078
                    # Write the calculated/formatted values to the csv file
                    writer.writerow([date_and_time, lat, lon, speed])

    @staticmethod
    def parse_log_file(folder_name: str, folder_path: str, filename: str) -> None:
        """
        Parses a log file and saves the data into CSV files.

        Args:
            folder_name (str): Name of the folder to save parsed files.
            folder_path (str): Path to the folder containing the log file.
            filename (str): Name of the log file to parse.

        Returns:
            None
        """
        input_filename_noext = os.path.splitext(filename)[0]
        with open(os.path.join(folder_path, filename)) as csvfile:
            reader = csv.reader(csvfile)
            data = {'NMEA': []}
            for row in reader:
                if row[0][0] == '#':
                    if 'Version' in row[0] or 'Header' in row[0] or len(row[0]) == 2:
                        pass
                    elif len(row[0]) > 1:
                        data[row[0][2:]] = [row[1:]]
                else:
                    data[row[0]].append(row[1:])
        nmea_data = data.pop('NMEA')
        output_directory = os.path.join(outputs_directory, folder_name)
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        output_directory = os.path.join(outputs_directory, folder_name, input_filename_noext)
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        for key, values in data.items():
            data[key] = pd.DataFrame(values[1:], columns=values[0])
            data[key].to_csv(os.path.join(output_directory, key + '.csv'), index=False)
        parser.nmea_to_csv(nmea_data, os.path.join(output_directory, 'NMEA.csv'))
