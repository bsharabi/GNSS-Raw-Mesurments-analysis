import csv
import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import ndarray
from datetime import datetime, timedelta
import simplekml
from GnssTool import  EphemerisManager
import navpy
import os

WEEKSEC = 604800
LIGHTSPEED = 2.99792458e8

parent_directory = os.getcwd()
inputs_directory = os.path.join(parent_directory,'data','inputs')
outputs_directory= os.path.join(parent_directory,'data','outputs')
ephemeris_data_directory = os.path.join(parent_directory, 'data', 'outputs', 'ephemeris')

class gnssTool:
    """
    A class for GNSS data processing and analysis.

    Attributes:
        manager (EphemerisManager): An instance of EphemerisManager class.
        measurements (DataFrame): Dataframe containing GNSS measurements.
        ECEF_DATA (list): List of dictionaries containing ECEF data.
        ecef_list (list): List to store ECEF coordinates.
        lla_list (list): List to store LLA coordinates.
        lla_df (DataFrame): DataFrame containing LLA coordinates.
    """
    def __init__(self, input_filepath: str, ephemeris_data_directory: str) -> None:
        """
        Initializes a gnssTool object.

        Args:
            input_filepath (str): Path to the input file.
            ephemeris_data_directory (str): Path to the directory containing ephemeris data.

        Returns:
            None
        """
        self.manager = EphemerisManager(ephemeris_data_directory)
        self.measurements = self.__gnss_to_dataFrame(input_filepath)
        self.ECEF_DATA=[]
        self.ecef_list = []
        self.lla_list = []
        self.lla_df=None
        
    def __gnss_to_dataFrame(self, input_filepath: str) -> DataFrame:
        """
        Converts GNSS data to a pandas DataFrame.

        Args:
            input_filepath (str): Path to the input file.

        Returns:
            DataFrame: DataFrame containing GNSS measurements.
        """
        with open(input_filepath) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0][0] == '#':
                    if 'Fix' in row[0]:
                        android_fixes = [row[1:]]
                    elif 'Raw' in row[0]:
                        measurements = [row[1:]]
                else:
                    if row[0] == 'Fix':
                        android_fixes.append(row[1:])
                    elif row[0] == 'Raw':
                        measurements.append(row[1:])

        android_fixes = DataFrame(android_fixes[1:], columns=android_fixes[0])
        measurements = DataFrame(measurements[1:], columns=measurements[0])

        # Format satellite IDs
        measurements.loc[measurements['Svid'].str.len() == 1, 'Svid'] = '0' + measurements['Svid']
        measurements.loc[measurements['ConstellationType'] == '1', 'Constellation'] = 'G'
        measurements.loc[measurements['ConstellationType'] == '3', 'Constellation'] = 'R'
        measurements['SvName'] = measurements['Constellation'] + measurements['Svid']

        # Remove all non-GPS measurements
        measurements = measurements.loc[measurements['Constellation'] == 'G']

        # Convert columns to numeric representation
        measurements['Cn0DbHz'] = pd.to_numeric(measurements['Cn0DbHz'])
        measurements['TimeNanos'] = pd.to_numeric(measurements['TimeNanos'])
        measurements['FullBiasNanos'] = pd.to_numeric(measurements['FullBiasNanos'])
        measurements['ReceivedSvTimeNanos'] = pd.to_numeric(measurements['ReceivedSvTimeNanos'])
        measurements['PseudorangeRateMetersPerSecond'] = pd.to_numeric(measurements['PseudorangeRateMetersPerSecond'])
        measurements['ReceivedSvTimeUncertaintyNanos'] = pd.to_numeric(measurements['ReceivedSvTimeUncertaintyNanos'])

        # A few measurement values are not provided by all phones
        # We'll check for them and initialize them with zeros if missing
        if 'BiasNanos' in measurements.columns:
            measurements['BiasNanos'] = pd.to_numeric(measurements['BiasNanos'])
        else:
            measurements['BiasNanos'] = 0
        if 'TimeOffsetNanos' in measurements.columns:
            measurements['TimeOffsetNanos'] = pd.to_numeric(measurements['TimeOffsetNanos'])
        else:
            measurements['TimeOffsetNanos'] = 0

        measurements['GpsTimeNanos'] = measurements['TimeNanos'] - (measurements['FullBiasNanos'] - measurements['BiasNanos'])
        gpsepoch = datetime(1980, 1, 6, 0, 0, 0)
        measurements['UnixTime'] = pd.to_datetime(measurements['GpsTimeNanos'], utc=True, origin=gpsepoch)
        measurements['UnixTime'] = measurements['UnixTime']

        # Split data into measurement epochs
        measurements['Epoch'] = 0
        measurements.loc[measurements['UnixTime'] - measurements['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
        measurements['Epoch'] = measurements['Epoch'].cumsum()

        measurements['Pseudo-Range'] = (measurements['TimeNanos'] - measurements['FullBiasNanos'] - measurements['ReceivedSvTimeNanos']) / LIGHTSPEED

        # This should account for rollovers since it uses a week number specific to each measurement
        measurements['tRxGnssNanos'] = measurements['TimeNanos'] + measurements['TimeOffsetNanos'] - (measurements['FullBiasNanos'].iloc[0] + measurements['BiasNanos'].iloc[0])
        measurements['GpsWeekNumber'] = np.floor(1e-9 * measurements['tRxGnssNanos'] / WEEKSEC)
        measurements['tRxSeconds'] = 1e-9 * measurements['tRxGnssNanos'] - WEEKSEC * measurements['GpsWeekNumber']
        measurements['tTxSeconds'] = 1e-9 * (measurements['ReceivedSvTimeNanos'] + measurements['TimeOffsetNanos'])
        # Calculate pseudorange in seconds
        measurements['prSeconds'] = measurements['tRxSeconds'] - measurements['tTxSeconds']

        # Convert to meters
        measurements['PrM'] = LIGHTSPEED * measurements['prSeconds']
        measurements['PrSigmaM'] = LIGHTSPEED * 1e-9 * measurements['ReceivedSvTimeUncertaintyNanos']

        return measurements

    def __calculate_satellite_position(self, ephemeris:DataFrame, transmit_time:DataFrame) -> DataFrame:
        """
        Calculates satellite positions.

        Args:
            ephemeris (): Ephemeris data.
            transmit_time (): Transmission time.

        Returns:
            DataFrame: DataFrame containing satellite positions.
        """
        mu = 3.986005e14
        OmegaDot_e = 7.2921151467e-5
        F = -4.442807633e-10
        sv_position = DataFrame()
        sv_position['sv'] = ephemeris.index
        sv_position.set_index('sv', inplace=True)
        sv_position['t_k'] = transmit_time - ephemeris['t_oe']
        A = ephemeris['sqrtA'].pow(2)
        n_0 = np.sqrt(mu / A.pow(3))
        n = n_0 + ephemeris['deltaN']
        M_k = ephemeris['M_0'] + n * sv_position['t_k']
        E_k = M_k
        err = pd.Series(data=[1]*len(sv_position.index))
        i = 0
        while err.abs().min() > 1e-8 and i < 10:
            new_vals = M_k + ephemeris['e']*np.sin(E_k)
            err = new_vals - E_k
            E_k = new_vals
            i += 1

        sinE_k = np.sin(E_k)
        cosE_k = np.cos(E_k)
        delT_r = F * ephemeris['e'].pow(ephemeris['sqrtA']) * sinE_k
        delT_oc = transmit_time - ephemeris['t_oc']
        sv_position['delT_sv'] = ephemeris['SVclockBias'] + ephemeris['SVclockDrift'] * delT_oc + ephemeris['SVclockDriftRate'] * delT_oc.pow(2)

        v_k = np.arctan2(np.sqrt(1-ephemeris['e'].pow(2))*sinE_k, (cosE_k - ephemeris['e']))

        Phi_k = v_k + ephemeris['omega']

        sin2Phi_k = np.sin(2*Phi_k)
        cos2Phi_k = np.cos(2*Phi_k)

        du_k = ephemeris['C_us']*sin2Phi_k + ephemeris['C_uc']*cos2Phi_k
        dr_k = ephemeris['C_rs']*sin2Phi_k + ephemeris['C_rc']*cos2Phi_k
        di_k = ephemeris['C_is']*sin2Phi_k + ephemeris['C_ic']*cos2Phi_k

        u_k = Phi_k + du_k

        r_k = A*(1 - ephemeris['e']*np.cos(E_k)) + dr_k

        i_k = ephemeris['i_0'] + di_k + ephemeris['IDOT']*sv_position['t_k']

        x_k_prime = r_k*np.cos(u_k)
        y_k_prime = r_k*np.sin(u_k)

        Omega_k = ephemeris['Omega_0'] + (ephemeris['OmegaDot'] - OmegaDot_e)*sv_position['t_k'] - OmegaDot_e*ephemeris['t_oe']

        sv_position['x_k'] = x_k_prime*np.cos(Omega_k) - y_k_prime*np.cos(i_k)*np.sin(Omega_k)
        sv_position['y_k'] = x_k_prime*np.sin(Omega_k) + y_k_prime*np.cos(i_k)*np.cos(Omega_k)
        sv_position['z_k'] = y_k_prime*np.sin(i_k)
        return sv_position

    def __least_squares(self, xs:ndarray, measured_pseudorange:ndarray, x0:ndarray, b0:int) -> tuple:
        """
        Performs least squares estimation.

        Args:
            xs (): Input coordinates.
            measured_pseudorange (): Measured pseudorange.
            x0 (): Initial guess for coordinates.
            b0 (): Initial guess for clock bias.

        Returns:
            tuple: Tuple containing estimated coordinates, clock bias, and norm of delta P.
        """
        dx = 100*np.ones(3)
        b = b0
        # set up the G matrix with the right dimensions. We will later replace the first 3 columns
        # note that b here is the clock bias in meters equivalent, so the actual clock bias is b/LIGHTSPEED
        G = np.ones((measured_pseudorange.size, 4))
        iterations = 0
        while np.linalg.norm(dx) > 1e-3:
            # Eq. (2):
            r = np.linalg.norm(xs - x0, axis=1)
            # Eq. (1):
            phat = r + b0
            # Eq. (3):
            deltaP = measured_pseudorange - phat
            G[:, 0:3] = -(xs - x0) / r[:, None]
            # Eq. (4):
            sol = np.linalg.inv(np.transpose(G) @ G) @ np.transpose(G) @ deltaP
            # Eq. (5):
            dx = sol[0:3]
            db = sol[3]
            x0 = x0 + dx
            b0 = b0 + db
        norm_dp = np.linalg.norm(deltaP)
        return x0, b0, norm_dp
        
    def calculate_ECEF_to_file(self, folder_name: str) -> None:
        """
        Calculates ECEF coordinates and saves them to a CSV file.

        Args:
            folder_name (str): Name of the folder to save the CSV file.

        Returns:
            None
        """

        c = 3 * np.power(10, 8)
        for epoch in zip(self.measurements['Epoch'].unique()):
            one_epoch = self.measurements.loc[(self.measurements['Epoch'] == epoch) & (self.measurements['prSeconds'] < 0.1)] 
            one_epoch = one_epoch.drop_duplicates(subset='SvName').set_index('SvName')
            
            if len(one_epoch.index) > 4:
                timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
                sats = one_epoch.index.unique().tolist()
                ephemeris = self.manager.get_ephemeris(timestamp, sats)
                sv_position = self.__calculate_satellite_position(ephemeris, one_epoch['tTxSeconds'])

                xs = sv_position[['x_k', 'y_k', 'z_k']].to_numpy()
                pr = one_epoch['PrM'] + LIGHTSPEED * sv_position['delT_sv']
                pr = pr.to_numpy()

                for sat, pos, pseudo_range, cn0, d1,d2 in zip(one_epoch.index, xs, pr, one_epoch['Cn0DbHz'],one_epoch['PseudorangeRateMetersPerSecond'],one_epoch['CarrierFrequencyHz']):
                    self.ECEF_DATA.append({
                        'GPS time': timestamp,
                        'SatPRN (ID)': sat,
                        'Sat.X': pos[0],
                        'Sat.Y': pos[1],
                        'Sat.Z': pos[2],
                        'Pseudo-Range': pseudo_range,
                        'CN0': cn0,
                        'Doppler':(np.float64(d1) / c) * np.float64(d2)})
                    
        self.ECEF_DATA=DataFrame.from_records(self.ECEF_DATA)
        output_directory = os.path.join(outputs_directory,folder_name)
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        self.ECEF_DATA.to_csv(os.path.join(output_directory,'gnss_parsed_output.csv'))
   
    def calculate_LLA(self, csv_parsed: str) -> None:
        """
        Calculates LLA coordinates.

        Args:
            csv_parsed (str): Path to the parsed CSV file.

        Returns:
            None
        """
        df = pd.read_csv(os.path.join(csv_parsed, 'gnss_parsed_output.csv'))
        b = 0
        x = np.array([0, 0, 0])
        grouped = df.groupby('GPS time')

        for name, group in grouped:
            sat_coords = group[['Sat.X', 'Sat.Y', 'Sat.Z']].values.tolist()
            xs = np.array(sat_coords)

            pseudo_ranges = group['Pseudo-Range'].tolist()
            pr = np.array(pseudo_ranges)

            x, b, dp = self.__least_squares(xs, pr, x, b)

            self.ecef_list.append(x)

        ecef_array = np.stack(self.ecef_list, axis=0)
        lla_array = np.stack(navpy.ecef2lla(ecef_array), axis=1)

        # Create a DataFrame for latitude, longitude, and altitude
        self.lla_df=lla_df = DataFrame(lla_array, columns=['Latitude', 'Longitude', 'Altitude'])

        # Repeat the satellite coordinates DataFrame to match the length of the LLA DataFrame
        repeated_sat_coords = df[['Sat.X', 'Sat.Y', 'Sat.Z']].iloc[:len(lla_df)].reset_index(drop=True)

        # Convert lla_array and repeated_sat_coords to numpy arrays
        lla_array = np.array(lla_df)
        sat_coords_array = np.array(repeated_sat_coords)

        # Create a combined array
        combined_array = np.hstack((lla_array, sat_coords_array))

        # Convert combined array to DataFrame
        result_df = DataFrame(combined_array, columns=['Latitude', 'Longitude', 'Altitude', 'Sat.X', 'Sat.Y', 'Sat.Z'])

        # Save the result to CSV
        result_df.to_csv(os.path.join(csv_parsed, 'csv_parsed_output_LLA.csv'), index=False)
        
    def create_kml(self, folder: str) -> None:
        """
        Creates a KML file.

        Args:
            folder (str): Name of the folder to save the KML file.

        Returns:
            None
        """
        kml = simplekml.Kml()

        for i, row in self.lla_df.iterrows():
            kml.newpoint(name=f"Point {i+1}", coords=[(row['Longitude'], row['Latitude'], row['Altitude'])])

        kml.save(os.path.join(folder,'computed_path.kml'))
                   

