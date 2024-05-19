import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import simplekml
from GnssTool import  EphemerisManager

import navpy
import os
# Constants
WEEKSEC = 604800
LIGHTSPEED = 2.99792458e8
mu = 3.986005e14
OmegaDot_e = 7.2921151467e-5
F = -4.442807633e-10


parent_directory = os.getcwd()
inputs_directory = os.path.join(parent_directory,'data','inputs')
outputs_directory= os.path.join(parent_directory,'data','outputs')
ephemeris_data_directory = os.path.join(parent_directory, 'data', 'outputs', 'ephemeris')

class gnssTool:
    
    def __init__(self,input_filepath:str,ephemeris_data_directory:str) -> None:
        self.manager = EphemerisManager(ephemeris_data_directory)
        self.measurements = self.__gnss_to_dataFrame(input_filepath)
        self.ECEF_DATA=[]
        self.ecef_list = []
        self.lla_list = []
        self.lla_df=None
        
    def __gnss_to_dataFrame(self,input_filepath:str)-> pd.DataFrame:
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

        android_fixes = pd.DataFrame(android_fixes[1:], columns=android_fixes[0])
        measurements = pd.DataFrame(measurements[1:], columns=measurements[0])

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

    def __calculate_satellite_position(self,ephemeris, transmit_time):
        mu = 3.986005e14
        OmegaDot_e = 7.2921151467e-5
        F = -4.442807633e-10
        sv_position = pd.DataFrame()
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

    def __least_squares(self,xs, measured_pseudorange, x0, b0):
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
        
    def calculate_ECEF_to_file(self,folder_name:str):

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
                    
        self.ECEF_DATA=pd.DataFrame.from_records(self.ECEF_DATA)
        output_directory = os.path.join(outputs_directory,folder_name)
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        self.ECEF_DATA.to_csv(os.path.join(output_directory,'gnss_parsed_output.csv'))

    def calculate_LLA(self, csv_parsed):
        df = pd.read_csv(os.path.join(csv_parsed,'gnss_parsed_output.csv'))
        b = 0
        x = np.array([0, 0, 0])
        grouped = df.groupby('GPS time')


        for name, group in grouped:
            # Collect the 'Sat.X', 'Sat.Y', 'Sat.Z' columns into a list of lists
            sat_coords = group[['Sat.X', 'Sat.Y', 'Sat.Z']].values.tolist()
            xs=np.array(sat_coords)
              
            # Collect the 'Pseudo-Range' column into a list
            pseudo_ranges = group['Pseudo-Range'].tolist()
            pr=np.array(pseudo_ranges)

            x, b, dp = self.__least_squares(xs, pr, x, b)

            self.ecef_list.append(x)
            
        ecef_array = np.stack(self.ecef_list, axis=0)
        lla_array = np.stack(navpy.ecef2lla(ecef_array), axis=1)

        # Extract the first position as a reference for the NED transformation
        ref_lla = lla_array[0, :]
        ned_array = navpy.ecef2ned(ecef_array, ref_lla[0], ref_lla[1], ref_lla[2])

        # Convert back to Pandas and save to csv
        self.lla_df = pd.DataFrame(lla_array, columns=['Latitude', 'Longitude', 'Altitude'])
        ned_df = pd.DataFrame(ned_array, columns=['N', 'E', 'D'])
        self.lla_df.to_csv(os.path.join(csv_parsed,'csv_parsed_output_LLA.csv'), index=False)
    
    def create_kml(self,folder):
        # # Create KML file
        kml = simplekml.Kml()

        for i, row in self.lla_df.iterrows():
            kml.newpoint(name=f"Point {i+1}", coords=[(row['Longitude'], row['Latitude'], row['Altitude'])])

        kml.save(os.path.join(folder,'computed_path.kml'))
                   


# Perform coordinate transformations using the Navpy library






# # Add computed positions to the output_data
# output_data['Sat.X'] = ecef_array[:, 0]
# output_data['Sat.Y'] = ecef_array[:, 1]
# output_data['Sat.Z'] = ecef_array[:, 2]
# output_data['Lat'] = lla_array[:, 0]
# output_data['Lon'] = lla_array[:, 1]
# output_data['Alt'] = lla_array[:, 2]

# # Output to CSV
# output_data.to_csv(kml_parsed_output_filepath, index=False)
# print(f"Final output saved to {kml_parsed_output_filepath}")

