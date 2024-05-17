# GNSS Satellite Position Calculation

This project provides a method to calculate the position of a GNSS satellite in ECEF (Earth-Centered, Earth-Fixed) coordinates using raw GNSS measurements and ephemeris data.

## Overview

GNSS raw data and ephemeris data are essential for calculating the position of a satellite. This project outlines the necessary steps to process these data and compute the satellite's position accurately.

## Raw GNSS Data Fields

- `TimeNanos`: Current time in nanoseconds.
- `FullBiasNanos`: Full bias in nanoseconds.
- `BiasNanos`: Bias in nanoseconds.
- `ReceivedSvTimeNanos`: Received satellite time in nanoseconds.
- `Svid`: Satellite ID.
- `TimeOffsetNanos`: Time offset in nanoseconds.

## Ephemeris Data Fields

- `sv`: Satellite ID.
- `time`: Ephemeris reference time.
- `SVclockBias`: Satellite clock bias.
- `SVclockDrift`: Satellite clock drift.
- `SVclockDriftRate`: Satellite clock drift rate.
- `IODE`: Issue of Data Ephemeris.
- `C_rs`, `C_uc`, `C_us`, `C_ic`, `C_is`, `C_rc`: Harmonic coefficients.
- `deltaN`: Mean motion difference.
- `M_0`: Mean anomaly at reference time.
- `e`: Eccentricity.
- `sqrtA`: Square root of the semi-major axis.
- `t_oe`: Time of ephemeris.
- `Omega_0`: Longitude of ascending node.
- `i_0`: Inclination angle at reference time.
- `omega`: Argument of perigee.
- `OmegaDot`: Rate of right ascension.
- `IDOT`: Rate of inclination angle.
- `GPSWeek`: GPS week number.
- `t_oc`: Clock data reference time.
- `TGD`: Group delay.

## Steps to Calculate Satellite Position

1. **Extract Necessary Data**: Extract relevant raw GNSS and ephemeris data fields.
2. **Convert Times**:
    - Compute GNSS system time.
    - Compute received time.
3. **Calculate Transmission Time**: Determine the time when the signal was transmitted from the satellite.
4. **Correct Transmission Time**: Apply satellite clock corrections using ephemeris data.
5. **Calculate Pseudorange**: Compute the pseudorange based on corrected transmission time.
6. **Retrieve Satellite Ephemeris Data**: Obtain the satellite's orbital parameters.
7. **Compute Satellite Position**:
    - Compute mean anomaly and solve Kepler's equation iteratively.
    - Calculate true anomaly and orbital radius.
    - Determine the position in the orbital plane.
    - Transform the orbital coordinates to ECEF coordinates using orbital parameters.

## Example Calculation

1. **Extract Data**:
    ```python
    TimeNanos = 1234567890000000
    FullBiasNanos = -987654321000000
    BiasNanos = 0
    ReceivedSvTimeNanos = 567890000000
    Svid = 22
    TimeOffsetNanos = 0
    ```

2. **Convert Times**:
    ```python
    GNSS_System_Time = TimeNanos - (FullBiasNanos + BiasNanos)
    GNSS_System_Time = 2222222210000000
    GNSS_System_Time_sec = 2222.22221  # in seconds

    Received_Time_sec = ReceivedSvTimeNanos * 1e-9
    ```

3. **Calculate Transmission Time**:
    ```python
    Transmission_Time_sec = GNSS_System_Time_sec - Received_Time_sec
    Transmission_Time_sec = 1654.33221
    ```

4. **Correct Transmission Time**:
    ```python
    SVclockBias = 2.1234e-5
    SVclockDrift = -1.1234e-12
    t_oc = 123456

    Delta_t = SVclockBias + SVclockDrift * (Transmission_Time_sec - t_oc)
    Corrected_Transmission_Time = Transmission_Time_sec - Delta_t
    ```

5. **Calculate Pseudorange**:
    ```python
    c = 299792458  # Speed of light in m/s
    Pseudorange = c * Corrected_Transmission_Time
    ```

6. **Compute Satellite Position**:
    - Use the ephemeris data to solve Kepler's equation for mean anomaly.
    - Compute the true anomaly and orbital radius.
    - Transform orbital coordinates to ECEF coordinates.

## Implementation

For detailed implementation, refer to the Python scripts in this repository.

## References

- GNSS documentation and signal structure.
- Ephemeris data format and processing.
- Satellite navigation principles.

## License

This project is licensed under the MIT License.
