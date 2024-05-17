# GNSS Satellite Position Calculation

This project provides a method to calculate the position of a GNSS satellite in ECEF (Earth-Centered, Earth-Fixed) coordinates using raw GNSS measurements and ephemeris data.

## Overview

Accurately calculating the position of a GNSS satellite involves processing raw GNSS data and using ephemeris data to derive the satellite's position in its orbit. This position is then transformed into ECEF coordinates.

## Raw GNSS Data Fields

- `utcTimeMillis`: UTC time in milliseconds.
- `TimeNanos`: Time in nanoseconds.
- `LeapSecond`: Leap seconds.
- `TimeUncertaintyNanos`: Uncertainty in time measurement.
- `FullBiasNanos`: Full bias in nanoseconds.
- `BiasNanos`: Bias in nanoseconds.
- `BiasUncertaintyNanos`: Uncertainty in bias measurement.
- `DriftNanosPerSecond`: Drift in nanoseconds per second.
- `DriftUncertaintyNanosPerSecond`: Uncertainty in drift measurement.
- `HardwareClockDiscontinuityCount`: Clock discontinuity count.
- `Svid`: Satellite ID.
- `TimeOffsetNanos`: Time offset in nanoseconds.
- `State`: State of the measurement.
- `ReceivedSvTimeNanos`: Received satellite time in nanoseconds.
- `ReceivedSvTimeUncertaintyNanos`: Uncertainty in received satellite time.
- `Cn0DbHz`: Carrier-to-noise density.
- `PseudorangeRateMetersPerSecond`: Pseudorange rate in meters per second.
- `PseudorangeRateUncertaintyMetersPerSecond`: Uncertainty in pseudorange rate.
- `AccumulatedDeltaRangeState`: Accumulated delta range state.
- `AccumulatedDeltaRangeMeters`: Accumulated delta range in meters.
- `AccumulatedDeltaRangeUncertaintyMeters`: Uncertainty in accumulated delta range.
- `CarrierFrequencyHz`: Carrier frequency in Hz.
- `CarrierCycles`: Carrier cycles.
- `CarrierPhase`: Carrier phase.
- `CarrierPhaseUncertainty`: Uncertainty in carrier phase.
- `MultipathIndicator`: Multipath indicator.
- `SnrInDb`: Signal-to-noise ratio in dB.
- `ConstellationType`: Constellation type.
- `AgcDb`: Automatic gain control in dB.
- `BasebandCn0DbHz`: Baseband carrier-to-noise density.
- `FullInterSignalBiasNanos`: Full inter-signal bias in nanoseconds.
- `FullInterSignalBiasUncertaintyNanos`: Uncertainty in full inter-signal bias.
- `SatelliteInterSignalBiasNanos`: Satellite inter-signal bias in nanoseconds.
- `SatelliteInterSignalBiasUncertaintyNanos`: Uncertainty in satellite inter-signal bias.
- `CodeType`: Code type.
- `ChipsetElapsedRealtimeNanos`: Chipset elapsed real-time in nanoseconds.
- `Constellation`: Constellation.
- `SvName`: Satellite name.
- `GpsTimeNanos`: GPS time in nanoseconds.
- `UnixTime`: Unix time.
- `Epoch`: Epoch.
- `Pseudo-Range`: Pseudo-range.
- `tRxGnssNanos`: GNSS receive time in nanoseconds.
- `GpsWeekNumber`: GPS week number.
- `tRxSeconds`: Receive time in seconds.
- `tTxSeconds`: Transmit time in seconds.
- `prSeconds`: Pseudorange seconds.
- `PrM`: Pseudorange in meters.
- `PrSigmaM`: Pseudorange sigma in meters.

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
- `IODC`: Issue of Data Clock.
- `TransTime`: Transmission time.
- `FitIntvl`: Fit interval.
- `source`: Source.
- `LeapSeconds`: Leap seconds.

## Steps to Calculate Satellite Position

1. **Extract Necessary Data**: Extract relevant raw GNSS and ephemeris data fields.

2. **Convert Times**:
    - **GNSS System Time**: 
        ```python
        GNSS_System_Time = TimeNanos - (FullBiasNanos + BiasNanos)
        GNSS_System_Time_sec = GNSS_System_Time * 1e-9  # Convert to seconds
        ```

    - **Received Time**: 
        ```python
        Received_Time_sec = ReceivedSvTimeNanos * 1e-9  # Convert to seconds
        ```

3. **Calculate Transmission Time**:
    ```python
    Transmission_Time_sec = GNSS_System_Time_sec - Received_Time_sec
    ```

4. **Correct Transmission Time**:
    ```python
    SVclockBias = 2.1234e-5
    SVclockDrift = -1.1234e-12
    t_oc = 123456  # Example value

    Delta_t = SVclockBias + SVclockDrift * (Transmission_Time_sec - t_oc)
    Corrected_Transmission_Time = Transmission_Time_sec - Delta_t
    ```

5. **Calculate Pseudorange**:
    ```python
    c = 299792458  # Speed of light in m/s
    Pseudorange = c * Corrected_Transmission_Time
    ```

6. **Retrieve Satellite Ephemeris Data**: Use the satellite ID (`Svid`) to fetch the corresponding ephemeris data.

7. **Compute Satellite Position**:
    - **Mean Anomaly**:
        ```python
        from math import sqrt, sin, cos, tan, atan2

        mu = 3.986005e14  # Earth's gravitational constant
        a = sqrtA ** 2
        n_0 = sqrt(mu / a**3)
        t_k = Corrected_Transmission_Time - t_oe

        M_k = M_0 + (n_0 + deltaN) * t_k
        ```

    - **Eccentric Anomaly (E)**: Solve Kepler's equation iteratively.
        ```python
        E_k = M_k  # Initial guess
        for _ in range(10):  # Iterate to solve Kepler's equation
            E_k = M_k + e * sin(E_k)
        ```

    - **True Anomaly (ν)**:
        ```python
        sin_v = sqrt(1 - e**2) * sin(E_k) / (1 - e * cos(E_k))
        cos_v = (cos(E_k) - e) / (1 - e * cos(E_k))
        v_k = atan2(sin_v, cos_v)
        ```

    - **Orbital Radius (r) and Position in Orbital Plane (x', y')**:
        ```python
        r_k = a * (1 - e * cos(E_k))
        x_prime = r_k * cos(v_k)
        y_prime = r_k * sin(v_k)
        ```

    - **Corrected Orbital Plane Coordinates (x'', y'', z'')**:
        ```python
        u_k = v_k + omega
        i_k = i_0 + IDOT * t_k
        Omega_k = Omega_0 + (OmegaDot - 7.2921151467e-5) * t_k

        x_double_prime = x_prime * cos(u_k) - y_prime * sin(u_k)
        y_double_prime = x_prime * sin(u_k) + y_prime * cos(u_k)
        z_double_prime = y_prime * sin(i_k)
        ```

    - **ECEF Coordinates**:
        ```python
        x_ecef = x_double_prime * cos(Omega_k) - y_double_prime * cos(i_k) * sin(Omega_k)
        y_ecef = x_double_prime * sin(Omega_k) + y_double_prime * cos(i_k) * cos(Omega_k)
        z_ecef = y_double_prime * sin(i_k)
        ```

## Implementation

For detailed implementation, refer to the Python scripts in this repository. These scripts include functions to parse raw GNSS and ephemeris data, compute satellite positions, and convert them to ECEF coordinates.

## References

- GNSS documentation and signal structure.
- Ephemeris data format and processing.
- Satellite navigation principles.

## License

This project is licensed under the MIT License.

