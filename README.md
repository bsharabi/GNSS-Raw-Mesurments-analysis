### GNSS Data Processing

This project processes GNSS (Global Navigation Satellite System) data files, performs calculations related to satellite positions and coordinates, and generates output files. It includes modules for managing ephemeris data, parsing GNSS log files, and processing GNSS data.

### Installation Instructions

1. Clone the repository:
```bash
   git clone <https://github.com/bsharabi/GNSS-Raw-Mesurments-analysis.git>
```
2. Navigate to the project directory
```bash  
   cd <GNSS-Raw-Mesurments-analysis>
```
3. Install dependencies:
```bash  
   pip install -r requirements.txt
```
### How to run
1. Ensure that the GNSS log files are placed in the appropriate directories within the data/inputs folder. 
2. Run the controller.py script
```bash
  python controller.py
```
3. The script will process the GNSS data, generate output files, and save them in the data/outputs folder.


### License
This project is licensed under the MIT License. See the LICENSE file for details.