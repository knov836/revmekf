# RevMEKF

# Testing files
The following test files are available:

- **test_variants.py** — synthetic data  
- **test_variants_file_odo_file0.py** — data using odometry, accelerometer, and pressure sensors  
- **test_variants_file_imu.py** — dataset from an autonomous vehicle  
- **test_variants_file_imu_sbg.py** — dataset from an SBG device  

Each file allows testing of three filter variants:  
- Integration of gyroscope  
- MEKF  
- Rev-MEKF  

on their respective datasets.