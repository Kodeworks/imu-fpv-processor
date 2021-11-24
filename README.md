# imu-fpv-processor 0.1.1

This repository largely revolves around the class named FloatService. FloatService serves the purpose of estimating the
height and the two angles of the x- and y-axis to the horizontal plane, of an IMU sensor.<br>


## Setup

### Windows
Download Python from [Python's active releases](https://www.python.org/downloads/). We currently run **Python 3.8**.

`python -m pip install --upgrade -U pip`<br>
`python -m pip install virtualenv`<br>
`virtualenv venv`<br>
`venv\scripts\activate`<br>
`python -m pip install -r requirements.txt`<br>

### Linux
`sudo apt remove python3.8 python3.8-pip`<br>
`rm -rf .local`<br>
`sudo apt install python3.8 python3.8-pip`<br>
`sudo python -m pip install --upgrade -U pip`<br>
`sudo python -m pip install virtualenv`<br>
`virtualenv venv`<br>
`source venv/bin/activate`<br>
`pip install -r requirements.txt`<br>

## Testing
Unit tests are run by running  

`pytest test\test_float_service.py`

## Processing and visualizing datasets
Running float_service_stats.py will use a dataset that needs to be put in the data folder beforehand. This dataset can
be downloaded from 
[this public drive](https://drive.google.com/drive/folders/1qK7dj_Xnk2Dm116jCTDztkMFMz2Flqn7?usp=sharing).
<br>

# FloatService
FloatService receives data rows of **[accelX, accelY, accelZ, gyroX, gyroY]**. The input data is placed in 
FloatService.input.<br>
FloatService estimates the pose of the sensor, and places an equal number of estimated **[angleX, angleY, height]** data
rows in FloatService.output, to the number of given input data rows.<br>
The goal of FloatService is ultimately to observe how a floating device moves while sitting in potentially rough
sea.<br>
After initialization, the FloatService.process()-call follows the steps of **preprocessing**, **pose estimation** and
**post processing**.

## Initialization
FloatService(name: str, input, output, dev_mode: bool = False) expects references to input and output, which should be
a zero initialized np.ndarray, np.matrix, np.memmap or any numpy matrix-like container.

## Preprocessing
### Sensor bias update
Based on the N newest data rows, an average value for each sensor degree of freedom is calculated and passed to an
adaptive average filter. The resulting values are later used to correct for any innate sensor offset.

### NaN-value handling
Taking into consideration that there will be the occasional data loss - for which the response is to send a
corresponding number of NaN-containing data rows - this must be handled explicitly. The current solution is to attempt
to interpolate useable input from what intact input is to be found, or simply perform a discarding of the entire burst
if it is found to contain a number of NaN-rows above some threshold.

### Data conversion
The pose estimation process uses acceleration measured in m/s^2 and angular velocity in rad/s. The sensors currently in
use produce acceleration in g and deg/s. Thus data must be converted before the pose estimation process is run.

### Outlier filtering
Currently performed by running the input through a low-pass filter (Butterfield Filter)

## Pose estimation
### Angle estimation
The two estimated angles of the sensor represent the angle between the sensor's x- and y-axis, and their respective
projection onto the horizontal plane. This choice of angle representation makes for a computationally cheap Kalman
filter, combining accelerometer and gyroscope data, since the accelerometer based angles may be calculated
independently of each other.<br>
Currently, the Kalman filter is activated only each N rows (being set to 10 rows), and the angles are otherwise
numerically integrated from the gyroscope sampled angular velocity.

### Vertical acceleration estimation
Using the two estimated angles, the sensor's acceleration vector [accX, accY, accZ] is rotated, and the z component of
the resulting acceleration vector is corrected for the gravitational constant, and assumed to be the vertical
acceleration of the sensor.

### Vertical velocity estimation
Vertical velocity is numerically integrated from the vertical acceleration. It is then corrected by a mean vertical
velocity, in case the vertical acceleration is biased in either direction. A small damping factor is also applied to 
the velocity estimate, in order to counteract acceleration bias over longer stretches of time.

### Vertical position (height) estimation
Height is numerically integrated from the vertical velocity. Average correction and a tiny damping is also applied to
the height estimate.

## Post processing
As of now, the only post processing applied to the output is a low-pass filter applied to the height estimate. This is
due to the average-correction being performed in height estimation, which tends to leave the result quite "jagged".
<br>
<br>
<br>

## Contact us ##
Any questions may be directed to [simen@kodeworks.no](mailto:simen@kodeworks.no).