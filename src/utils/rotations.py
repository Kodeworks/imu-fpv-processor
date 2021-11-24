import numpy as np


class Rotations:
    @staticmethod
    def bank_angle(self, angle_x, angle_y):
        return np.arctan(np.sqrt(np.tan(angle_x) ** 2 + np.tan(angle_y) ** 2))

    @staticmethod
    def absolute_rotation(rotation):
        """
        Given three rotation angles, calculate absolute rotation along some axis.
        :param rotation: Angles of rotation in [roll, pitch, yaw].
        :return: Absolute rotation as a single angle.
        """
        return np.sqrt(rotation[0]**2 + rotation[1]**2 + rotation[2]**2)

    @staticmethod
    def rotation_axis(rotation, abs_rot):
        """
        Calculate the rotation axis around which the sensor rotates for some interval where the axis is assumed
        stationary.
        :param rotation: The recorded rotations in roll, pitch and yaw direction. Can be either actual angles or angular
        velocity.
        :param abs_rot: The absolute rotation given the three angles.
        :returns: An [x, y, z] directed, normalized rotation axis.
        """
        return np.asarray([rotation[0], rotation[1], rotation[2]])/abs_rot

    @staticmethod
    def rotation_matrix(sensor_angles):
        """
        Produce a rotation matrix given a sensor reading of rotations.
        :param sensor_angles: A set of [roll, pitch, yaw] angles collected from sensors.
        :return: The rotation matrix needed to rotate a system of coordinates according to the given measured rotation.
        """
        abs_rot = Rotations.absolute_rotation(sensor_angles)
        # If the entire rotation is found to be zero, return the identity matrix
        if abs_rot == 0.0:
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        [vx, vy, vz] = Rotations.rotation_axis(sensor_angles, abs_rot=abs_rot)

        c = np.cos(abs_rot)
        s = np.sin(abs_rot)
        r1 = [c + vx**2*(1-c), vx*vy*(1-c)-vz*s, vx*vz*(1-c)+vy*s]
        r2 = [vx*vy*(1-c)+vz*s, c+vy**2*(1-c), vy*vz*(1-c)-vx*s]
        r3 = [vx*vz*(1-c)-vy*s, vz*vy*(1-c)+vx*s, c+vz**2*(1-c)]

        return np.array([r1,
                         r2,
                         r3])

    @staticmethod
    def rotate_system(sys_ax, sensor_angles):
        """
        Rotates a sensor in the global reference frame according to local gyro data.

        Sensor is represented as three unit vectors emerging from origo, denoting the three axes of the system:
          [x, y, z]
         =
        [[x_X, y_X, z_X],
         [x_Y, y_Y, z_Y],
         [x_Z, y_Z, z_Z]]

         Where x, y, z are 3x1 the axes of the inner system, represented by coordinates from the outer system, X, Y, Z,
         Such that x_Y is the outer Y-coordinate of the inner x-axis.

        :param sys_ax: Orientation of sensor axes in a 3x3 matrix. Columns are 3x1 position vectors.
        :param sensor_angles: Some angles rotated around an assumed-to-be-stationary axis.
        :return: A 3x3 matrix of the axes of the new local frame. Columns are 3x1 position vectors.
        """
        rot_mat = Rotations.rotation_matrix(sensor_angles=sensor_angles)
        res = np.matmul(sys_ax, rot_mat)
        return res
