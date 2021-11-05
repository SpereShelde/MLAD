import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
from sklearn import linear_model
import matplotlib.pyplot as plt

# import tensorflow as tf
# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

# latest_ckp = tf.train.latest_checkpoint('selected_features_models\\50-0-12layers-256-64')
# print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')

# x = np.array([1000, 1100, 1200, 1300, 1400, 1500]).reshape((-1, 1))
# y = np.array([math.radians(36), math.radians(30), math.radians(20), math.radians(14), math.radians(6), math.radians(0)]).reshape(-1)
# plt.scatter(x, y)
# plt.show()
# regr = linear_model.LinearRegression()
# regr.fit(x, y)
# print(regr.coef_, regr.intercept_)
# #
# # exit(0)
#
# x = np.array([1500, 1600, 1700, 1800, 1900, 2000]).reshape((-1, 1))
# y = np.array([math.radians(0), -math.radians(4), -math.radians(9), -math.radians(13), -math.radians(18), -math.radians(20)]).reshape(-1)
# plt.scatter(x, y)
# plt.show()
# regr = linear_model.LinearRegression()
# regr.fit(x, y)
# print(regr.coef_, regr.intercept_)

# [0.00128656] -0.34574141372840095
# [-0.00072805] 2.658718227204696

# y_true = tf.cast([
#     [
#         [1., 1., 2., 2.]
#     ],
#     [
#         [1., 1., 2., 3.]
#     ],
# ], tf.float32)
# y_pred = tf.cast([
#     [-1., -1., -2., -2.],
#     [1., 1., 2., 3.],
# ], tf.float32)
#
# y_true_2d = tf.reshape(y_true, [y_true.shape[0], y_true.shape[2]])
# # print(y_true_2d.shape)
# cos_loss = tf.keras.losses.cosine_similarity(y_true_2d, y_pred, axis=1)
# print(cos_loss)
# sub = tf.subtract(y_true_2d, y_pred)
# # print(sub)
# ae_loss = tf.reduce_mean(tf.abs(tf.subtract(y_true_2d, y_pred)), axis=1)
# print(ae_loss)
# loss = ae_loss + cos_loss
# print(loss)


# a = tf.ones([1,100],tf.int32)
# reduce_m = tf.math.reduce_mean(a)
# print(reduce_m)
# features = ['/CAN/AccPedal', '/CAN/AmbientTemperature', '/CAN/ENG_Trq_DMD', '/CAN/ENG_Trq_ZWR', '/CAN/ENG_Trq_m_ex', '/CAN/EngineSpeed_CAN', '/CAN/OilTemperature1', '/CAN/Trq_Indicated', '/CAN/VehicleSpeed', '/CAN/WheelSpeed_LL', '/CAN/WheelSpeed_LR', '/CAN/WheelSpeed_RL', '/CAN/WheelSpeed_RR']
# ['/GPS/Used satellites', '/GPS/Acceleration', '/CAN/WheelSpeed_FR', '/Plugins/Pitch', '/Plugins/Accelerometer_X', '/Plugins/Slip_angle', '/Plugins/Accelerometer_Z', '/Math/Longitude_Vehicle', '/GPS/GPS fix quality', '/Plugins/Gyroscope_X', '/Plugins/Body_acceleration_X', '/CAN/AirIntakeTemperature', '/Plugins/Velocity_Y', '/Math/Longitude_IMU', '/Plugins/Magnetometer_X', '/CAN/SteerAngle1', '/CAN/Yawrate1', '/Plugins/Gyroscope_Y', '/Plugins/Body_acceleration_Y', '/Plugins/Magnetometer_Y', '/Plugins/GNSS_status', '/GPS/Velocity', '/CAN/Engine_02_BZ', '/Plugins/Gyroscope_Z', '/Plugins/Magnetometer_Z', '/Plugins/Roll', '/Plugins/Body_acceleration_Z', '/CAN/Trq_FrictionLoss', '/Plugins/Accelerometer_Y', '/Plugins/Velocity_X', '/CAN/SCS_01_CHK', '/CAN/WheelSpeed_RL', '/CAN/BrkVoltage', '/CAN/WheelSpeed_FL', '/GPS/Direction', '/GPS/Z', '/CAN/EngineTemperature', '/GPS/Distance', '/CAN/Engine_02_CHK']
