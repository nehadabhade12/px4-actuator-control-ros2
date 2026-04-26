"""
PX4 SITL Actuator-Level Quadrotor Controller

CONTROL STRUCTURE:
- Altitude PID → desired acceleration → thrust
- Attitude PD → roll/pitch corrections
- Mixer → individual motor commands

"""
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from px4_msgs.msg import ActuatorMotors
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleOdometry

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math


class FinalHover(Node):

    def __init__(self):
        super().__init__('final_hover')

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.act_pub = self.create_publisher(
            ActuatorMotors, '/fmu/in/actuator_motors', 10)

        self.mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 10)

        self.cmd_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', 10)

        self.sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_cb,
            sensor_qos
        )

        # state
        self.z_est = 0.0
        self.vz_est = 0.0
        self.x_est = 0.0
        self.vx_est = 0.0

        self.ekf_bad = False

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.prev_roll = 0.0
        self.prev_pitch = 0.0

        # target
        self.z_target = -0.5

        # physics
        self.hover = 0.73
        self.g = 9.81

        # pid
        self.int_err = 0.0
        self.T_prev = self.hover

        self.counter = 0
        self.dt = 0.02

        self.timer = self.create_timer(self.dt, self.loop)

    # odometry callback
    def odom_cb(self, msg):

        self.x_est = msg.position[0]
        self.vx_est = msg.velocity[0]

        z_raw = msg.position[2]
        vz_raw = msg.velocity[2]

        z_pred = self.z_est + self.vz_est * self.dt

        if abs(z_raw - z_pred) > 0.8:
            self.z_est = z_pred
            self.ekf_bad = True
            return

        self.ekf_bad = False

        alpha = 0.2
        self.z_est = (1 - alpha) * self.z_est + alpha * z_raw
        self.vz_est = (1 - alpha) * self.vz_est + alpha * vz_raw

        w, x, y, z = msg.q
        self.roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        self.pitch = math.asin(max(-1.0, min(1.0, 2*(w*y - z*x))))
        self.yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    # vehicle command
    def send_cmd(self, cmd, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = cmd
        msg.param1 = p1
        msg.param2 = p2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)

    # main control loop
    def loop(self):

        now = int(self.get_clock().now().nanoseconds / 1000)

        # offboard heartbeat
        mode = OffboardControlMode()
        mode.direct_actuator = True
        mode.timestamp = now
        self.mode_pub.publish(mode)

        self.counter += 1

        # offboard and arm
        if self.counter == 100:
            self.get_logger().info("OFFBOARD")
            self.send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

        if self.counter == 110:
            self.get_logger().info("ARM")
            self.send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

        # altitude control
        if self.counter < 300:
            T = self.hover + 0.02

        elif self.ekf_bad:
            T = self.hover

        else:
            error = self.z_est - self.z_target
            vz = self.vz_est

            Kp = 0.2
            Kd = 0.8
            Ki = 0.04

            self.int_err += error * self.dt
            self.int_err = max(-0.2, min(0.2, self.int_err))

            az_cmd = Kp * error + Kd * vz + Ki * self.int_err
            T_target = self.hover * (1 + az_cmd / self.g)

            T = 0.85 * self.T_prev + 0.15 * T_target

            max_delta = 0.015
            T = max(self.T_prev - max_delta, min(self.T_prev + max_delta, T))

            self.T_prev = T
            T = max(0.65, min(0.80, T))

        # position control
        x_curr = self.x_est
        vx_curr = self.vx_est

        x_target = 2.0 if self.counter > 600 else 0.0

        Kp_pos = 0.4
        desired_vx = (x_target - x_curr) * Kp_pos
        desired_vx = max(-0.3, min(0.3, desired_vx))

        Kp_vel = 0.12
        pitch_des = (vx_curr - desired_vx) * Kp_vel
        pitch_des = max(-0.07, min(0.07, pitch_des))

        # attitude control
        K_att = 0.30
        D_att = 0.05
        K_yaw = 0.03

        d_roll = (self.roll - self.prev_roll) / self.dt
        d_pitch = (self.pitch - self.prev_pitch) / self.dt

        corr_roll = -(K_att * self.roll + D_att * d_roll)
        corr_pitch = -(K_att * (self.pitch - pitch_des) + D_att * d_pitch)
        corr_yaw = -(K_yaw * self.yaw)

        self.prev_roll = self.roll
        self.prev_pitch = self.pitch

        # mixer
        m1 = T - corr_roll + corr_pitch + corr_yaw
        m2 = T + corr_roll - corr_pitch + corr_yaw
        m3 = T + corr_roll + corr_pitch - corr_yaw
        m4 = T - corr_roll - corr_pitch - corr_yaw

        motors = [
            float(max(0.0, min(1.0, m1))),
            float(max(0.0, min(1.0, m2))),
            float(max(0.0, min(1.0, m3))),
            float(max(0.0, min(1.0, m4)))
        ]

        msg = ActuatorMotors()
        msg.timestamp = now
        msg.control = motors + [float('nan')] * 8
        self.act_pub.publish(msg)

        if self.counter % 50 == 0:
            self.get_logger().info(
                f"x={self.x_est:.2f}, vx={self.vx_est:.2f}, z={self.z_est:.2f}"
            )


def main():
    rclpy.init()
    node = FinalHover()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()