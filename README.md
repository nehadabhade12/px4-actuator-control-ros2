# PX4 Actuator-Level Quadrotor Control (ROS 2)

This project implements low-level actuator control of a quadrotor in PX4 SITL using ROS 2 by directly publishing motor commands to `/fmu/in/actuator_motors`.

## Features
- Direct motor-level control (bypassing PX4 internal controllers)
- Altitude stabilization using PID
- Attitude stabilization (roll, pitch)
- Setpoint-based position control

## Architecture

Position → Velocity → Attitude → Motor Mixing

## Key Control

Altitude:
T = T_hover (1 + a_z / g)

a_z = Kp·e + Kd·vz + Ki·∫e

