#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Wrapper for autonomous agents required for tracking and checking of used sensors
"""

from __future__ import annotations, print_function

import operator

import carla
from py_trees.blackboard import Blackboard

from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.autoagents.sensor_interface import CallBack
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

try:
    RL_AVAILABLE = True
    from srunner.extension.rl_integrate.data.simulator import Simulator
    from srunner.extension.rl_integrate.macad_agent import PathTracker, RlAgent
except ImportError as e:
    import logging
    logging.debug("RL extension unavailable")
    RL_AVAILABLE = False


class AgentWrapper(object):

    """
    Wrapper for autonomous agents required for tracking and checking of used sensors
    """

    _agent: AutonomousAgent = None
    _sensors_list: list[carla.Sensor] = []

    def __init__(self, agent):
        """
        Set the autonomous agent
        """
        self._agent = agent

    def __call__(self):
        """
        Pass the call directly to the agent
        """
        return self._agent()

    def setup_sensors(self, vehicle: carla.Actor, debug_mode: bool = False):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        bp_library = CarlaDataProvider.get_world().get_blueprint_library()
        for sensor_spec in self._agent.sensors():
            # These are the sensors spawned on the carla world
            bp = bp_library.find(str(sensor_spec["type"]))
            if sensor_spec["type"].startswith("sensor.camera"):
                bp.set_attribute("image_size_x", str(sensor_spec["width"]))
                bp.set_attribute("image_size_y", str(sensor_spec["height"]))
                bp.set_attribute("fov", str(sensor_spec["fov"]))
                sensor_location = carla.Location(
                    x=sensor_spec["x"], y=sensor_spec["y"], z=sensor_spec["z"]
                )
                sensor_rotation = carla.Rotation(
                    pitch=sensor_spec["pitch"],
                    yaw=sensor_spec["yaw"],
                    roll=sensor_spec["roll"],
                )
            elif sensor_spec["type"].startswith("sensor.lidar"):
                bp.set_attribute("range", str(sensor_spec["range"]))
                bp.set_attribute(
                    "rotation_frequency", str(sensor_spec["rotation_frequency"])
                )
                bp.set_attribute("channels", str(sensor_spec["channels"]))
                bp.set_attribute("upper_fov", str(sensor_spec["upper_fov"]))
                bp.set_attribute("lower_fov", str(sensor_spec["lower_fov"]))
                bp.set_attribute(
                    "points_per_second", str(sensor_spec["points_per_second"])
                )
                sensor_location = carla.Location(
                    x=sensor_spec["x"], y=sensor_spec["y"], z=sensor_spec["z"]
                )
                sensor_rotation = carla.Rotation(
                    pitch=sensor_spec["pitch"],
                    roll=sensor_spec["roll"],
                    yaw=sensor_spec["yaw"],
                )
            elif sensor_spec["type"].startswith("sensor.other.gnss"):
                sensor_location = carla.Location(
                    x=sensor_spec["x"], y=sensor_spec["y"], z=sensor_spec["z"]
                )
                sensor_rotation = carla.Rotation()

            # create sensor
            sensor_transform = carla.Transform(sensor_location, sensor_rotation)
            sensor = CarlaDataProvider.get_world().spawn_actor(
                bp, sensor_transform, vehicle
            )
            # setup callback
            sensor.listen(
                CallBack(sensor_spec["id"], sensor, self._agent.sensor_interface)
            )
            self._sensors_list.append(sensor)

        if RL_AVAILABLE and isinstance(self._agent, RlAgent):
            blackboard = Blackboard()
            actor_id = self._agent.actor_id
            actor_config = self._agent.actor_config

            if actor_config.get("collision_sensor", "off") == "on":
                Simulator.register_collision_sensor(actor_id, vehicle)
            if actor_config.get("lane_sensor", "off") == "on":
                Simulator.register_lane_invasion_sensor(actor_id, vehicle)
            if actor_config.get("enable_planner", False):
                try:
                    check_path_trackers = operator.attrgetter("rl_path_trackers")
                    rl_path_trackers = check_path_trackers(blackboard)
                except AttributeError:
                    rl_path_trackers = {}
                    blackboard.set("rl_path_trackers", rl_path_trackers)

                rl_path_trackers[actor_id] = PathTracker(
                    vehicle,
                    actor_config["start_pos"],
                    actor_config["end_pos"],
                    actor_config.get("target_speed", 5.55) * 3.6,  # m/s to km/h
                    actor_config.get("opt_dict", None),
                    actor_config.get("planned_route_path", None),
                )

            # Register the actor
            try:
                check_actors = operator.attrgetter("rl_actors")
                rl_actors = check_actors(blackboard)
            except AttributeError:
                rl_actors = {}
                blackboard.set("rl_actors", rl_actors)
            rl_actors[actor_id] = vehicle.id

        # Tick once to spawn the sensors
        if CarlaDataProvider.is_sync_mode():
            CarlaDataProvider.get_world().tick()
        else:
            CarlaDataProvider.get_world().wait_for_tick()

    def cleanup(self):
        """
        Remove and destroy all sensors
        """
        self._agent.destroy()

        for i, _ in enumerate(self._sensors_list):
            if self._sensors_list[i] is not None:
                self._sensors_list[i].stop()
                self._sensors_list[i].destroy()
                self._sensors_list[i] = None
        self._sensors_list = []
