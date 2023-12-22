from __future__ import annotations

import carla
import numpy as np

from srunner.extension.rl_integrate.sensors.carla_birdeye_view import BirdViewProducer
from srunner.extension.rl_integrate.sensors.derived_sensors import (
    CollisionSensor,
    LaneInvasionSensor,
)


class SensorDataProvider:
    """
    All sensor data will be buffered in this class

    The data can be retreived in following data structure:

    {
        'camera': {
            'actor_id': {
                'sensor_id': (frame: int, processed_data: np.ndarray),
                ...
            },
            ...
        },
        'collision': {
            'actor_id': CollisionSensor,
            ...
        },
        'lane_invasion': {
            'actor_id': LaneInvasionSensor,
            ...
        },
        ...
    }
    """

    _camera_config: dict[str, dict] = {}
    _camera_data_dict: dict[str, dict[str, tuple[int, np.ndarray]]] = {}
    _collision_sensors: dict[str, CollisionSensor] = {}
    _lane_invasion_sensors: dict[str, LaneInvasionSensor] = {}
    _birdeye_sensors: dict[tuple[int, int], BirdViewProducer] = {}
    _filter_camera_ids: set[str] = set(["ManualControl"])

    @staticmethod
    def update_camera_config(actor_id: str, config: dict):
        """
        Updates the camera config

        Args:
            actor_id (str): actor id
            config (Dict): camera config
        """
        SensorDataProvider._camera_config[actor_id] = config

    @staticmethod
    def update_camera_data(actor_id: str, data: "dict[str, tuple[int, np.ndarray]]"):
        """Updates the camera data

        Args:
            actor_id (str): actor id
            data (Dict): image data from sensor_interface.get_data(). E.g.

            data = {
                "sensor_id": (frame : int, processed_data : ndarray),
                ...
            }
        """
        if data is not None:
            filter_data = {
                k: v
                for k, v in data.items()
                if k not in SensorDataProvider._filter_camera_ids
            }
            SensorDataProvider._camera_data_dict[actor_id] = filter_data

    @staticmethod
    def update_collision_sensor(actor_id: str, sensor: CollisionSensor):
        """
        Updates a collision sensor
        """
        SensorDataProvider._collision_sensors[actor_id] = sensor

    @staticmethod
    def update_birdeye_sensor(spec: "tuple[int, int]", sensor: BirdViewProducer):
        """Updates a birdeye sensor"""
        SensorDataProvider._birdeye_sensors[spec] = sensor

    @staticmethod
    def update_lane_invasion_sensor(actor_id: str, sensor: LaneInvasionSensor):
        """Updates a lane invasion sensor"""
        SensorDataProvider._lane_invasion_sensors[actor_id] = sensor

    @staticmethod
    def get_camera_data(actor_id: str) -> "dict[str, tuple[int, np.ndarray]]":
        """Returns the camera data of the actor

        Returns:
            Dict: image data from sensor_interface.get_data(). E.g.

            data = {
                "sensor_id": (frame : int, processed_data : ndarray),
                ...
            }
        """
        data = SensorDataProvider._camera_data_dict.get(actor_id, None)
        if data is not None and "bev" in data:
            data = data.copy()
            spec = (
                SensorDataProvider._camera_config[actor_id]["x_res"],
                SensorDataProvider._camera_config[actor_id]["y_res"],
            )
            data["bev"] = (
                data["bev"][0],
                SensorDataProvider.get_birdeye_data(spec, data["bev"][1]),
            )

        return data

    @staticmethod
    def get_birdeye_data(spec: "tuple[int, int]", actor: carla.Actor) -> np.ndarray:
        """Generate a birdeye data for the actor

        Args:
            spec (tuple): width, height
            actor (carla.Actor): actor

        Returns:
            ndarray: birdeye data
        """
        return BirdViewProducer.as_rgb(
            SensorDataProvider._birdeye_sensors[spec].produce(actor)
        )

    @staticmethod
    def get_collision_sensor(actor_id: str) -> CollisionSensor:
        """
        Returns:
            CollisionSensor: collision sensor of the actor
        """
        return SensorDataProvider._collision_sensors.get(actor_id, None)

    @staticmethod
    def get_lane_invasion_sensor(actor_id: str) -> LaneInvasionSensor:
        """
        Returns:
            LaneInvasionSensor: lane invasion sensor of the actor
        """
        return SensorDataProvider._lane_invasion_sensors.get(actor_id, None)

    @staticmethod
    def get_birdeye_sensor(spec: "tuple[int, int]") -> BirdViewProducer:
        """Get a birdeye sensor based on the spec (width, height)

        Args:
            spec (tuple): width, height

        Returns:
            BirdViewProducer: birdeye sensor of the actor
        """
        return SensorDataProvider._birdeye_sensors.get(spec, None)

    @staticmethod
    def get_all_data() -> "dict[str, dict]":
        """Returns all sensor data"""
        return {
            "camera": SensorDataProvider._camera_data_dict,
            "birdeye": SensorDataProvider._birdeye_sensors,
            "collision": SensorDataProvider._collision_sensors,
            "lane_invasion": SensorDataProvider._lane_invasion_sensors,
        }

    @staticmethod
    def cleanup(soft_reset: bool = False):
        """Cleanup the sensor data

        Args:
            soft_reset (bool, optional): If True, the sensors will not be destroyed. Defaults to False.
        """

        def destroy_sensor(sensor_wrapper):
            sensor: carla.Sensor = sensor_wrapper.sensor
            if sensor.is_alive:
                if sensor.is_listening:
                    sensor.stop()
                sensor.destroy()
            sensor_wrapper.sensor = None

        if soft_reset:
            list(
                map(lambda x: x.reset(), SensorDataProvider._collision_sensors.values())
            )
            list(
                map(
                    lambda x: x.reset(),
                    SensorDataProvider._lane_invasion_sensors.values(),
                )
            )
        else:
            list(map(destroy_sensor, SensorDataProvider._collision_sensors.values()))
            list(
                map(destroy_sensor, SensorDataProvider._lane_invasion_sensors.values())
            )
            SensorDataProvider._camera_config = {}
            SensorDataProvider._birdeye_sensors = {}
            SensorDataProvider._collision_sensors = {}
            SensorDataProvider._lane_invasion_sensors = {}

        SensorDataProvider._camera_data_dict = {}
