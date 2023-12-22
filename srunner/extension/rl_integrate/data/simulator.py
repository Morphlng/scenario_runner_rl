from __future__ import annotations

import json
import logging
import math
import random
from functools import lru_cache

import carla

import srunner.extension.rl_integrate.data.local_carla_api as local_carla
from srunner.extension.rl_integrate.data.sensor_data_provider import SensorDataProvider
from srunner.extension.rl_integrate.misc import get_attributes
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

logger = logging.getLogger(__name__)


class Weather:
    """Weather presets for Simulator"""

    PRESETS: dict[int, carla.WeatherParameters] = {
        0: carla.WeatherParameters.ClearNoon,
        1: carla.WeatherParameters.CloudyNoon,
        2: carla.WeatherParameters.WetNoon,
        3: carla.WeatherParameters.WetCloudyNoon,
        4: carla.WeatherParameters.MidRainyNoon,
        5: carla.WeatherParameters.HardRainNoon,
        6: carla.WeatherParameters.SoftRainNoon,
        7: carla.WeatherParameters.ClearSunset,
        8: carla.WeatherParameters.CloudySunset,
        9: carla.WeatherParameters.WetSunset,
        10: carla.WeatherParameters.WetCloudySunset,
        11: carla.WeatherParameters.MidRainSunset,
        12: carla.WeatherParameters.HardRainSunset,
        13: carla.WeatherParameters.SoftRainSunset,
    }


class Simulator:
    """Simulator class for interacting with CARLA simulator

    This class will establish a connection with CARLA simulator and provide a set of APIs for
    interacting with the simulator. It also provides a set of APIs for interacting with the
    sensors attached to the ego vehicle.

    The connection could either via carla.Client or a BridgeServer. The former is used for
    connecting to a simulator running on the same machine. The latter is used for connecting
    to a simulator running on a remote machine.

    Note:
        There are two kinds of id used in this class:
        1. actor_id: the id which is speicified by user in the config file
        2. id: the id which is assigned by CARLA simulator
        You should judge by the name and the argument type to determine which id is used.
    """

    data_provider = CarlaDataProvider
    sensor_provider = SensorDataProvider
    game_time = GameTime

    @staticmethod
    def get_world():
        """Get the world.

        Returns:
            carla.World: The world.
        """
        return Simulator.data_provider.get_world()

    @staticmethod
    def get_map():
        """Get the map.

        Returns:
            carla.Map: The map.
        """
        return Simulator.data_provider.get_map()

    @staticmethod
    def is_sync_mode():
        """Get the synchronous mode.

        Returns:
            bool: The synchronous mode.
        """
        return Simulator.data_provider.is_sync_mode()

    @staticmethod
    def get_traffic_manager(port=None):
        """Get a traffic manager.
        This function will try to find an existing TM on the given port.
        If no port is given, it will use the current port in the env.

        Returns:
            carla.TrafficManager: The traffic manager.
        """
        if port is None:
            port = Simulator.get_traffic_manager_port()

        return Simulator.data_provider._client.get_trafficmanager(port)

    @staticmethod
    def get_traffic_manager_port():
        """Get the traffic manager port.

        Returns:
            int: The traffic manager port.
        """
        return Simulator.data_provider.get_traffic_manager_port()

    @staticmethod
    def get_actor_by_id(id: int, from_world: bool = False) -> carla.Actor:
        """Get an actor by id.

        Args:
            id (int): Actor id.
            from_world (bool): If True, get the actor directly from the Simulator. Otherwise, get it from the registed dictionary.

        Returns:
            carla.Actor: The actor.
        """
        return (
            Simulator.get_world().get_actor(id)
            if from_world
            else Simulator.data_provider.get_actor_by_id(id)
        )

    @staticmethod
    def get_actor_by_rolename(rolename: str, from_world: bool = True) -> carla.Actor:
        """Get an actor by rolename. This is mainly used for Ego Vehicles.

        Args:
            rolename (str): Actor rolename. None if not found.
            from_world (bool): If True, get the actor directly from the world. Otherwise, get it from the registed dictionary.

        Returns:
            carla.Actor: Actor with the rolename specified. None if not found
        """
        actors = (
            Simulator.get_world().get_actors()
            if from_world
            else Simulator.data_provider.get_actors(actor_only=True)
        )

        for actor in actors:
            if (
                "role_name" in actor.attributes
                and actor.attributes["role_name"] == rolename
            ):
                return actor

        return None

    @staticmethod
    def get_actor_control(id: int, to_dict: bool = True):
        """Get an actor's last control.

        Args:
            id (int): Actor id.
            to_dict (bool): If True, convert the control to a dictionary.

        Returns:
            carla.VehicleControl | carla.WalkerControl | dict: The actor's control.
        """
        actor = Simulator.get_actor_by_id(id)
        control = actor.get_control()
        if to_dict:
            control = get_attributes(control)

        return control

    @staticmethod
    def get_actor_location(id: int, use_local_api: bool = False) -> "carla.Location":
        """Get an actor's location.

        Args:
            id (int): Actor id.
            use_local_api (bool): If True, return a Location object.

        Returns:
            carla.Location: The actor's location.
        """
        carla_loc = Simulator.data_provider.get_location_by_id(id)
        if carla_loc is not None and use_local_api:
            return local_carla.Location.from_simulator_location(carla_loc)
        return carla_loc

    @staticmethod
    def get_actor_velocity(id: int, use_local_api: bool = False) -> carla.Vector3D:
        """Get an actor's velocity.

        Args:
            id (int): Actor id.
            use_local_api (bool): If True, return a Vector3D object when using use_vector.

        Returns:
            float | carla.Vector3D: Vector3D object representing velocity or speed in m/s.
        """
        velocity = Simulator.data_provider.get_velocity_by_id(id, use_vector=True)

        if velocity is not None and use_local_api:
            velocity = local_carla.Vector3D.from_simulator_vector(velocity)
        return velocity

    @staticmethod
    def get_actor_acceleration(
        id: int, use_local_api: bool = False
    ) -> "carla.Vector3D":
        """Get an actor's acceleration.

        Args:
            id (int): Actor id.
            use_local_api (bool): If True, return a Vector3D object.

        Returns:
            carla.Vector3D: The actor's acceleration in m/s^2.
        """
        acceleration = Simulator.data_provider.get_acceleration_by_id(id)
        if acceleration is not None and use_local_api:
            acceleration = local_carla.Vector3D.from_simulator_vector(acceleration)
        return acceleration

    @staticmethod
    def get_actor_transform(id: int, use_local_api: bool = False) -> "carla.Transform":
        """Get an actor's transform.

        Args:
            id (int): Actor id.
            use_local_api (bool): If True, return a Transform object.

        Returns:
            carla.Transform: The actor's transform.
        """
        carla_transform = Simulator.data_provider.get_transform_by_id(id)
        if carla_transform is not None and use_local_api:
            return local_carla.Transform.from_simulator_transform(carla_transform)
        return carla_transform

    @staticmethod
    def get_actor_forward(id: int, use_local_api: bool = False) -> "carla.Vector3D":
        """Get an actor's forward vector.

        Args:
            id (int): Actor id.
            use_local_api (bool): If True, return a Vector3D object.

        Returns:
            carla.Vector3D: The actor's forward vector. (global reference, unit vector)
        """
        carla_transform = Simulator.data_provider.get_transform_by_id(id)
        forward_vector = carla_transform.get_forward_vector()

        if forward_vector is not None and use_local_api:
            forward_vector = local_carla.Vector3D.from_simulator_vector(forward_vector)
        return forward_vector

    @staticmethod
    @lru_cache
    def get_actor_bounding_box(
        id: int, use_local_api: bool = False
    ) -> "carla.BoundingBox":
        """Get an actor's bounding box.

        Args:
            id (int): Actor id.
            use_local_api (bool): If True, return a BoundingBox object.

        Returns:
            carla.BoundingBox | BoundingBox: The actor's bounding box.
        """
        actor = Simulator.get_actor_by_id(id)
        carla_bb = getattr(actor, "bounding_box", None)

        if carla_bb is not None and use_local_api:
            carla_bb = local_carla.BoundingBox.from_simulator_bounding_box(carla_bb)
        return carla_bb

    @staticmethod
    def get_actor_fire_angle(id: int) -> local_carla.FireAngle:
        """Get an actor's current fire angle (yaw, pitch)

        Args:
            id (int): Actor id

        Returns:
            fire_angle (FireAngle): Current fire angle of the weapon this actor is holding

        Note:
            This is a rpc call, it may be slow.
        """
        actor = Simulator.get_actor_by_id(id)
        if hasattr(actor, "get_fire_angle"):
            fire_angle = actor.get_fire_angle()
            return local_carla.FireAngle.from_simulator_fire_angle(fire_angle)
        else:
            return local_carla.FireAngle()

    @staticmethod
    @lru_cache
    def get_actor_fire_angle_range(id: int) -> local_carla.FireAngleRange:
        """Get an actor's fire angle range (yaw_min, yaw_max, pitch_min, pitch_max)

        Args:
            id (int): Actor id

        Returns:
            fire_angle_range (FireAngleRange): The fire range of the weapon this actor is holding

        Note:
            This is a rpc call, it may be slow. We used cache to speed up.
        """
        actor = Simulator.get_actor_by_id(id)
        if hasattr(actor, "get_fire_angle_range"):
            fire_angle_range = actor.get_fire_angle_range()
            return local_carla.FireAngleRange.from_simulator_fire_angle_range(
                fire_angle_range
            )
        else:
            return local_carla.FireAngleRange()

    @staticmethod
    def get_actor_health(id: int) -> float:
        """Get an actor's health point

        Args:
            id (int): Actor id.

        Returns:
            float: current health point
        """
        actor = Simulator.get_actor_by_id(id)
        return getattr(actor, "health_point", 100.0)

    @staticmethod
    def get_actor_ammo(id: int) -> int:
        """Get an actor's ammo left

        Args:
            id (int): Actor id

        Returns:
            int: ammo left
        """
        actor = Simulator.get_actor_by_id(id)
        return getattr(actor, "ammo_left", 0)

    @staticmethod
    def get_actor_waypoint(id: int, use_local_api: bool = False) -> "carla.Waypoint":
        """Get an actor's waypoint, projected on the road.

        Args:
            id (int): Actor id.
            use_local_api (bool): If True, return a Waypoint object.

        Returns:
            carla.Waypoint: The actor's waypoint.
        """
        actor = Simulator.get_actor_by_id(id)
        lane_type = (
            carla.LaneType.Driving
            if isinstance(actor, carla.Vehicle)
            else carla.LaneType.Any
        )
        wpt = Simulator.data_provider.get_map().get_waypoint(
            actor.get_transform().location, project_to_road=True, lane_type=lane_type
        )

        if wpt is not None and use_local_api:
            wpt = local_carla.Waypoint.from_simulator_waypoint(wpt)
        return wpt

    @staticmethod
    def get_actor_camera_data(actor_id: str):
        """Get an actor's camera data.

        Args:
            actor_id (str): Actor id.

        Returns:
            Dict: image data from sensor_interface.get_data(). E.g.

            data = {
                "sensor_id": (frame : int, processed_data : ndarray),
                ...
            }
        """
        return Simulator.sensor_provider.get_camera_data(actor_id)

    @staticmethod
    def get_actor_collision_sensor(actor_id: str):
        """Get an actor's collision sensor.

        Args:
            actor_id (str): Actor id.

        Returns:
            CollisionSensor: The collision sensor.
        """
        coll_sensor = Simulator.sensor_provider.get_collision_sensor(actor_id)
        return coll_sensor

    @staticmethod
    def get_actor_lane_invasion_sensor(actor_id: str):
        """Get an actor's lane invasion sensor.

        Args:
            actor_id (str): Actor id.

        Returns:
            LaneInvasionSensor: The lane invasion sensor.
        """
        lane_sensor = Simulator.sensor_provider.get_lane_invasion_sensor(actor_id)
        return lane_sensor

    @staticmethod
    def set_weather(
        index: "int | list", extra_spec: dict = {}
    ) -> local_carla.WeatherParameters:
        """Set the weather.

        Args:
            index (int | list): The index of the weather.
            extra_spec (dict): Extra weather specs.

        Returns:
            weather specs (WeatherParamerters)
        """
        if isinstance(index, (list, tuple)):
            index = random.choice(index)

        if index == -1:
            weather = Simulator.get_world().get_weather()
        else:
            try:
                weather = Weather.PRESETS[index]
            except KeyError as e:
                logger.warning("Weather preset %s not found, using default 0", e)
                weather = Weather.PRESETS[0]

            for key in extra_spec:
                if hasattr(weather, key):
                    setattr(weather, key, extra_spec[key])

            Simulator.get_world().set_weather(weather)

        return local_carla.WeatherParameters.from_simulator_weather_parameters(weather)

    @staticmethod
    def set_actor_speed(id: int, speed: float, ret_command: bool = False):
        """Set the target speed of an actor.

        This function can be used to set a initial speed for an actor. Note that, this speed will be applied
        before the physics simulation starts. Therefore, the actor final speed may be different from the target speed.

        Args:
            id (int): Actor id.
            speed (float): The target speed in m/s.
            ret_command (bool): If True, return the command instead of applying it.
        """
        actor = Simulator.get_actor_by_id(id)
        if speed == 0:
            target_speed = carla.Vector3D(0, 0, 0)
        else:
            yaw = actor.get_transform().rotation.yaw * (math.pi / 180)
            vx = speed * math.cos(yaw)
            vy = speed * math.sin(yaw)
            target_speed = carla.Vector3D(vx, vy, 0)

        if not ret_command and hasattr(actor, "set_target_velocity"):
            actor.set_target_velocity(target_speed)
            return None
        else:
            return carla.command.ApplyTargetVelocity(id, target_speed)

    @staticmethod
    def set_actor_angular_speed(id: int, speed: float, ret_command: bool = False):
        """Set the target angular speed of an actor.

        Args:
            id (int): Actor id.
            speed (float): The target speed in m/s.
            ret_command (bool): If True, return the command instead of applying it.
        """
        actor = Simulator.get_actor_by_id(id)
        if speed == 0:
            target_speed = carla.Vector3D(0, 0, 0)
        else:
            yaw = actor.get_transform().rotation.yaw * (math.pi / 180)
            vx = speed * math.cos(yaw)
            vy = speed * math.sin(yaw)
            target_speed = carla.Vector3D(vx, vy, 0)

        if not ret_command and hasattr(actor, "set_target_angular_velocity"):
            actor.set_target_angular_velocity(target_speed)
            return None
        else:
            return carla.command.ApplyTargetAngularVelocity(id, target_speed)

    @staticmethod
    def request_new_actor(
        model: str,
        spawn_point: carla.Transform,
        attach_to: carla.Actor = None,
        rolename: str = "scenario",
        autopilot: bool = False,
        random_location: bool = False,
        color: carla.Color = None,
        actor_category: str = "car",
        safe_blueprint: bool = False,
        blueprint: carla.ActorBlueprint = None,
        immortal: bool = True,
        tick: bool = True,
        actor_attributes: dict = {},
        weapon_attributes: dict = {},
    ) -> carla.Actor:
        """Request a new actor.

        Args:
            model (str): The model name.
            spawn_point (carla.Transform): The spawn point.
            attach_to (carla.Actor): The actor to attach to. (Sensor only)
            rolename (str): The actor's rolename.
            autopilot (bool): Whether to enable autopilot.
            random_location (bool): Whether to spawn the actor at a random location.
            color (carla.Color): The actor's color.
            actor_category (str): The actor's category.
            safe_blueprint (bool): Whether to use the safe blueprint.
            blueprint (carla.ActorBlueprint): The blueprint to use.
            immortal (bool): Whether to make the actor immortal. (Walker only)
            tick (bool): Whether to tick the world after creation.

        Returns:
            carla.Actor: The actor.
        """
        actor = Simulator.data_provider.request_new_actor(
            model,
            spawn_point,
            rolename,
            autopilot,
            random_location,
            color,
            actor_category,
            safe_blueprint,
            blueprint,
            attach_to,
            immortal,
            tick,
        )
        Simulator.set_actor_attributes(
            actor.id, simulatePhysics=True, mobility="movable", **actor_attributes
        )
        Simulator.set_weapon_attributes(actor.id, 0, **weapon_attributes)
        return actor

    @staticmethod
    def register_actor(actor: carla.Actor, add_to_pool: bool = False):
        """Register an actor.

        Args:
            actor (carla.Actor): The actor.
            add_to_pool (bool): If True, add the actor to the actor pool. By doing this, we will handle the actor's destruction.
        """
        if add_to_pool:
            Simulator.data_provider._carla_actor_pool[actor.id] = actor
        Simulator.data_provider.register_actor(actor)

    @staticmethod
    def register_collision_sensor(actor_id: str, actor: carla.Actor):
        """Register a collision sensor.

        Args:
            actor_id (str): The actor id.
            actor (carla.Actor): The actor which the sensor is attached to.
        """
        from srunner.extension.rl_integrate.sensors.derived_sensors import (
            CollisionSensor,
        )

        Simulator.sensor_provider.update_collision_sensor(
            actor_id, CollisionSensor(actor)
        )

    @staticmethod
    def register_lane_invasion_sensor(actor_id: str, actor: carla.Actor):
        """Register a lane invasion sensor.

        Args:
            actor_id (str): The actor id.
            actor (carla.Actor): The actor which the sensor is attached to.
        """
        from srunner.extension.rl_integrate.sensors.derived_sensors import (
            LaneInvasionSensor,
        )

        Simulator.sensor_provider.update_lane_invasion_sensor(
            actor_id, LaneInvasionSensor(actor)
        )

    @staticmethod
    def register_birdeye_sensor(spec: "tuple[int, int]"):
        """Register a birdeye sensor.

        Args:
            actor_id (str): The actor id.
            spec (Tuple[int, int]): The sensor spec (width, height).
        """
        from srunner.extension.rl_integrate.sensors.carla_birdeye_view import (
            BirdViewCropType,
            BirdViewProducer,
            PixelDimensions,
        )

        if Simulator.sensor_provider.get_birdeye_sensor(spec) is not None:
            return

        producer = BirdViewProducer(
            target_size=PixelDimensions(width=spec[0], height=spec[1]),
            render_lanes_on_junctions=False,
            pixels_per_meter=4,
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
        )
        Simulator.sensor_provider.update_birdeye_sensor(spec, producer)

    @staticmethod
    def set_actor_transform(
        id: int, transform: carla.Transform, ret_command: bool = False
    ):
        """Set an actor's transform.

        Args:
            id (int): Actor id.
            transform (carla.Transform): The transform to set.
            ret_command (bool): If True, return the command instead of applying it.
        """
        if not ret_command:
            actor = Simulator.get_actor_by_id(id)
            actor.set_transform(transform)
            return None
        else:
            return carla.command.ApplyTransform(id, transform)

    @staticmethod
    def apply_actor_control(id: int, control=None, ret_command: bool = False):
        """Apply control to an actor.

        Args:
            id (int): Actor id.
            control (carla.VehicleControl | str): The control to apply.
            ret_command (bool): If True, return the command instead of applying it.
        """
        actor = Simulator.get_actor_by_id(id)
        if not actor or not actor.is_active or not hasattr(actor, "apply_control"):
            return None

        if control is None:
            control = actor.get_control()
        elif control == "reset":
            control = type(actor.get_control())()
        elif control == "stop":
            if isinstance(actor, carla.Vehicle):
                control = carla.VehicleControl(
                    throttle=0.0, steer=0.0, brake=1.0, hand_brake=True
                )
            elif isinstance(actor, carla.Walker):
                control = carla.WalkerControl(
                    direction=carla.Vector3D(0, 0, 0), speed=0.0
                )

        if ret_command:
            if isinstance(actor, carla.Vehicle):
                return carla.command.ApplyVehicleControl(id, control)
            elif isinstance(actor, carla.Walker):
                return carla.command.ApplyWalkerControl(id, control)
        else:
            actor.apply_control(control)
            return None

    @staticmethod
    def toggle_actor_physic(id: int, physic_on: bool, ret_command: bool = False):
        """Toggle an actor's physic.

        Args:
            id (int): Actor id.
            physic (bool): Whether to enable physic.
            ret_command (bool): If True, return the command instead of applying it.
        """
        if not ret_command:
            actor = Simulator.get_actor_by_id(id)
            actor.set_simulate_physics(physic_on)
            return None
        else:
            return carla.command.SetSimulatePhysics(id, physic_on)

    @staticmethod
    def toggle_actor_autopilot(id: int, autopilot: bool):
        """Toggle an actor's autopilot.

        Args:
            id (int): Actor id.
            autopilot (bool): Whether to enable autopilot.
        """
        actor = Simulator.get_actor_by_id(id)
        if hasattr(actor, "set_autopilot"):
            actor.set_autopilot(autopilot, Simulator.get_traffic_manager_port())
        else:
            logger.warning("Trying to toggle autopilot on a non-vehicle actor")

    @staticmethod
    def set_actor_attributes(id: int, **attributes):
        """This is a custom UE function to set an actor's attributes.

        Args:
            id (int): Actor id.
            **attributes: The attributes to reset.

        Example:
            >>> set_actor_attributes(114, HealthPoint=100.0)
        """
        if len(attributes) == 0:
            return

        json_str = json.dumps(attributes)
        try:
            actor = Simulator.get_actor_by_id(id)
            actor.set_attributes(json_str)
        except AttributeError:
            pass
        except Exception as e:
            logger.exception(e)
            logger.warning("Failed to set actor %d's attributes: %s", id, json_str)

    @staticmethod
    def set_weapon_attributes(id: int, weapon_index: int = 0, **attributes):
        """This is a custom UE function to set a weapon's attributes.

        Args:
            id (int): Actor id
            weapon_index (int, optional): The weapon index to specify which weapon. Defaults to 0.
            **attributes: The attributes to reset.

        Example:
            >>> set_weapon_attributes(114, HealthPoint=100.0)
        """
        if len(attributes) == 0:
            return

        json_str = json.dumps(attributes)
        try:
            actor = Simulator.get_actor_by_id(id)
            actor.set_weapon_attributes(weapon_index, json_str)
        except AttributeError:
            pass
        except Exception as e:
            logger.exception(e)
            logger.warning(
                "Failed to set actor %d weapon(%d)'s attributes: %s",
                id,
                weapon_index,
                json_str,
            )

    @staticmethod
    def toggle_world_settings(
        synchronous_mode=False,
        fixed_delta_seconds=0.05,
        no_rendering_mode=False,
        substepping=True,
        max_substep_delta_time=0.01,
        max_substeps=10,
        max_culling_distance=0.0,
        deterministic_ragdolls=True,
        **kwargs,
    ):
        """Toggle world settings

        Args:
            Please refer to: https://carla.readthedocs.io/en/0.9.13/python_api/#carla.WorldSettings
        """
        Simulator.toggle_sync_mode(synchronous_mode, fixed_delta_seconds)

        world = Simulator.get_world()
        settings = world.get_settings()
        settings.no_rendering_mode = no_rendering_mode
        settings.substepping = substepping
        settings.max_substep_delta_time = max_substep_delta_time
        settings.max_substeps = max_substeps
        settings.max_culling_distance = max_culling_distance
        settings.deterministic_ragdolls = deterministic_ragdolls
        if kwargs:
            for key, value in kwargs.items():
                setattr(settings, key, value)
        world.apply_settings(settings)

    def tick():
        """Tick the simulator.

        Returns:
            int: The current frame number.
        """
        world = Simulator.get_world()
        if Simulator.data_provider.is_sync_mode():
            frame = world.tick()
        else:
            world.wait_for_tick()

        snapshot = world.get_snapshot()
        timestamp = snapshot.timestamp
        frame = timestamp.frame

        Simulator.data_provider.on_carla_tick()
        Simulator.game_time.on_carla_tick(timestamp)
        return frame

    @staticmethod
    def teleport_actor(
        id: int, transform: (carla.Transform | tuple), ret_command: bool = False
    ):
        """Teleport an actor to given transform, guaranteeing that the actor will hold still after teleporting.

        Args:
            id (int): Actor id.
            transform (carla.Transform | tuple): The transform to teleport to, either a carla.Transform object or a tuple of (x, y, z).
            ret_command (bool): If True, return the command instead of applying it.
        """
        if not isinstance(transform, carla.Transform):
            transform = Simulator.generate_spawn_point(Simulator.get_map(), transform)

        batch = []
        batch.append(Simulator.toggle_actor_physic(id, False, ret_command))
        batch.append(Simulator.apply_actor_control(id, "stop", ret_command))
        batch.append(Simulator.set_actor_transform(id, transform, ret_command))
        batch.append(Simulator.apply_actor_control(id, "reset", ret_command))
        batch.append(Simulator.toggle_actor_physic(id, True, ret_command))
        return batch

    @staticmethod
    def send_batch(batch_of_command: list, tick: int = 0):
        """Send a batch of commands to the simulator.

        This will ensure that all command are executed in the same tick.

        Args:
            batch_of_command (list): A list of carla.command objects.
            tick (int): Tick how many times after sending the batch. Default to 0.

        Returns:
            list[command.Response]: A list of carla.command.Response objects.
        """
        batch = list(filter(lambda x: x is not None, batch_of_command))
        if len(batch) > 0:
            res = Simulator._client.apply_batch_sync(batch)
        else:
            res = []

        while tick:
            # We didn't use the `due_tick_cue` directly in `apply_batch_sync`
            # Because it only applies to synchronous mode
            Simulator.tick()
            tick -= 1

        return res

    @staticmethod
    def toggle_sync_mode(is_sync: bool, fixed_delta_seconds: float = 0.05):
        """Toggle the simulator sync mode.

        Args:
            is_sync (bool): Whether to enable sync mode.
            fixed_delta_seconds (float): The fixed delta seconds to use in sync mode.
        """
        world = Simulator.get_world()
        traffic_manager = Simulator.get_traffic_manager()

        world_settings = world.get_settings()
        world_settings.synchronous_mode = is_sync
        world_settings.fixed_delta_seconds = fixed_delta_seconds
        world.apply_settings(world_settings)
        traffic_manager.set_synchronous_mode(is_sync)

        Simulator.data_provider._sync_flag = is_sync

    @staticmethod
    def add_callback(func):
        """Add a callback to the simulator.

        Args:
            func (callable): A function to be called on every tick.

        Returns:
            id (int) : The id of the callback.

        Example:
            >>> simulator.add_callback(lambda snapshot: print(snapshot.timestamp))
        """
        return Simulator.get_world().on_tick(func)

    @staticmethod
    def remove_callback(callback_id: int):
        """Remove a callback from the simulator.

        Args:
            callback_id (int): The id of the callback.
        """
        Simulator.get_world().remove_on_tick(callback_id)

    @staticmethod
    def clean_ego_attachment(ego_id: int, ignores: "list[int]" = None):
        """Clean up the ego vehicle's attachment.

        Args:
            ego_id (int): The id of the ego vehicle.
            ignores (list[int]): Attachment that is meant to be kept.

        Returns:
            list[int]: A list of the id of the removed actors.
        """
        colli_sensor = Simulator.sensor_provider.get_collision_sensor("ego")
        colli_id = -1
        if colli_sensor is not None:
            colli_id = colli_sensor.sensor.id

        lane_sensor = Simulator.sensor_provider.get_lane_invasion_sensor("ego")
        lane_id = -1
        if lane_sensor is not None:
            lane_id = lane_sensor.sensor.id

        if ignores is None:
            ignores = [colli_id, lane_id]
        else:
            ignores = ignores + [colli_id, lane_id]

        return Simulator.data_provider.remove_actor_attachment(ego_id, ignores)

    @staticmethod
    def cleanup(soft_reset: bool = False, completely: bool = True):
        """Clean up the simulator.

        Args:
            soft_reset (bool): Whether to destroy actors or not.
        """
        Simulator.sensor_provider.cleanup(soft_reset)
        Simulator.data_provider.cleanup(soft_reset, completely)
        Simulator.game_time.restart()

    @staticmethod
    @lru_cache
    def generate_spawn_point(
        map: carla.Map,
        pos: "tuple[float, float, float]",
        rot: "tuple[float, float, float]" = None,
        project_to_road: bool = True,
        min_z: float = 0.5,
    ):
        """Generate a spawn point.

        Args:
            map (carla.Map): The map object.
            pos (list|tuple): The position of the spawn point in (x, y, z, yaw=0)
            rot (list|tuple): The rotation of the spawn point in (pitch, yaw, roll)
            project_to_road (bool): Whether to project the spawn point to the road.
            min_z (float): The minimum z value.

        Returns:
            spawn_point (carla.Transform): The spawn point.
        """
        location = carla.Location(pos[0], pos[1], max(pos[2], min_z))
        rotation = carla.Rotation()

        if rot is not None:
            rotation = carla.Rotation(*rot)
        elif project_to_road:
            wpt = map.get_waypoint(location, project_to_road=True)
            if wpt is not None:
                rotation = wpt.transform.rotation

        if len(pos) > 3:
            rotation.yaw = pos[3]

        return carla.Transform(location, rotation)

    @staticmethod
    def get_lane_end(
        map: carla.Map, location: "tuple[float, float, float] | carla.Location"
    ):
        """Return the lane end waypoint from given location.

        Args:
            map (carla.Map): The map object.
            location (carla.Location | list[float, float, float] | tuple[float, float, float]): The location.

        Returns:
            carla.Waypoint: The lane end waypoint.
        """
        if isinstance(location, (list, tuple)):
            location = carla.Location(*location)

        current_waypoint = map.get_waypoint(location, project_to_road=True)
        lane_end = current_waypoint.next_until_lane_end(1.0)[-1]

        return lane_end
