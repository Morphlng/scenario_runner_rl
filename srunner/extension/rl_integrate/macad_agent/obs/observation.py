from __future__ import annotations

import math

import numpy as np
from py_trees.blackboard import Blackboard

from srunner.extension.rl_integrate.data.local_carla_api import Transform, Vector3D
from srunner.extension.rl_integrate.data.measurement import Measurement
from srunner.extension.rl_integrate.misc import flatten_dict


class Observation:
    @staticmethod
    def encode_obs(
        actor_id: str,
        py_measurements: "dict[str, Measurement]",
        action_mask: np.ndarray = None,
    ):
        """Encode sensor and measurements into obs based on state-space config.

        Args:
            actor_id (str): Actor identifier
            py_measurements (dict): measurement file of ALL actors
            action_mask (np.ndarray): action mask for the actor

        Returns:
            obs (dict): properly encoded observation data for each actor
        """
        blackboard = Blackboard()
        actor_configs = blackboard.get("rl_actor_configs")
        add_action_mask = actor_configs[actor_id].get("add_action_mask", False)

        image = py_measurements[actor_id].camera_dict
        if image is not None and len(image) > 0:
            image = image[actor_configs[actor_id]["camera_type"]][1]
            image = Observation.preprocess_image(image)

        # Structure the observation
        obs = {"camera": image}

        # Add semantic info if required
        semantic_info = Observation.get_semantic_info(
            actor_id, py_measurements, reference="self"
        )
        obs["state"] = flatten_dict(semantic_info)

        # Add action_mask if required
        if add_action_mask:
            obs["action_mask"] = action_mask

        return obs

    @staticmethod
    def decode_image(actor_id: str, img: np.ndarray):
        """Decode actor observation into original image reversing the pre_process() operation.
        Args:
            actor_id (str): Actor identifier
            img (np.ndarray): Encoded observation image of an actor

        Returns:
            image (np.ndarray): Original actor camera view
        """
        blackboard = Blackboard()
        actor_configs = blackboard.get("rl_actor_configs")

        # Reverse the processing operation
        if actor_configs[actor_id].get("use_depth_camera", False):
            img = np.tile(img.swapaxes(0, 1), 3)
        else:
            img = img.swapaxes(0, 1) * 128 + 128
        return img

    @staticmethod
    def get_semantic_info(
        target_id: str,
        py_measurements: "dict[str, Measurement]",
        reference: str = "self",
    ):
        """Get semantic information from the current frame.

        Args:
            target_id (str): Actor we focus on
            py_measurements (dict): measurement file containing all actors' information
            reference (str): reference frame for the semantic information. "global" | "self" | "world"

        Returns:
            semantic_info (dict): semantic information
        """
        blackboard = Blackboard()
        global_sensor = blackboard.get("rl_global_sensor")
        actor_configs = blackboard.get("rl_actor_configs")
        focus_actors = actor_configs[target_id]["focus_actors"]
        ignore_actors = actor_configs[target_id]["ignore_actors"]
        measurement_type = actor_configs[target_id]["measurement_type"]

        if "all" in focus_actors:
            focus_actors = set(py_measurements.keys())

        semantic_info = {"self": {}, "ego": {}, "others": {}}
        # Iterate through each actor
        for actor_id in py_measurements:
            if actor_id not in (focus_actors - ignore_actors):
                continue

            # First, get all information in world frame
            measurement = py_measurements[actor_id]

            # Location in world frame
            transform = measurement.transform
            location = transform.location
            heading = measurement.transform.rotation.yaw

            # Project bounding box to world frame
            bbox = measurement.bounding_box

            # Velocity in world frame
            velocity = measurement.velocity

            # Nearest waypoint in world frame
            waypoint_locations = [
                waypoint.transform.location
                for waypoint in measurement.planned_waypoints
            ]
            waypoints = [(wp.x, wp.y) for wp in waypoint_locations]

            if reference == "global" and global_sensor is not None:
                target_transform = Transform.from_simulator_transform(
                    global_sensor.get_transform()
                )
                axis_mapping = target_transform.get_axis_mapping()
            elif reference == "self":
                target_transform = py_measurements[target_id].transform
                axis_mapping = None
            else:
                target_transform = None

            if target_transform is not None:
                location = Observation.transform_vector(
                    location, target_transform, axis_mapping
                )
                velocity = (
                    Observation.transform_vector(
                        velocity + transform.location,
                        target_transform,
                        axis_mapping,
                    )
                    - location
                )
                heading = (
                    -heading
                    if reference == "global"
                    else heading - target_transform.rotation.yaw
                )

                # Provide planned waypoints only for the actor itself
                if actor_id == target_id:
                    waypoint_locations = [
                        Observation.transform_vector(
                            wp_loc, target_transform, axis_mapping
                        )
                        for wp_loc in waypoint_locations
                    ]
                    waypoints = [(wp.x, wp.y) for wp in waypoint_locations]
                else:
                    waypoints = [(0, 0)] * len(waypoints)

            zero_check = lambda x: 0 if abs(x) < 1e-5 else x
            info = Observation.filter_semantic_info(
                {
                    "active": 1,
                    "is_ego": 1 if actor_id in ["hero", "ego"] else 0,
                    "x": zero_check(location.x),
                    "y": zero_check(location.y),
                    "heading": math.cos(math.radians(zero_check(heading))),
                    "vx": zero_check(velocity.x),
                    "vy": zero_check(velocity.y),
                    "speed": zero_check(measurement.speed),
                    "road_offset": zero_check(measurement.road_offset),
                    # for bounding box, [:, :2] (bottom 4 points), [::2, :2] (diagonal points)
                    # "bounding_box": bb_cords[::2, :2],
                    "extent": bbox.extent.as_numpy_array(),
                    "waypoints": [
                        [zero_check(waypoint[0]), zero_check(waypoint[1])]
                        for waypoint in waypoints
                    ],
                },
                measurement_type,
            )

            # TODO: a hack to get extra info (e.g. Ammo)
            for info in measurement_type:
                if ":" in info:
                    actor_id, key = map(lambda x: x.strip(), info.split(":"))
                    if actor_id == "self":
                        info[key] = py_measurements[target_id].get(key)
                    else:
                        info[key] = py_measurements[actor_id].get(key)

            # Update dict
            if actor_id == target_id:
                semantic_info["self"] = info
            elif actor_id.lower() in [
                "ego",
                "hero",
            ]:
                semantic_info["ego"] = info
            else:
                semantic_info["others"][actor_id] = info

        return semantic_info

    @staticmethod
    def transform_points(
        points: np.ndarray,
        reference_transform: Transform,
        axis_mapping: Transform.AxisMap = None,
        inverse_transform: bool = True,
    ) -> np.ndarray:
        """Transform a set of points from one frame to another.

        Args:
            points (np.ndarray): The points to be transformed.
            reference_transform (Transform): The reference frame.
            axis_mapping (Transform.AxisMap, optional): The axis mapping between reference frame.
            inverse_transform (bool): Whether to transform into global frame or actor frame. Defaults to actor frame.

        Returns:
            np.ndarray: _description_
        """
        transformed_points = (
            reference_transform.inverse_transform_points(points)
            if inverse_transform
            else reference_transform.transform_points(points)
        )

        if axis_mapping is not None:
            return Observation.points_axis_map(transformed_points, axis_mapping)
        else:
            return transformed_points

    @staticmethod
    def points_axis_map(points: np.ndarray, axis_mapping: Transform.AxisMap = None):
        """Map the axis of a point based on given axis mapping.

        Args:
            point (np.ndarray): The point to be mapped.
            axis_mapping (Transform.AxisMap): The axis mapping between reference frame.

        Returns:
            point (np.ndarray): Mapped point. This will always be a 2D array.
        """
        if axis_mapping is not None:
            if len(points.shape) == 1:
                points = points.reshape(1, -1)

            return np.array(
                [
                    [
                        point[axis_mapping.x.index],
                        point[axis_mapping.y.index],
                        point[axis_mapping.z.index],
                    ]
                    for point in points
                ]
            )
        else:
            return points

    @staticmethod
    def transform_vector(
        vector: Vector3D,
        reference_transform: Transform,
        axis_mapping: Transform.AxisMap = None,
        inverse_transform: bool = True,
    ) -> Vector3D:
        """Transform a vector from one frame to another.

        Args:
            vector (Vector3D): The vector to be transformed.
            reference_transform (Transform): The reference frame.
            axis_mapping (Transform.AxisMap, optional): The axis mapping between reference frame.
            inverse_transform (bool): Whether to transform into global frame or actor frame. Defaults to actor frame.

        Returns:
            Vector3D: The transformed vector.
        """
        transformed_vector = (
            reference_transform.inverse_transform_location(vector)
            if inverse_transform
            else reference_transform.transform_location(vector)
        )

        if axis_mapping is not None:
            return Observation.vector_axis_map(transformed_vector, axis_mapping)
        else:
            return transformed_vector

    @staticmethod
    def vector_axis_map(vector: Vector3D, axis_mapping: Transform.AxisMap = None):
        """Map the axis of a vector based on given axis mapping.

        Args:
            vector (Vector3D): The vector to be mapped.
            axis_mapping (Transform.AxisMap): The axis mapping between reference frame.

        Returns:
            vector (Vector3D): Mapped vector.
        """
        if axis_mapping is not None:
            return Vector3D(
                getattr(vector, axis_mapping.x.axis),
                getattr(vector, axis_mapping.y.axis),
                getattr(vector, axis_mapping.z.axis),
            )
        else:
            return vector

    @staticmethod
    def filter_semantic_info(semantic_info: dict, filter_set: "set[str]"):
        """Filter out the semantic information that is not needed.

        Args:
            semantic_info (dict): semantic information containing all information
            filter_set (set): set of semantic information (keys) to be kept

        Returns:
            semantic_info (dict): filtered semantic information
        """
        if "all" in filter_set:
            return semantic_info

        new_info = {}
        for key in semantic_info.keys():
            if key in filter_set:
                new_info[key] = semantic_info[key]

        return new_info

    @staticmethod
    def preprocess_image(image: np.ndarray):
        """Process image raw data to array data.

        Args:
            image (np.ndarray): image data from Callback.

        Returns:
            np.ndarray: Image array.
        """
        image = (image.astype(np.float32) - 128) / 128
        return image
