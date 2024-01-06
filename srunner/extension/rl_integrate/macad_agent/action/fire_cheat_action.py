from __future__ import annotations

import math

import carla
import numpy as np

from srunner.extension.rl_integrate.data.simulator import Simulator
from srunner.extension.rl_integrate.macad_agent.action.action_interface import (
    AbstractAction,
    ActionInterface,
)


class DirectAction(AbstractAction):
    def __init__(
        self,
        action: dict,
        duration: int = 60,
        grid_size: int = 10,
        pitch_correction: float = 0,
        z_correction: float = 0,
        side_length: int = 3,
        target: carla.Actor = None,
    ):
        super().__init__(action, duration)
        self.max_duration = duration
        self.grid_size = grid_size
        self.pitch_correction = pitch_correction
        self.z_correction = z_correction
        self.side_length = side_length
        self.target = target

    def run_step(self, actor: carla.Actor):
        """Return a carla control signal based on the actor type and the action

        Args:
            actor (carla.Actor): The actor to run the action

        Returns:
            None
        """
        self.duration -= 1
        idx = self.action["idx"]

        if idx == 0 or self.target is None:
            return None

        if self.duration == self.max_duration - 1:
            actor_transform = actor.get_transform()
            target_location_world = self.target.get_location()

            # Raw input x, y
            x, y = self.get_grid_coordinates(self.side_length, idx)
            target_grid_x, target_grid_y = self.get_grid_coord_from_location(
                actor_transform, target_location_world
            )

            # Relative x, y to target
            tx, ty = target_grid_x + x, target_grid_y + y

            # Fire location
            target_location_world = self.get_grid_center_location(
                Simulator.get_world(),
                actor_transform,
                grid_coord=(tx, ty),
                grid_size=self.grid_size,
            )
            yaw, pitch = self.fire_strike(
                actor_transform,
                target_location_world,
                self.pitch_correction,
                self.z_correction,
            )
            actor.set_fire_angle(yaw, pitch)
        elif self.max_duration - self.duration == 20:
            actor.open_fire()

        return None

    @staticmethod
    def get_grid_coordinates(side_length: int, cell_number: int):
        # Calculate the row and column indices
        row = (cell_number - 1) // side_length
        col = (cell_number - 1) % side_length

        # Calculate the x and y coordinates
        x = col - (side_length // 2)
        y = (side_length // 2) - row

        return x, y

    @staticmethod
    def get_grid_coord_from_location(
        cannon_transform: carla.Transform,
        target_location: carla.Vector3D,
        grid_size: int = 10,
    ):
        cannon_location = cannon_transform.location
        forward_vector = cannon_transform.get_forward_vector()
        right_vector = cannon_transform.get_right_vector()

        # Calculate the vector from the cannon location to the target location
        target_vector = target_location - cannon_location

        # Use dot product and cross product to find the projections relative to the cannon's forward and right vectors
        forward_projection = target_vector.dot(forward_vector)
        right_projection = target_vector.dot(right_vector)

        # Map the projections to grid coordinates based on grid size
        grid_coord_x = round(right_projection / grid_size)
        grid_coord_y = round(forward_projection / grid_size)

        return grid_coord_x, grid_coord_y

    @staticmethod
    def get_grid_center_location(
        world: carla.World,
        cannon_transform: carla.Transform,
        grid_coord: "tuple[int, int]",
        grid_size: int = 10,
    ) -> carla.Location:
        cannon_location = cannon_transform.location
        forward_vector = cannon_transform.get_forward_vector()
        right_vector = cannon_transform.get_right_vector()

        forward_offset = forward_vector * grid_size * grid_coord[1]
        right_offset = right_vector * grid_size * grid_coord[0]
        target_location = cannon_location + forward_offset + right_offset

        proj = world.ground_projection(target_location + carla.Location(z=10), 100)
        return target_location if proj is None else proj.location

    @staticmethod
    def fire_strike(
        actor_transform: carla.Transform,
        target_location: carla.Location,
        pitch_correction: float = 0.0,
        z_correction: float = 0.0,
    ):
        """Calculate the yaw and pitch angle for the actor to fire at the target

        Args:
            actor_transform (carla.Transform): The transform of the actor
            target_location (carla.Location): The location of the target
            pitch_correction (float, optional): Correct the model's pitch angle to start from the horizontal plane. Defaults to 0.0.
            z_correction (float, optional): Correct the model's z axis to start from the muzzle. Defaults to 0.0.

        Returns:
            yaw, pitch (float, float): The yaw and pitch angle for the actor to fire at the target (in degrees)
        """
        # Adjust actor transform for z correction
        transform = carla.Transform(
            actor_transform.location + carla.Location(z=z_correction),
            actor_transform.rotation,
        )

        # Convert to actor-relative coordinates
        world_to_actor = np.array(transform.get_inverse_matrix())
        target_location = np.array(
            [target_location.x, target_location.y, target_location.z, 1.0]
        )
        relative_location = np.dot(world_to_actor, target_location)

        # Yaw and pitch calculations
        x, y, z = relative_location[:3]
        yaw = math.degrees(math.atan2(y, x))
        pitch = math.degrees(math.asin(z / math.sqrt(x**2 + y**2 + z**2)))
        return yaw, pitch + pitch_correction

    def update_action(self, action):
        if self.duration <= 0:
            self.action = action
            self.duration = self.max_duration


class FireCheatAction(ActionInterface):
    def __init__(self, action_config: dict):
        """Initialize the action converter for open fire action space

        Args:
            action_config (dict): A dictionary of action config
        """
        super().__init__(action_config)
        self.duration = action_config.get("duration", 60)
        self.grid_size = action_config.get("grid_size", 10)
        self.target_name = action_config.get("target", "Ego")
        self.pitch_correction = action_config.get("pitch_correction", 0)
        self.z_correction = action_config.get("z_correction", 0)
        self.attack_range = action_config.get("attack_range", 150)
        self.target = None

        action_space = action_config["discrete_action_set"]
        side_length = int(math.sqrt(len(action_space) - 1))

        self.direct_action = DirectAction(
            {"idx": 0},
            self.duration,
            self.grid_size,
            self.pitch_correction,
            self.z_correction,
            side_length,
            self.target,
        )

    def convert_single_action(self, action: int, done_state: bool = False):
        """Convert the action of a model output to an AbstractAction instance

        Args:
            action: Action for a single actor
            done_state (bool): Whether the actor is done. If done, return a stop action

        Returns:
            DirectAction: A direct action instance
        """
        if done_state:
            return self.stop_action(env_action=False)
        else:
            self.direct_action.update_action({"idx": action})
            return self.direct_action

    def get_action_mask(self, actor: carla.Actor, action_space: "dict"):
        """Fire action is always applicable

        Note:
            Ammo out should cause the actor to be done
        """
        self._find_target_actor()
        self.direct_action.target = self.target

        # Check if last action is done
        prev_idx = self.direct_action.action["idx"]
        if prev_idx != 0 and self.direct_action.duration > 0:
            mask = {action: 0 for action in action_space.values()}

            mask[prev_idx] = 1
            return mask

        # Check whether ego is within the fire range
        if self.target is not None:
            if not self._is_target_within_range(actor, self.target):
                mask = {action: 0 for action in action_space.values()}
                mask[0] = 1
                return mask

        return True

    def stop_action(self, env_action: bool = True, use_discrete: bool = False):
        """Return the stop action representation in low-level action space

        Args:
            env_action (bool): Whether using env action space
            use_discrete (bool): Whether using discrete action space

        Returns:
            DirectAction: if env_action is False, return the stop action in the action space of the action handler.
            EnvAction: a valid action in the env action space
        """

        if not env_action:
            return DirectAction({"idx": 0})

        return 0

    def _find_target_actor(self):
        if self.target is None or not self.target.is_active:
            self.target = Simulator.get_actor_by_rolename(self.target_name)

    def _is_target_within_range(
        self,
        cannon: carla.Actor,
        target: carla.Actor,
    ):
        target_loc = target.get_location()
        cannon_loc = cannon.get_location()
        return target_loc.distance(cannon_loc) <= self.attack_range
