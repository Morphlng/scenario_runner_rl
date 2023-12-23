from __future__ import annotations

import math

import carla
from shapely.geometry import Point, Polygon

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
    ):
        super().__init__(action, duration)
        self.max_duration = duration
        self.grid_size = grid_size
        self.pitch_correction = pitch_correction
        self.z_correction = z_correction

    def run_step(self, actor: carla.Actor):
        """Return a carla control signal based on the actor type and the action

        Args:
            actor (carla.Actor): The actor to run the action

        Returns:
            None
        """
        self.duration -= 1
        x, y = self.action["x"], self.action["y"]

        # (0, 0) points to self
        if x == 0 and y == 0:
            return None

        if self.duration == self.max_duration - 1:
            actor_transform = actor.get_transform()
            target_location = self.get_grid_center_location(
                Simulator.get_world(),
                actor_transform,
                grid_coord=(x, y),
                grid_size=self.grid_size,
            )
            yaw, pitch = self.fire_strike(
                actor_transform.location,
                target_location,
                self.pitch_correction,
                self.z_correction,
            )
            actor.set_fire_angle(yaw, pitch)
        elif self.max_duration - self.duration == 20:
            actor.open_fire()

        return None

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
        actor_location: carla.Location,
        target_location: carla.Location,
        pitch_correction: float = 0.0,
        z_correction: float = 0.0,
    ):
        dx = target_location.x - actor_location.x
        dy = target_location.y - actor_location.y
        dz = target_location.z - (actor_location.z + z_correction)

        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance <= 1e-6:
            return None, None

        pitch = math.degrees(math.asin(dz / distance)) + pitch_correction
        yaw = math.degrees(math.atan2(dy, dx))

        return yaw, pitch

    def update_action(self, action):
        if self.duration <= 0:
            self.action = action
            self.duration = self.max_duration


class FireAction(ActionInterface):
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
        self.target = None

        self.direct_action = DirectAction(
            {"x": 0, "y": 0},
            self.duration,
            self.grid_size,
            self.pitch_correction,
            self.z_correction,
        )

    def convert_single_action(
        self, action: "tuple[list, float]", done_state: bool = False
    ):
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
            self.direct_action.update_action({"x": action[0], "y": action[1]})
            return self.direct_action

    def get_action_mask(self, actor: carla.Actor, action_space: "tuple[dict, dict]"):
        """Fire action is always applicable

        Note:
            Ammo out should cause the actor to be done
        """
        self._find_target_actor()

        # Check if last action is done
        prev_x, prev_y = self.direct_action.action["x"], self.direct_action.action["y"]
        if prev_x != 9 and prev_y != 0 and self.direct_action.duration > 0:
            mask = (
                {action: 0 for action in action_space[0].values()},
                {action: 0 for action in action_space[1].values()},
            )
            mask[0][prev_x] = 1
            mask[1][prev_y] = 1
            return mask

        # Check whether ego is within the fire range
        if self.target is not None:
            if not self._is_target_within_range(
                actor, self.target, action_space, self.grid_size
            ):
                mask = (
                    {action: 0 for action in action_space[0].values()},
                    {action: 0 for action in action_space[1].values()},
                )
                mask[0][0] = 1
                mask[1][0] = 1
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
            return DirectAction({"x": 0, "y": 0})

        return (10, 0)

    def _find_target_actor(self):
        if self.target is None or not self.target.is_alive or not self.target.is_active:
            self.target = Simulator.get_actor_by_rolename(self.target_name)

    def _is_target_within_range(
        self,
        cannon: carla.Actor,
        target: carla.Actor,
        action_space: "tuple[dict, dict]",
        grid_size: int = 10,
    ) -> bool:
        target_location = target.get_location()
        cannon_transform = cannon.get_transform()
        horizontal_range = list(action_space[0].values())
        vertical_range = list(action_space[1].values())

        cannon_location = cannon_transform.location
        forward_vector = cannon_transform.get_forward_vector()
        right_vector = cannon_transform.get_right_vector()

        # Calculate the corners of the grid
        bottom_left = (
            cannon_location
            + (forward_vector * grid_size * vertical_range[0])
            + (right_vector * grid_size * horizontal_range[0])
        )
        bottom_right = (
            cannon_location
            + (forward_vector * grid_size * vertical_range[0])
            + (right_vector * grid_size * horizontal_range[-1])
        )
        top_left = (
            cannon_location
            + (forward_vector * grid_size * vertical_range[-1])
            + (right_vector * grid_size * horizontal_range[0])
        )
        top_right = (
            cannon_location
            + (forward_vector * grid_size * vertical_range[-1])
            + (right_vector * grid_size * horizontal_range[-1])
        )

        point = Point(target_location.x, target_location.y)
        polygon = Polygon(
            [
                (bottom_left.x, bottom_left.y),
                (top_left.x, top_left.y),
                (top_right.x, top_right.y),
                (bottom_right.x, bottom_right.y),
            ]
        )

        return polygon.contains(point)
