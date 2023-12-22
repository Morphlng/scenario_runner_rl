from __future__ import annotations

import math

import carla

from srunner.extension.rl_integrate.macad_agent.action.action_interface import (
    AbstractAction,
    ActionInterface,
)


class DirectAction(AbstractAction):
    def __init__(self, action: dict, duration: int = 20):
        super().__init__(action, duration)
        self.max_duration = duration

    def run_step(self, actor: carla.Actor):
        """Return a carla control signal based on the actor type and the action

        Args:
            actor (carla.Actor): The actor to run the action

        Returns:
            carla.VehicleControl: The control signal
        """
        self.duration -= 1
        x, y = self.action["x"], self.action["y"]

        # (9, 0) points to self
        if (actor.ammo_left <= 0) or (x == 9 and y == 0):
            if self.duration == 0:
                self.duration = self.max_duration
            return None

        if self.duration == self.max_duration - 1:
            actor_transform = actor.get_transform()
            rotation = actor_transform.rotation
            location = actor_transform.location

            target_location = self.get_grid_center_location(
                location, rotation, grid_coord=(x, y)
            )
            yaw, pitch = self.fire_strike(location, target_location)
            actor.set_fire_angle(yaw, pitch)
        elif self.duration == 0:
            actor.open_fire()
            self.duration = self.max_duration

        return None

    @staticmethod
    def get_grid_center_location(
        cannon_location: carla.Location,
        cannon_rotation: carla.Rotation,
        grid_coord: "tuple[int, int]",
    ):
        grid_size = 10

        cannon_x, cannon_y, cannon_z = (
            cannon_location.x,
            cannon_location.y,
            cannon_location.z,
        )
        # correct facing deviation
        cannon_yaw = math.radians(cannon_rotation.yaw + 102.54)

        # offset of grid center to cannon
        grid_offset_x = grid_size * (grid_coord[0] - 9)
        grid_offset_y = grid_size * (grid_coord[1] - 0)

        # world location in Carla
        grid_center_x = (
            cannon_x
            + grid_offset_x * math.cos(cannon_yaw)
            - grid_offset_y * math.sin(cannon_yaw)
        )
        grid_center_y = (
            cannon_y
            + grid_offset_x * math.sin(cannon_yaw)
            + grid_offset_y * math.cos(cannon_yaw)
        )
        grid_center_z = cannon_z

        return carla.Location(x=grid_center_x, y=grid_center_y, z=grid_center_z)

    @staticmethod
    def fire_strike(actor_location: carla.Location, target_location: carla.Location):
        dx = target_location.x - actor_location.x
        dy = target_location.y - actor_location.y
        dz = target_location.z - actor_location.z

        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance <= 1e-6:
            return None, None

        pitch = math.degrees(math.asin(dz / distance))
        yaw = math.degrees(math.atan2(dy, dx))

        return yaw, pitch

    def update_action(self, action):
        self.action = action


class FireAction(ActionInterface):
    def __init__(self, action_config: dict):
        """Initialize the action converter for open fire action space

        Args:
            action_config (dict): A dictionary of action config
        """
        super().__init__(action_config)
        self.direct_action = DirectAction({"x": 9, "y": 0})

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

    def get_action_mask(self, actor, action_space):
        """Fire action is always applicable

        Note:
            Ammo out will cause the actor to be done
        """
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
            return DirectAction({"x": 9, "y": 0})

        return (9, 0)
