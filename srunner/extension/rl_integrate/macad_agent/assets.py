from __future__ import annotations

import math
from dataclasses import dataclass

from srunner.extension.rl_integrate.macad_agent.agents.navigation import RoadOption


@dataclass
class EnvAssets:
    distance_to_goal_threshold: float = 1.0
    """Threshold for reaching the goal."""

    orientation_to_goal_threshold: float = math.pi / 4.0
    """Threshold for considering the orientation to be aligned with the goal."""

    ground_z: float = 10.0
    """Dummy z value when we don't care about the z value."""

    action_ticks: int = 1
    """How many ticks each actor.apply_control() lasts."""

    step_ticks: int = 10
    """How many ticks each env.step() takes."""

    @property
    def commands_enum(self):
        """Number index to string like RoadOption mapping"""
        return {
            0.0: "REACH_GOAL",
            5.0: "GO_STRAIGHT",
            4.0: "TURN_RIGHT",
            3.0: "TURN_LEFT",
            2.0: "LANE_FOLLOW",
        }

    @property
    def command_ordinal(self):
        """String like RoadOption mapping to number index"""
        return {
            "REACH_GOAL": 0,
            "GO_STRAIGHT": 1,
            "TURN_RIGHT": 2,
            "TURN_LEFT": 3,
            "LANE_FOLLOW": 4,
            "CHANGE_LANE_LEFT": 5,
            "CHANGE_LANE_RIGHT": 6,
        }

    @property
    def road_option_to_commands(self):
        """RoadOption to string like RoadOption mapping"""
        return {
            RoadOption.VOID: "REACH_GOAL",
            RoadOption.STRAIGHT: "GO_STRAIGHT",
            RoadOption.RIGHT: "TURN_RIGHT",
            RoadOption.LEFT: "TURN_LEFT",
            RoadOption.LANEFOLLOW: "LANE_FOLLOW",
            RoadOption.CHANGELANELEFT: "CHANGE_LANE_LEFT",
            RoadOption.CHANGELANERIGHT: "CHANGE_LANE_RIGHT",
        }

    @property
    def default_actor_done_criteria(self):
        """The default done criteria for the actor."""
        return ["timeout"]

    @property
    def default_episode_done_criteria(self):
        """The default done criteria for the env."""
        return [
            "npc_done",
            "ego_collision",
            "ego_offroad",
            "ego_reach_goal",
            "ego_timeout",
        ]

    @property
    def default_low_level_discrete_actions(self):
        """The default discrete action set for low_level (throttle/steer) control"""
        return {
            0: [-0.5, 0.0],
            1: [-0.15, -0.15],
            2: [-0.15, 0.15],
            3: [0.0, 0.0],
            4: [0.25, -0.3],
            5: [0.25, 0.3],
            6: [0.75, -0.15],
            7: [0.75, 0.15],
            8: [1.0, 0.0],
        }

    @property
    def default_vehicle_atomic_discrete_actions(self):
        """The default discrete action set for atomic control"""
        return (
            # Planning actions
            {
                0: "stop",
                1: "lane_follow",
                2: "left_lane_change",
                3: "right_lane_change",
                4: "turn_left",
                5: "turn_right",
            },
            # Target speed
            {
                0: 0,
                1: 6,
                2: 12,
                3: 18,
                4: 24,
            },
        )

    @property
    def default_vehicle_route_discrete_actions(self):
        """The default discrete action set for vehicle follow control"""
        return {0: 0, 1: 6, 2: 12, 3: 18, 4: 24}

    @property
    def default_walker_discrete_actions(self):
        """The default discrete action set for walker control"""
        return (
            # direction
            {
                0: "stay",
                1: "front",
                2: "left",
                3: "right",
            },
            # target speed
            {i: i for i in range(4)},
        )

    @property
    def default_walker_speed_discrete_actions(self):
        """The default discrete action set for walker speed only control"""
        return {i: i for i in range(4)}

    @property
    def default_walker_patrol_discrete_actions(self):
        """The default discrete action set for walker patrol control"""
        return (
            # direction
            {
                0: "stable",
                1: "left",
                2: "right",
            },
            # target speed
            {0: 0, 1: 2, 2: 4, 3: 6},
        )

    @property
    def default_fire_discrete_actions(self):
        return ({i: i - 10 for i in range(21)}, {i: i for i in range(11)})

    @property
    def default_fire_cheat_discrete_actions(self):
        return {i: i for i in range(26)}

    @property
    def default_pseudo_discrete_actions(self):
        """The default discrete action set for pseudo control"""
        return {0: "null"}

    @property
    def default_action_conf(self):
        """Default action space config for an agent."""
        return {
            "type": "low_level_action",
            "use_discrete": True,
            "discrete_action_set": self.default_low_level_discrete_actions,
            "action_range": {
                "throttle": [0, 1.0],
                "brake": [-1.0, 0],
                "steer": [-0.5, 0.5],
            },
            "preprocess_fn": None,
        }


ENV_ASSETS = EnvAssets()

__all__ = ["ENV_ASSETS"]
