from __future__ import annotations

import importlib
import itertools
import os
import sys
from typing import Any

import numpy as np
from gym.spaces import Box, Discrete, MultiDiscrete, Space

from srunner.extension.rl_integrate.macad_agent.action.action_interface import (
    AbstractAction,
    ActionInterface,
)
from srunner.extension.rl_integrate.macad_agent.assets import ENV_ASSETS


class AgentAction:
    def __init__(self, action_config: dict):
        """Initialize action space for a single agent

        Args:
            action_config (dict): The action configuration
        """
        self.action_config: dict = action_config
        self.action_type: str = action_config["type"]
        self.is_discrete: bool = action_config.get("use_discrete", True)
        self.discrete_action_set: "dict[int, Any] | tuple[dict]" = action_config.get(
            "discrete_action_set", None
        )

        if self.is_discrete and self.discrete_action_set is None:
            attr_name = (
                f"default_{self.action_type.split('_action')[0]}_discrete_actions"
            )
            self.discrete_action_set = getattr(ENV_ASSETS, attr_name, None)
            if self.discrete_action_set is None:
                raise ValueError(
                    f"Please provide a discrete action set for {self.action_type}"
                )

        self.action_padding = False
        self.space_type = self.get_space_type()
        self.action_space = self.get_action_space()
        self._action_handler = self._init_action_handler()

    def _init_action_handler(self) -> ActionInterface:
        """Initilize the action handler based on the action type

        This can either be a path to a python file or a python module. For example:

        1. "macad_gym/carla/action/custom_action.py".
        2. "custom_action". By default, the module will be searched in the current directory.
        """
        if ".py" in self.action_type:
            module_name = os.path.basename(self.action_type).split(".")[0]
            sys.path.append(os.path.dirname(self.action_type))
            module_control = importlib.import_module(module_name)
            action_handler_name = module_control.__name__.title().replace("_", "")
        else:
            sys.path.append(os.path.dirname(__file__))
            module_control = importlib.import_module(self.action_type)
            action_handler_name = (
                self.action_type.split(".")[-1].title().replace("_", "")
            )

        # Initialize and return the class
        return getattr(module_control, action_handler_name)(self.action_config)

    def convert_single_action(
        self, action: "int | Any", done_state=False
    ) -> AbstractAction:
        """Convert a model output action to an AbstractAction

        Args:
            action (int | Any): The action to be converted
            done_state (bool, optional): Whether the actor is done. Defaults to False.

        Returns:
            AbstractAction: An action instance
        """
        if self.action_padding:
            action = self._pad2origin_action(action)

        if self.space_type == "Discrete":
            action = self.discrete_action_set[int(action)]
        elif self.space_type == "MultiDiscrete":
            action = [
                action_set[int(action[i])]
                for i, action_set in enumerate(self.discrete_action_set)
            ]

        return self._action_handler.convert_single_action(action, done_state)

    def get_space_type(self) -> str:
        """Return the space type of the action space"""
        if self.is_discrete:
            if isinstance(self.discrete_action_set, dict):
                return "Discrete"
            else:
                return "MultiDiscrete"
        else:
            return "Box"

    def get_stop_action(self, abstrct_action: bool = False) -> "int | Any":
        """Return a stop action to patch the action input to env

        Returns:
            int | Any: A valid value in action space, representing stop behavior
        """
        return self._action_handler.stop_action(
            env_action=(not abstrct_action), use_discrete=self.is_discrete
        )

    def get_action_space(self) -> Space:
        """Get the action space for this agent"""
        if self.space_type == "Discrete":
            action_space = Discrete(len(self.discrete_action_set))
        elif self.space_type == "MultiDiscrete":
            action_space = MultiDiscrete(
                [len(action_set) for action_set in self.discrete_action_set]
            )
        else:
            # TODO: configure the action space
            action_space = Box(-np.inf, np.inf, shape=(2,))

        return action_space

    def get_action_mask(self, actor) -> np.ndarray:
        """Get the action mask for a given actor

        Args:
            actor (carla.Actor): The actor

        Returns:
            np.ndarray: A numpy array of action mask
        """
        action_mask = self._action_handler.get_action_mask(
            actor, self.discrete_action_set
        )

        if action_mask == True:
            if self.space_type == "MultiDiscrete":
                mask = np.ones(
                    sum([len(action_set) for action_set in self.discrete_action_set]),
                    dtype=np.float32,
                )
            else:
                mask = np.ones(len(self.discrete_action_set), dtype=np.float32)
        elif isinstance(action_mask, dict):
            # Discrete
            mask = np.array(
                [action_mask[action] for action in self.discrete_action_set.values()],
                dtype=np.float32,
            )
        elif isinstance(action_mask, tuple):
            # MultiDiscrete
            mask = np.concatenate(
                [
                    [action_mask[idx][action] for action in space.values()]
                    for idx, space in enumerate(self.discrete_action_set)
                ],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Invalid action mask type: {type(action_mask)}")

        if self.action_padding:
            start_idx, end_idx = self.valid_actions[0], self.valid_actions[-1]
            new_mask = np.zeros(self.padding_size, dtype=np.float32)
            new_mask[start_idx : end_idx + 1] = self._mask_in_pad_space(mask)
            mask = new_mask

        return mask

    def enable_action_padding(self, padding_size: int):
        """Enable action padding

        Args:
            padding_size (int): The padding size
        """
        self.action_padding = True
        self.padding_size = padding_size
        self.valid_actions = self.get_valid_actions()

    def get_valid_actions(self):
        """Get valid actions in a padded action space

        Returns:
            list: A list of valid actions (start from 0)
        """

        if isinstance(self.action_space, Discrete):
            return list(range(self.action_space.n))
        elif isinstance(self.action_space, MultiDiscrete):
            return list(range(np.prod(self.action_space.nvec)))
        else:
            raise ValueError(
                "Action padding is not supported for Continuous action space"
            )

    def _pad2origin_action(self, action):
        """Convert the padded action to the original action space

        Args:
            action (np.ndarray): The action to be padded

        Returns:
            list | int: The original action
        """
        if isinstance(self.action_space, Discrete):
            return np.clip(action, 0, self.action_space.n - 1)
        elif isinstance(self.action_space, MultiDiscrete):
            multi_discrete_values = []
            for n in reversed(self.action_space.nvec):
                multi_discrete_values.append(action % n)
                action //= n
            return list(reversed(multi_discrete_values))

    def _mask_in_pad_space(self, origin_mask):
        """Pad the action mask

        Args:
            origin_mask (np.ndarray): The origin action mask

        Returns:
            np.ndarray: The padded action mask
        """
        if not isinstance(self.action_space, MultiDiscrete):
            # For Discrete spaces, the mask remains unchanged
            return origin_mask

        # Split the mask into sections based on the MultiDiscrete dimensions
        sections = []
        start_idx = 0
        for dim in self.action_space.nvec:
            sections.append(origin_mask[start_idx : start_idx + dim])
            start_idx += dim

        # Generate valid combinations
        new_mask = []
        for combination in itertools.product(*sections):
            # 1 if all values in the combination are valid (i.e., 1 in the original mask), 0 otherwise
            new_mask.append(int(all(combination)))

        return new_mask
