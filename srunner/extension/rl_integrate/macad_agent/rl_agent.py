from __future__ import annotations

import json
import math
import operator

import carla
from py_trees.blackboard import Blackboard

from srunner.extension.rl_integrate.data.data_collector import DataCollector
from srunner.extension.rl_integrate.data.local_carla_api import Waypoint
from srunner.extension.rl_integrate.data.simulator import Simulator
from srunner.extension.rl_integrate.macad_agent.action.agent_action import AgentAction
from srunner.extension.rl_integrate.macad_agent.assets import ENV_ASSETS
from srunner.extension.rl_integrate.macad_agent.macad_agent import MacadAgent
from srunner.extension.rl_integrate.macad_agent.model import load_model
from srunner.extension.rl_integrate.macad_agent.obs.observation import Observation
from srunner.extension.rl_integrate.macad_agent.path_tracker import PathTracker


class RlManager:
    @staticmethod
    def reset():
        blackboard = Blackboard()
        blackboard.set("rl_measurements", {}, True)
        blackboard.set("rl_static_object_measurements", {}, True)
        blackboard.set("rl_actor_configs", {}, True)
        blackboard.set("rl_actors", {}, True)
        blackboard.set("rl_agents", {}, True)
        blackboard.set("rl_path_trackers", {}, True)
        blackboard.set("rl_models", {}, True)
        blackboard.set("rl_global_sensor", None, True)

    @staticmethod
    def remove_by_name(actor_id):
        blackboard = Blackboard()

        rl_actors = blackboard.get("rl_actors")
        rl_actors.pop(actor_id, None)

        if len(rl_actors) == 0:
            RlManager.reset()
        else:
            blackboard.get("rl_actor_configs").pop(actor_id, None)
            blackboard.get("rl_path_trackers").pop(actor_id, None)
            blackboard.get("rl_agents").pop(actor_id, None)

    @staticmethod
    def register_model(model_path, params_path):
        blackboard = Blackboard()
        try:
            check_model = operator.attrgetter("rl_models")
            rl_models = check_model(blackboard)
        except AttributeError:
            rl_models = {}
            blackboard.set("rl_models", rl_models)

        if model_path not in rl_models:
            model, pmap = load_model(model_path, params_path)
            rl_models[model_path] = (model, pmap)
        else:
            model, pmap = rl_models[model_path]

        return model, pmap


class RlAgent(MacadAgent):
    """
    Reinforcement Learning Agent to control the ego via input actions
    """

    def setup(self, config: str | dict):
        """
        Setup the agent parameters
        """
        super().setup(config)

        self.actor_id = self.actor_config["actor_id"]
        focus_actors = set(self.actor_config.get("focus_actors", ["all"]))
        ignore_actors = set(self.actor_config.get("ignore_actors", []))
        measurement_type = set(self.actor_config.get("measurement_type", ["all"]))
        self.actor_config.update(
            {
                "focus_actors": focus_actors,
                "ignore_actors": ignore_actors,
                "measurement_type": measurement_type,
            }
        )

        blackboard = Blackboard()
        try:
            check_actor_configs = operator.attrgetter("rl_actor_configs")
            rl_actor_configs = check_actor_configs(blackboard)
        except AttributeError:
            rl_actor_configs = {}
            blackboard.set("rl_actor_configs", rl_actor_configs)
        rl_actor_configs.update({self.actor_id: self.actor_config})

        discrte_action_set = self.actor_config.get("discrete_action_set", "")
        if discrte_action_set != "":
            discrte_action_set = json.loads(discrte_action_set)
            if isinstance(discrte_action_set, dict):
                discrte_action_set = {int(k): v for k, v in discrte_action_set.items()}
            else:
                discrte_action_set = [
                    {int(k): v for k, v in discrte_action_set[i].items()}
                    for i in range(len(discrte_action_set))
                ]
        else:
            discrte_action_set = None

        action_extra_config = self.actor_config.get("action_extra_config", "")
        if action_extra_config != "":
            extra_config = json.loads(action_extra_config)
        else:
            extra_config = {}

        self.agent_action = AgentAction(
            {
                "type": self.actor_config.get("action_type", "low_level_action"),
                "discrete_action_set": discrte_action_set,
                **extra_config,
            }
        )
        self.abstract_action = None
        self.actor = None
        self.model = None
        self._state = None

        if self.actor_config.get("model_config", None):
            self.model, self.pmap = RlManager.register_model(
                self.actor_config["model_config"]["model_path"],
                self.actor_config["model_config"]["params_path"],
            )
            self._state = self.model.get_policy(
                self.pmap(self.actor_id, 1)
            ).get_initial_state()

        blackboard.get("rl_agents")[self.actor_id] = self

    def destroy(self):
        super().destroy()
        RlManager.remove_by_name(self.actor_id)

    def run_step(self, observation, timestamp=None):
        """
        Args
            observation: observation from data collector.

            timestamp: not used

        Return:
            Carla Control
        """
        planned_action = carla.VehicleControl()
        if "static" in self.actor_config.get("type", "vehicle_4w"):
            planned_action.hand_brake = True
            planned_action.brake = 1.0
            return planned_action
        elif self.actor_config.get("enable_planner", False):
            planned_action = (
                Blackboard().get("rl_path_trackers")[self.actor_id].run_step()
            )

        if self.model is None:
            return planned_action

        if self.abstract_action is None or self.abstract_action.done():
            output, self._state, _ = self.model.compute_single_action(
                observation,
                self._state,
                policy_id=self.pmap(self.actor_id, 1),
                explore=True,
            )
            self.abstract_action = self.agent_action.convert_single_action(
                output, False
            )
            self.abstract_action.duration = min(
                self.abstract_action.duration, ENV_ASSETS.step_ticks
            )

        return self.abstract_action.run_step(self.actor)

    def __call__(self):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        if self.actor is None:
            self.actor = Simulator.get_actor_by_id(
                Blackboard().get("rl_actors")[self.actor_id]
            )
        elif not self.actor.is_alive or not self.actor.is_active:
            return None

        # Create a dummy obs for debug like this:
        # if self.model is not None:
        #     observation = self.model.raw_user_config["model"]["custom_model_config"]["space_obs"].sample()

        origin_obs = Observation.encode_obs(
            self.actor_id,
            Blackboard().get("rl_measurements"),
            self.agent_action.get_action_mask(self.actor),
        )

        observation = {"obs": origin_obs["state"]}
        if "action_mask" in origin_obs:
            observation["action_mask"] = origin_obs["action_mask"]

        control = self.run_step(observation, timestamp=None)
        return control

    @staticmethod
    def on_carla_tick():
        blackboard = Blackboard()
        rl_actors: "dict[str, int]" = blackboard.get("rl_actors")
        actor_configs: "dict[str, dict]" = blackboard.get("rl_actor_configs")
        path_trackers: "dict[str, PathTracker]" = blackboard.get("rl_path_trackers")
        if rl_actors is None:
            return

        id_to_actor = {v: k for k, v in rl_actors.items()}
        all_actors = Simulator.data_provider.get_actors()
        misc_cnt = 0

        actor_map = {}
        for id, _ in all_actors:
            name = id_to_actor.get(id, None)
            if name is None:
                name = f"misc_{misc_cnt}"
                misc_cnt += 1
            actor_map[name] = id

        py_measurements = DataCollector.get_data(actor_map, actor_configs)
        for actor_id, measurement in py_measurements.items():
            config = actor_configs.get(actor_id, None)

            if config is not None:
                if config.get("enable_planner", False):
                    path_tracker = path_trackers[actor_id]
                    distance_threshold = (
                        math.inf if actor_id in ["hero", "ego"] else 2.0
                    )
                    planned_waypoints = path_tracker.get_nearest_waypoints(
                        nums=7, interval=2.0, distance_threshold=distance_threshold
                    )
                else:
                    try:
                        wp = Simulator.get_actor_waypoint(rl_actors[actor_id])
                        plan, _ = PathTracker.generate_target_waypoint_list(
                            wp, 0, 20, measurement.transform.rotation.yaw
                        )
                        planned_waypoints = [p[0] for p in plan][:7] if plan else [wp]
                        while len(planned_waypoints) < 7:
                            last_wpt = planned_waypoints[-1]
                            next_wpt = last_wpt.next(2.0)
                            planned_waypoints.append(
                                next_wpt[0] if next_wpt else last_wpt
                            )
                    except:
                        planned_waypoints = [None] * 7

                measurement.planned_waypoint = Waypoint.from_simulator_waypoint(
                    planned_waypoints[0]
                )
                measurement.planned_waypoints = [
                    Waypoint.from_simulator_waypoint(wpt)
                    for wpt in planned_waypoints[2:]
                ]

                if planned_waypoints[0] is not None:
                    measurement.orientation_diff = (
                        DataCollector.calculate_orientation_diff(
                            measurement.transform,
                            measurement.planned_waypoint.transform,
                        )
                    )
                    measurement.road_offset = measurement.planned_waypoint.transform.inverse_transform_location(
                        measurement.transform.location
                    ).y

                start_pos = config.get("start_pos", None)
                end_pos = config.get("end_pos", None)
                if start_pos == end_pos:
                    measurement.exp_info.distance_to_goal = -1
                else:
                    measurement.exp_info.distance_to_goal = (
                        measurement.planned_waypoint.transform.location.distance(
                            carla.Location(*end_pos)
                        )
                    )
            else:
                measurement.planned_waypoints = [Waypoint() for _ in range(5)]
                measurement.planned_waypoint = measurement.planned_waypoints[0]
                measurement.road_offset = 0

        blackboard.set("rl_measurements", py_measurements, True)

    @staticmethod
    def register_agent(actor: carla.Actor, actor_config: dict = None):
        blackboard = Blackboard()
        rl_actors = blackboard.get("rl_actors")

        if actor.id in rl_actors.values():
            return

        role_name = actor.attributes.get("role_name", f"agent_{len(rl_actors)}")
        if actor.type_id.startswith("vehicle"):
            actor_type = "vehicle_4w"
        elif actor.type_id.startswith("walker"):
            actor_type = "walker"
        else:
            actor_type = "obstacle"

        if actor_config is None:
            actor_config = {
                "actor_id": role_name.lower(),
                "type": actor_type,
                "enable_planner": False,
                "send_measurements": True,
                "measurement_type": ["all"],
                "focus_actors": set(["all"]),
                "ignore_actors": set(),
                "collision_sensor": "off",
                "lane_sensor": "off",
            }

        actor_id = actor_config["actor_id"]
        blackboard.get("rl_actors")[actor_id] = actor.id
        blackboard.get("rl_actor_configs")[actor_id] = actor_config

        if actor_config["enable_planner"]:
            blackboard.get("rl_path_trackers")[actor_id] = PathTracker(
                actor,
                actor_config["start_pos"],
                actor_config["end_pos"],
                actor_config.get("target_speed", 5.55) * 3.6,
            )
