import json
import operator

from py_trees.blackboard import Blackboard

from srunner.extension.rl_integrate.data.simulator import Simulator
from srunner.extension.rl_integrate.macad_agent.rl_agent import RlAgent, RlManager
from srunner.extension.rl_integrate.misc import str2bool, str2list
from srunner.scenariomanager.actorcontrols.basic_control import BasicControl


class RlAgentControl(BasicControl):
    """
    Controller class for vehicles derived from BasicControl.

    This controller wraps the RLAgent class to control agent through
    the MACAD framework.
    """

    def __init__(self, actor, configs: dict):
        """
        Args:
            actor (carla.Actor): Vehicle actor that should be controlled.
            configs (dict): Dictionary containing the configuration for the controller.
        """
        super(RlAgentControl, self).__init__(actor)
        blackboard = Blackboard()

        try:
            check_actors = operator.attrgetter("rl_actors")
            rl_actors = check_actors(blackboard)
        except AttributeError:
            RlManager.reset()

        if configs is not None:
            if "config_file_path" in configs:
                with open(configs["config_file_path"], "r") as f:
                    configs = json.load(f)

            actor_loc = actor.get_location()
            start_pos = [actor_loc.x, actor_loc.y, actor_loc.z]
            end_pos = configs.get("end_pos", "")
            if end_pos != "":
                end_pos = str2list(end_pos, convert_to=float)
            else:
                end_pos = start_pos

            self._target_speed = float(configs.get("target_speed", 0)) / 3.6
            self._init_speed = float(configs.get("init_speed", 0)) / 3.6

            update_dict = {
                "actor_id": configs.get(
                    "actor_id",
                    actor.attributes.get(
                        "role_name", f"agent_{len(blackboard.get('rl_actors'))}"
                    ).lower(),
                ),
                "type": configs.get(
                    "type",
                    "vehicle_4w"
                    if actor.type_id.startswith("vehicle")
                    else "walker"
                    if actor.type_id.startswith("walker")
                    else "static_obstacle",
                ),
                "action_type": configs.get("action_type", "pseudo_action"),
                "enable_planner": str2bool(configs.get("enable_planner", "false")),
                "send_measurements": str2bool(configs.get("send_measurements", "true")),
                "measurement_type": str2list(configs.get("measurement_type", "all")),
                "focus_actors": str2list(configs.get("focus_actors", "all")),
                "ignore_actors": str2list(configs.get("ignore_actors", "")),
                "add_action_mask": str2bool(configs.get("add_action_mask", "false")),
                "target_speed": self._target_speed,
                "init_speed": self._init_speed,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "model_config": {
                    "model_path": configs.get("model_path", None),
                    "params_path": configs.get("params_path", None),
                }
                if (configs.get("model_path", "") != "")
                else None,
            }
            configs.update(update_dict)

            from srunner.autoagents.agent_wrapper import AgentWrapper

            self._wrapped_agent = AgentWrapper(RlAgent(configs))
            self._wrapped_agent.setup_sensors(actor)

    def reset(self):
        """
        Reset the controller
        """
        self._wrapped_agent.cleanup()
        self._actor = None

    def run_step(self):
        """
        Execute on tick of the controller's control loop
        """
        if self._reached_goal or self._target_speed == -1:
            Simulator.apply_actor_control(self._actor.id, "stop")
            return None

        if self._init_speed:
            Simulator.set_actor_speed(self._actor.id, self._init_speed)
            self._init_speed = False
        elif self._wrapped_agent._agent.actor_id in Blackboard().get("rl_measurements"):
            control = self._wrapped_agent()
            if control is not None:
                self._actor.apply_control(control)
            self._reached_goal = self._check_done()
        else:
            self._reached_goal = False

    def __del__(self):
        """
        Cleanup the controller
        """
        self.reset()

    def _check_done(self):
        """
        Check if the actor has reached the goal
        """
        if self._actor is None or not self._actor.is_active or not self._actor.is_alive:
            return True

        blackboard = Blackboard()
        rl_measurements = blackboard.get("rl_measurements")
        rl_path_trackers = blackboard.get("rl_path_trackers")
        rl_actor_configs = blackboard.get("rl_actor_configs")

        actor_id = self._wrapped_agent._agent.actor_id
        if actor_id not in rl_measurements:
            return False
        measurement = rl_measurements[actor_id]

        # Reach destination
        if (
            (actor_id in rl_path_trackers)
            and (rl_path_trackers[actor_id].is_done())
            or (0 < measurement.exp_info.distance_to_goal < 3)
            or (measurement.health_point <= 0)
        ):
            return True

        # Collision
        if str2bool(rl_actor_configs[actor_id].get("collision_sensor", "off")):
            collided = (
                measurement.collision.vehicles > 0
                or measurement.collision.pedestrians > 0
                or measurement.collision.others > 0
            )
            return bool(collided)

        return False
