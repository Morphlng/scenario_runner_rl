import contextlib
import importlib
import inspect
import io
import json
import logging
import os

import ray
from marllib import marl
from ray import tune
from ray.rllib.agents.trainer import Trainer
from ray.rllib.models import ModelCatalog


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class NullLogger:
    """Logger for RLlib to disable logging"""

    def __init__(self, config=None):
        self.config = config
        self.logdir = ""

    def _init(self):
        pass

    def on_result(self, result):
        pass

    def update_config(self, config):
        pass

    def close(self):
        pass

    def flush(self):
        pass


def find_key(dictionary, target_key):
    if target_key in dictionary:
        return dictionary[target_key]

    for key, value in dictionary.items():
        if isinstance(value, dict):
            result = find_key(value, target_key)
            if result is not None:
                return result

    return None


def form_algo_dict() -> "dict[str, tuple[str, Trainer]]":
    """Form a dictionary of all available algorithms in MARLlib

    Returns:
        dict[Algo_name, (Algo_type, Trainer_class)]: Dictionary of all available algorithms in MARLlib

        e.g. {"mappo": ("CC", MAPPOTrainer)}
    """

    trainers_dict = {}

    core_path = os.path.join(os.path.dirname(marl.__file__), "algos/core")
    for algo_type in os.listdir(core_path):
        if not os.path.isdir(os.path.join(core_path, algo_type)):
            continue
        for algo in os.listdir(os.path.join(core_path, algo_type)):
            if algo.endswith(".py") and not algo.startswith("__"):
                module_name = algo[:-3]  # remove .py extension
                module_path = f"marllib.marl.algos.core.{algo_type}.{module_name}"
                module = importlib.import_module(module_path)

                trainer_class_name = module_name.upper() + "Trainer"
                trainer_class = getattr(module, trainer_class_name, None)
                if trainer_class is None:
                    for name, obj in inspect.getmembers(module):
                        if name.endswith("Trainer"):
                            trainers_dict[module_name] = obj
                else:
                    trainers_dict[module_name] = (algo_type, trainer_class)

    return trainers_dict


def update_config(config: dict):
    # Extract config
    map_name = find_key(config, "map_name")
    model_name = find_key(config, "custom_model")
    model_arch_args = find_key(config, "model_arch_args")
    algo_name = find_key(config, "algorithm")
    share_policy = find_key(config, "share_policy")

    ######################
    ### environment info ###
    ######################
    env = marl.make_env("macad", map_name)
    env_instance, env_info = env
    algorithm = dotdict({"name": algo_name, "algo_type": ALGO_DICT[algo_name][0]})
    model_instance, model_info = marl.build_model(env, algorithm, model_arch_args)
    ModelCatalog.register_custom_model(model_name, model_instance)

    env_info = env_instance.get_env_info()
    policy_mapping_info = env_info["policy_mapping_info"]
    agent_name_ls = env_instance.agents
    env_info["agent_name_ls"] = agent_name_ls
    env_instance.close()

    config["model"]["custom_model_config"].update(env_info)

    ######################
    ### policy sharing ###
    ######################

    if "all_scenario" in policy_mapping_info:
        policy_mapping_info = policy_mapping_info["all_scenario"]
    else:
        policy_mapping_info = policy_mapping_info[map_name]

    if share_policy == "all":
        if not policy_mapping_info["all_agents_one_policy"]:
            raise ValueError(
                "in {}, policy can not be shared, change it to 1. group 2. individual".format(
                    map_name
                )
            )

        policies = {"av"}
        policy_mapping_fn = lambda agent_id, episode, **kwargs: "av"

    elif share_policy == "group":
        groups = policy_mapping_info["team_prefix"]

        if len(groups) == 1:
            if not policy_mapping_info["all_agents_one_policy"]:
                raise ValueError(
                    "in {}, policy can not be shared, change it to 1. group 2. individual".format(
                        map_name
                    )
                )

            policies = {"shared_policy"}
            policy_mapping_fn = lambda agent_id, episode, **kwargs: "shared_policy"

        else:
            policies = {
                "policy_{}".format(i): (
                    None,
                    env_info["space_obs"],
                    env_info["space_act"],
                    {},
                )
                for i in groups
            }
            policy_ids = list(policies.keys())
            policy_mapping_fn = tune.function(
                lambda agent_id: "policy_{}_".format(agent_id.split("_")[0])
            )

    elif share_policy == "individual":
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError(
                "in {}, agent number too large, we disable no sharing function".format(
                    map_name
                )
            )

        policies = {
            "policy_{}".format(i): (
                None,
                env_info["space_obs"],
                env_info["space_act"],
                {},
            )
            for i in range(env_info["num_agents"])
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[agent_name_ls.index(agent_id)]
        )

    else:
        raise ValueError("wrong share_policy {}".format(share_policy))

    # if happo or hatrpo, force individual
    if algo_name in ["happo", "hatrpo"]:
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError(
                "in {}, agent number too large, we disable no sharing function".format(
                    map_name
                )
            )

        policies = {
            "policy_{}".format(i): (
                None,
                env_info["space_obs"],
                env_info["space_act"],
                {},
            )
            for i in range(env_info["num_agents"])
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[agent_name_ls.index(agent_id)]
        )

    config.update(
        {"multiagent": {"policies": policies, "policy_mapping_fn": policy_mapping_fn}}
    )


def load_model(
    model_path: str, params_path: str, algo: str = None
) -> "tuple[Trainer, function]":
    """load model from given path

    Args:
        model_path (str): path to model
        params_path (str): path to params
        algo (str, optional): algorithm name, e.g. mappo. Defaults to None.

    Returns:
        agent (Trainer): agent object
        policy_mapping_fn (function): policy mapping function (only for multi-agent)
    """
    try:
        with open(params_path, "r") as f:
            params = json.load(f)
    except Exception as e:
        print("Error loading params: ", e)
        raise e

    silent_output = io.StringIO()
    with contextlib.redirect_stdout(silent_output):
        if not ray.is_initialized():
            ray.init(
                include_dashboard=False,
                configure_logging=True,
                logging_level=logging.ERROR,
                log_to_driver=False,
            )

        update_config(params)
        trainer: Trainer = ALGO_DICT[algo or find_key(params, "algorithm")][1](
            params, logger_creator=lambda config: NullLogger(config)
        )
        trainer.restore(model_path)

    pmap = find_key(trainer.config, "policy_mapping_fn")

    return trainer, pmap


ALGO_DICT = form_algo_dict()


if __name__ == "__main__":
    # Example of getting model
    agent, pmap = load_model(
        model_path="/home/morphlng/ray_results/Town01_no_type/checkpoint_000645/checkpoint-645",
        params_path="/home/morphlng/ray_results/Town01_no_type/params.json",
    )

    # RNN-based model have state
    agent_id = "car1"
    state = agent.get_policy(pmap(agent_id, 1)).get_initial_state()
