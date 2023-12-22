## ScenarioRunner-RL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is a fork from the official [carla-simulator/scenario_runner](https://github.com/carla-simulator/scenario_runner). It is meant to be used combined with model/checkpoint trained by our fork of [MARLlib](https://github.com/Morphlng/MARLlib).

> Currently based on branch tag/0.9.13.

## Getting started

Currently, you can try with:

```bash
cd scenario_runner

python scenario_runner.py --host 127.0.0.1 --port 2000 --openscenario srunner/examples/RLTest_Town01.xosc --sync --reloadWorld 
```

> Note: In order to keep the repository small, we will distribute the trained model/checkpoint in [Releases](https://github.com/Morphlng/scenario_runner_rl/releases). You can download them and put them in the `srunner/examples/rl_models` folder.

## Changes

The RL inference support is added as an "extension" to the original scenario_runner, with minimal changes to the original code. This project should be fully compatible to run non-RL scenarios.

### Added files

1. srunner/extension. Folder for extensions, currently only `rl_integrate` folder.
2. srunner/examples/rl_models. Folder for RL models/checkpoints.

### Modified files

1. srunner/autoagents/agent_wrapper.py

   - Code formatting and type annotations.
   - Import `rl_integrate` module. This is done in a try-except block, so that the original scenario_runner can still run without the module.
   - Add support for `RLAgent` class, which is a wrapper for RL inference.
   - Complement the destruction logic for `AgentWrapper` class.
2. srunner/autoagents/sensor_interface.py

   - Add support for `carla.ColorConverter`. This will not change the default behavior of the original scenario_runner.
3. srunner/scenariomanager/carla_data_provider.py

   - Code formatting and type annotations.
   - Add new data pool `_actor_velocity_vector_map` and `_actor_acceleration_map`.
   - For all `get_xxx` functions, add new function called `get_xxx_by_id`, which takes an actor id as input.
   - Other changes are made for our custom UE4 build, which should not affect the original scenario_runner.
4. srunner/scenariomanager/scenario_manager.py

   - Import `rl_integrate` module. This is done in a try-except block, so that the original scenario_runner can still run without the module.
   - Register all agents to RL blackboard
   - on each tick, update information for RL (`RlAgent.on_carla_tick()`)
   - add some function to check and enable heterogeneous agents action space padding.
5. srunner/examples/catalogs/ControllerCatalog.xosc
    - Add a new controller catagory "RLControl"
