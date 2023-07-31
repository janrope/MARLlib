# from ray.tune.integration.wandb import WandbLoggerCallback

from marllib import marl

print("make env")

# customize yours
env = marl.make_env(environment_name="rware", map_name="customized_map", n_agents=4, msg_bits=1, map_size="tiny")

print("init algo")
# initialize algorithm with appointed hyper-parameters
mappo = marl.algos.mappo(hyperparam_source="test")

print("build model")
# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

print("train")
# start training
mappo.fit(env, model, stop={'timesteps_total': 100000}, share_policy='group', num_workers=5, local_mode=False, num_gpus=1,
        #   callbacks=[WandbLoggerCallback(
        #                 project="marllib-init-test",
        #                 api_key="f2c78e1853776db64629bd880d537cc09e3e7a01",
        #                 log_config=True)]
)