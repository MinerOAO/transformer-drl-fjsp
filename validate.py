import gymnasium
# import gym
import env
import PPO_model
import torch
import time
import os
import copy

from env.case_generator import CaseGenerator

def get_validate_env(env_paras):
    '''
    Generate and return the validation environment from the validation set ()
    '''
    num_jobs = env_paras["num_jobs"]
    num_mas = env_paras["num_mas"]
    instance_num = env_paras["valid_batch_size"]
    file_path = "./data_dev/{0}{1}/".format(env_paras["num_jobs"], str.zfill(str(env_paras["num_mas"]),2))
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
        generator = CaseGenerator(num_jobs, num_mas, int(num_mas * 0.8), int(num_mas * 1.2), path=file_path, flag_doc=True, flag_same_opes=False)
        for i in range(instance_num):
            generator.get_case(i)

    valid_data_files = os.listdir(file_path)
    print(f"Vali File Path: {file_path}")
    for i in range(len(valid_data_files)):
        valid_data_files[i] = file_path+valid_data_files[i]
    env = gymnasium.make('fjsp-v0', case=valid_data_files, env_paras=env_paras, data_source='file')
    env.reset()
    return env

def validate(env_paras, model_policy):
    '''
    Validate the policy during training, and the process is similar to test
    '''
    env = get_validate_env(env_paras)
    start = time.time()
    batch_size = env_paras["valid_batch_size"]
    memory = PPO_model.Memory()
    print('There are {0} dev instances.'.format(batch_size))  # validation set is also called development set
    state = env.state
    done = False
    dones = env.done_batch
    while ~done:
        with torch.no_grad():
            actions = model_policy.act(state, memory, dones, flag_sample=False, flag_train=False)
        state, rewards, dones = env.step(actions)
        done = dones.all()
    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        print("Scheduling Error！！！！！！")
    makespan = copy.deepcopy(env.makespan_batch.mean())
    makespan_batch = copy.deepcopy(env.makespan_batch)
    env.reset()
    print(f"mean makespan: {makespan}")
    print('validating time: ', time.time() - start, '\n')
    return makespan, makespan_batch
