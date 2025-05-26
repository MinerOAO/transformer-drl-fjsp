import copy
import json
import os
import random
import time as time

import gymnasium
import pandas as pd
import torch
import numpy as np

import pynvml
import PPO_model
from env.load_data import nums_detec
from utils.gantt_chart import gantt_chart_plt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # PyTorch initialization
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    # Load config and init objects
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    test_paras = load_dict["test_paras"]
    env_paras["device"] = device
    model_paras["device"] = device
    env_test_paras = copy.deepcopy(env_paras)
    if test_paras["sample"]:
        env_test_paras["batch_size"] = test_paras["num_sample"]
    else:
        env_test_paras["batch_size"] = 1
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"] + 1 
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"] + 1 

    # model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    # model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    data_path = "./data_test/{0}/".format(test_paras["data_path"])
    test_files = []
    if test_paras["data_prefix"] == "":
        test_files = os.listdir(data_path)
    else:
        for root, ds, fs in os.walk(data_path):
            for f in fs:
                if f.startswith(test_paras["data_prefix"]):
                    test_files.append(f)
    test_files.sort(key=lambda x: x[:-4])
    num_ins = len(test_files)
    mod_files = os.listdir('./model/')[:]

    gantt = False

    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_paras, train_paras)
    rules = test_paras["rules"]
    envs = []  # Store multiple environments

    # Detect and add models to "rules"
    for root, ds, fs in os.walk('./model/'):
        for f in fs:
            if f.endswith('.pt'):
                rules.append(f)

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = f'./save/test_{str_time}_{test_paras["data_path"]}'
    os.makedirs(save_path)
    writer = pd.ExcelWriter(
        '{0}/makespan_{1}.xlsx'.format(save_path, str_time))  # Makespan data storage path
    writer_time = pd.ExcelWriter('{0}/time_{1}.xlsx'.format(save_path, str_time))  # time data storage path
    file_name = [test_files[i] for i in range(num_ins)]
    data_file = pd.DataFrame(file_name, columns=["file_name"])
    data_file.to_excel(writer, sheet_name='Sheet1', index=False)
    writer._save()
    # writer.close()
    data_file.to_excel(writer_time, sheet_name='Sheet1', index=False)
    writer_time._save()
    # writer_time.close()

    # Rule-by-rule (model-by-model) testing
    start = time.time()
    for i_rules in range(len(rules)): # replace with ?enumerate
        rule = rules[i_rules]
        # Load trained model
        if rule.endswith('.pt'):
            if device.type == 'cuda':
                model_CKPT = torch.load('./model/' + mod_files[i_rules], weights_only=True)
            else:
                model_CKPT = torch.load('./model/' + mod_files[i_rules], map_location='cpu', weights_only=True)
            print('\nloading checkpoint:', mod_files[i_rules])
            model.policy.load_state_dict(model_CKPT)
            model.policy_old.load_state_dict(model_CKPT)
        print('rule:', rule)

        # Schedule instance by instance
        step_time_last = time.time()
        makespans = []
        times = []
        for i_ins in range(num_ins):
            test_file = data_path + test_files[i_ins]
            with open(test_file) as file_object:
                line = file_object.readlines()
                ins_num_jobs, ins_num_mas, _ = nums_detec(line)
            env_test_paras["num_jobs"] = ins_num_jobs
            env_test_paras["num_mas"] = ins_num_mas

            env_creation_time = time.time()
            # Environment object already exists
            if len(envs) == num_ins:
                env = envs[i_ins]
            # Create environment object
            else:
                # Clear the existing environment
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if meminfo.used / meminfo.total > 0.7:
                    envs.clear()
                # DRL-S, each env contains multiple (=num_sample) copies of one instance
                if test_paras["sample"]:
                    env = gymnasium.make('fjsp-v0', case=[test_file] * test_paras["num_sample"],
                                   env_paras=env_test_paras, data_source='file')
                # DRL-G, each env contains one instance
                else:
                    env = gymnasium.make('fjsp-v0', case=[test_file], env_paras=env_test_paras, data_source='file')
                env.reset()
                envs.append(copy.deepcopy(env))
                # print("Create env[{0}]".format(i_ins))
            # print(f"Env creation time:{time.time() - env_creation_time}")

            # Schedule an instance/environment
            # DRL-S
            if test_paras["sample"]:
                makespan, time_re = schedule(env, model, memories, flag_sample=test_paras["sample"])
                makespans.append(torch.min(makespan))
                times.append(time_re)
            # DRL-G
            else:
                time_s = []
                makespan_s = []  # In fact, the results obtained by DRL-G do not change
                for j in range(test_paras["num_average"]): # num_average times
                    makespan, time_re = schedule(env, model, memories)
                    makespan_s.append(makespan)
                    time_s.append(time_re)
                    if gantt:
                        gantt_chart_plt(env.schedules_batch[0], env.nums_ope_batch[0], env.num_ope_biases_batch[0], env.num_jobs, int(makespan.item()), ins_name=test_files[i_ins])
                    env.reset()
                makespans.append(torch.mean(torch.tensor(makespan_s)))
                times.append(torch.mean(torch.tensor(time_s)))
            print(f"finish env {i_ins}")
        print("rule_spend_time: ", time.time() - step_time_last)

        # Save makespan and time data to files
        baseline = pd.read_excel("./history_save/benchmark baseline.xlsx")
        baseline_dict = dict(zip(baseline['Name'], baseline['Value']))
        baseline_value = {k: baseline_dict.get(k) for k in file_name}
        baseline_value = baseline_value.values()

        makespan_list = torch.tensor(makespans).t().tolist()
        data = pd.DataFrame(makespan_list, columns=[rule])
        data.to_excel(writer, sheet_name='Sheet1', index=False, startcol=i_rules + 1)
        writer._save()

        gap_data = [(y - float(x))/float(x) for x, y in zip(baseline_value, makespan_list)]
        data = pd.DataFrame(gap_data, columns=["Gap"])
        data.to_excel(writer, sheet_name='Sheet1', index=False, startcol=i_rules + 2)
        writer._save()
        # writer.close()
        data = pd.DataFrame(torch.tensor(times).t().tolist(), columns=[rule])
        data.to_excel(writer_time, sheet_name='Sheet1', index=False, startcol=i_rules + 1)
        writer_time._save()
        # writer_time.close()

        for env in envs:
            env.reset()
        
    writer.close()
    writer_time.close()
    print("total_spend_time: ", time.time() - start) # Collect total time among all instances Need improved TODO

def schedule(env, model, memories, flag_sample=False):
    # Get state and completion signal
    state = env.state
    dones = env.done_batch
    done = False  # Unfinished at the beginning
    last_time = time.time()
    i = 0
    while ~done:
        i += 1
        with torch.no_grad():
            actions = model.policy_old.act(state, memories, dones, flag_sample=flag_sample, flag_train=False)
        state, rewards, dones = env.step(actions)  # environment transit
        done = dones.all()
    spend_time = time.time() - last_time  # The time taken to solve this environment (instance)

    # Verify the solution
    # gantt_result = env.validate_gantt()[0]
    # if not gantt_result:
    #     print("Scheduling Error！！！！！！")
    return copy.deepcopy(env.makespan_batch), spend_time


if __name__ == '__main__':
    main()