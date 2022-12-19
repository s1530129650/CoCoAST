import pickle
import os
import json
import prettytable as pt
import numpy as np
import math
import logging
logger = logging.getLogger(__name__)

def save_pickle_data(path_dir, filename, data):
    full_path = path_dir + '/' + filename
    print("Save dataset to: %s" % full_path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    with open(full_path, 'wb') as output:
        pickle.dump(data, output,protocol=4)


def read_json_file(filename):
    with open(filename, 'r') as fp:
        data = fp.readlines()
    if len(data) == 1:
        data = json.loads(data[0])
    else:
        data = [json.loads(line) for line in data]
    return data

def save_json_data(data_dir, filename, data):
    os.makedirs(data_dir, exist_ok=True)
    file_name = os.path.join(data_dir, filename)
    with open(file_name, 'w') as output:
        if type(data) == list:
            if type(data[0]) in [str, list,dict]:
                for item in data:
                    output.write(json.dumps(item))
                    output.write('\n')

            else:
                json.dump(data, output)
        elif type(data) == dict:
            json.dump(data, output)
        else:
            raise RuntimeError('Unsupported type: %s' % type(data))
    print("saved dataset in " + file_name)

def percent_len(all_len,percentiles=None):
    if percentiles is None:
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ptiles_vers = list(np.percentile(all_len, np.array(percentiles)))
    ptiles_vers =[str(round(item,4)) for item in  ptiles_vers]
    tb = pt.PrettyTable()
    tb.field_names = ['mean'] + percentiles
    mean_value = round(np.mean(all_len), 1)
    tb.add_row([mean_value] + ptiles_vers)
    print(tb)
    latex_output = "& %.2f &"% float(mean_value)  + " &".join(ptiles_vers)
    print(latex_output)

def cal_r1_r5_r10(ranks):
    r1,r5,r10= 0,0,0
    data_len= len(ranks)
    for item in ranks:
        if item >=1:
            r1 +=1
            r5 += 1 
            r10 += 1
        elif item >=0.2:
            r5+= 1
            r10+=1
        elif item >=0.1:
            r10 +=1
    # print("& %.3f &%.3f &%.3f  "%(round(r1/data_len,4),  round(r5/data_len,4),   round(r10/data_len,4)))
    result = {"R@1":round(r1/data_len,3), "R@5": round(r5/data_len,3),  "R@10": round(r10/data_len,3)}
    return result

def time_format(time_cost):
    m, s = divmod(time_cost, 60)
    h, m = divmod(m, 60)
    # print("time_cost: %d" % (time_cost))
    return "%02d:%02d:%02d" % (h, m, s)

def array_split(original_data, core_num):
    data = []
    total_size = len(original_data)
    per_core_size = math.ceil(total_size / core_num)
    for i in range(core_num):
        lower_bound = i * per_core_size
        upper_bound = min((i + 1) * per_core_size, total_size)
        data.append(original_data[lower_bound:upper_bound])
    return data