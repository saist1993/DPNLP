import re
import ast
import traceback
import numpy as np
from os import walk
import matplotlib.pyplot as plt
from texttable import Texttable
from dataclasses import dataclass
from collections import defaultdict

pattern = re.compile(r"(?<=\d)\)")


def read_file(path:str):
    """reads and splits the log file in lines"""
    log_file = open(path, 'r')
    log_file = [line.replace('\n', '').strip() for line in log_file.readlines()]
    return log_file


def break_file_into_specific_run(file):
    """
    Returns a list of [start_index_of_run, end_index_of_run, uuid]
    to retrive one block: file[start_index_of_run:end_index_of_run+1]
    """
    # start of run - "INFO start of run"
    start_end_run_index = {}
    for index, line in enumerate(file):
        if "INFO start of run" in line:
            # find its UUID - unique id of the run.
            uuid = line.split(" ")[-1]
            start_end_run_index[uuid] = [index]
        if "INFO end of run" in line:
            # find its UUID - unique id of the run.
            uuid = line.split(" ")[-1]
            start_end_run_index[uuid].append(index)

    interm = [len(value) for key, value in start_end_run_index.items() if len(value) == 1]
    if interm == [1]:
        for key, value in start_end_run_index.items():
            if len(value) == 1:
                value.append(len(file) - 2)
                start_end_run_index[key] = value
    else:
        # remove all datapoints where the end is not well definied
        start_end_run_index = {key: value for key, value in start_end_run_index.items() if len(value) == 2}

    final_indexes = [[value[0], value[1], key] for key, value in start_end_run_index.items()]
    return final_indexes


@dataclass(unsafe_hash=True)
class DataPoint:
    '''
        data_point.arguments['emb_dim'] = 'ABC'
        data_point.eval_metric[0]['total_loss'] = 'PQR' -> valid
        data_point.eval_metric[1].['total_loss'] = 'ABC' -> test

        dp.arguments['emb_dim'], dp.eval_metric[0]['total_loss'], dp.eval_metric[1]['total_loss'], dp.attacker_data

    '''
    arguments: dict
    eval_metric: dict  # eval_metric[0] -> validation, eval_metric[1] -> test
    attacker_data: dict
    uuid: str


def split(sequence, sep):
    chunks = []
    x = 0
    chunks.append([])  # create an empty chunk to which we'd append in the loop
    for i in sequence:
        if sep not in i:
            chunks[x].append(i)
        else:
            x += 1
            chunks.append([])
    return [c for c in chunks if c]  # removes empty chunk


def is_nan(chunk):
    for c in chunk:
        if ' nan' in c:
            return True
    return False


def transform_chunk_to_datapoint(chunk, arguments, uuid):
    valid_dict = {}
    test_dict = {}
    attacker_dict = {}
    for c in chunk:
        c = re.sub(pattern, '', c.replace('tensor(', ''))
        c = c.replace("device='cuda:0'),", "")
        if 'valid dict:' in c:
            # validation logic
            try:
                valid_dict = {key.replace("valid_", ""): value for key, value in
                              ast.literal_eval(c.split("dict:")[1].strip()).items()}

            except:
                print(c)
                traceback.print_exc()
                raise IOError
        if 'test dict:' in c:
            # validation logic
            test_dict = {key.replace("test_", ""): value for key, value in
                         ast.literal_eval(c.split("dict:")[1].strip()).items()}

        if 'leakage dict:' in c:
            attacker_dict = ast.literal_eval(c.split("dict:")[1].strip())
    #         print(valid_dict)
    return DataPoint(arguments, [valid_dict, test_dict], attacker_dict, uuid)


def generate_data_from_run(run, uuid):
    # arguments are in block[1]
    arguments = ast.literal_eval(run[1].split("arguemnts:")[1].strip())
    chunks = split(run[2:], sep='INFO start of block')
    chunks = [transform_chunk_to_datapoint(c, arguments, uuid) for c in chunks if not is_nan(c)]
    return chunks

# chunks = generate_data_from_run(files[0][indexes[0][0]: indexes[0][1]+1], "123-123")

def get_final_clean_data(data_paths):
    final_data = []
    files = [read_file(path) for path in data_paths]
#     print(files)
    for file in files:
        try:
            indexes = break_file_into_specific_run(file)
            for start, end, uuid in indexes:
                chunks = generate_data_from_run(file[start:end+1], uuid)
                for c in chunks:
                    final_data.append(c)
        except:
            traceback.print_exc()
            continue
    return final_data

def get_indexes_to_pop(data):
    indexes_to_pop = []
    for index, d in enumerate(data):
        try:
            _ = d.eval_metric[1]['epoch_acc_main']
        except:
            indexes_to_pop.append(index)
            continue
    return indexes_to_pop



def filter_data_epoch(data, epoch):
    temp = []
    for node in data:
        if node.arguments['epochs'] == epoch:
            temp.append(node)
    return temp

def filter_data_seed(data, epoch):
    temp = []
    for node in data:
        if node.arguments['seed'] == epoch:
            temp.append(node)
    return temp

def filter_noise_adv(data, noise=True, adv=True):
    temp = []
    for node in data:
        if node.arguments['noise_layer'] == noise and node.arguments['is_adv'] == adv:
            temp.append(node)
    return temp

def filter_adv(data, noise=True):
    temp = []
    for node in data:
        if node.arguments['noise_layer'] == noise:
            temp.append(node)
    return temp


def filter_adv(data, seed=1234):
    temp = []
    for node in data:
        if node.arguments['seed'] == seed:
            temp.append(node)
    return temp