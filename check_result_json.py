import os
import argparse
import json

def calcaulte_avg(target_dict):
    if len(target_dict) == 0:
        return None

    result_avg_dict = dict()
    for metric_name in target_dict[list(target_dict.keys())[0]].keys():
        result_avg_dict[f'val/{metric_name}'] = [output[metric_name] for output in target_dict.values()]
    
    avg_result = {k: sum(v) / len(v) for k, v in result_avg_dict.items()}
    return avg_result


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def main(args):
    json_data = read_json(args.json_path)

    underthreshold_collector = dict()
    overthreshold_collector = dict()
    for instance, instance_metrics_dict in json_data.items():
        if instance_metrics_dict[args.metric_name] < args.threshold:
            underthreshold_collector[instance] = instance_metrics_dict
        else:
            overthreshold_collector[instance] = instance_metrics_dict
    
    print(f"underthreshold_collector: {len(underthreshold_collector)}")
    print(f"overthreshold_collector: {len(overthreshold_collector)}")

    underthreshold_avg = calcaulte_avg(underthreshold_collector)
    overthreshold_avg = calcaulte_avg(overthreshold_collector)
    print(f"underthreshold_avg: {underthreshold_avg}")
    print(f"overthreshold_avg: {overthreshold_avg}")


if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Check Json')

    # Dataset arguments
    parser.add_argument('--json_path', type=str, default='checkpoint_test_backup/TOP128NP5000_LOAD_G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6/models/test_results.json')
    parser.add_argument('--metric_name', type=str, default='crd', choices=['crd', 'cd', 'rrmse_geo', 'trmse_geo'])
    parser.add_argument('--threshold', type=float, default=0.01)
    args = parser.parse_args()
    print(f"args: {args}")
    main(args)



"""
python check_result_json.py --metric_name rrmse_geo --threshold 16
"""