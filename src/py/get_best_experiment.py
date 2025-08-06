import json
import argparse

def main(args):

  with open(args.json, "r") as f:
    experiment_state = json.load(f)

  trials = experiment_state["trial_data"]

  best_trial = None
  best_metric = float("-inf")
  i = 0
  for trial in trials:
    print(f"trial number {i}")
    results = trial[1]
    i+=1

    dict_results = json.loads(results)
    last_results = dict_results['last_result']

    if 'val_acc' in last_results.keys():

      if last_results['val_acc'] > best_metric:
        best_metric = dict_results['last_result']['val_acc']
        best_trial = trial

  configs = best_trial[0]
  dict_configs = json.loads(configs)
  print("Best trial id :", dict_configs["trial_id"])
  print("Best trial final metric:", best_metric)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='get best experiment fron ray tune.')
  parser.add_argument('--json', type=str, help='experiment state, json file')
  
  args = parser.parse_args()

  main(args) 