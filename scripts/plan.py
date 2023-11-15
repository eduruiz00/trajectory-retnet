import json
import pdb
from os.path import join
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import trajectory.utils as utils
from evaluation import evaluate

class Parser(utils.Parser):
    dataset: str = 'bullet-halfcheetah-medium-v0'
    model: str = 'retnet'
    config: str = 'config.offline'

#######################
######## setup ########
#######################

args = Parser().parse_args('plan')

#######################
####### models ########
#######################

loadpath = args.gpt_loadpath if args.model == "gpt" else args.retnet_loadpath
args.exp_name = args.gpt_exp_name if args.model == "gpt" else args.retnet_exp_name

dataset = utils.load_from_config(args.logbase, args.dataset, loadpath,
        'data_config.pkl')

model, model_epoch = utils.load_model(args.logbase, args.dataset, loadpath,
        epoch=args.model_epoch, device=args.device)

if args.model == "retnet":
    model.chunkwise_recurrent = False

time_str = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
writer = SummaryWriter(log_dir=f"runs/plan/{args.model}_{args.dataset}_{time_str}")

score, t, total_reward, terminal = evaluate(model, dataset, writer, args, render=True)

writer.close()

## save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal, 'model_epoch': model_epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
