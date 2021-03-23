import itertools
import numpy as np
import yaml
from scipy import stats


def decode_arch_str(arch_str):
    channels = ['8', '16', '24', '32', '40', '48', '56', '64']
    channels_str = arch_str.split(':')
    arch = []
    for channel_str in channels_str:
        arch.append(channels.index(channel_str))

    out = [arch[0:2], arch[2:4], [arch[4]]]
    return out


def get_arch_acc():
    with open('./nats_cifar10_acc_rank.yml', 'r') as f:
        ranks = yaml.load(f)
    arch_strs = ranks
    arch_accs = [ranks[i] for i in arch_strs]
    archs = [decode_arch_str(i) for i in arch_strs]
    return archs, arch_accs


def extract_score_list(archs):
    num_ops = 8
    model_encod_list = archs
    potential_yaml = [
        ['./path_rank/path_rank_0.yml'],
        ['./path_rank/path_rank_1.yml'],
        ['./path_rank/path_rank_2.yml'],
    ]
    loss_list = []
    for model_encod in model_encod_list:
        total_loss = 0

        for stage, stage_encod in enumerate(model_encod):
            with open(potential_yaml[stage][0], 'r') as f:
                # potential = str_to_dict(f.read())
                potential = dict(yaml.load(f))

            stage_model_pool = list(
                itertools.product(list(range(num_ops)), repeat=len(stage_encod)))
            assert len(stage_model_pool) == len(potential), \
                'length mismatch in stage {}. model pool {},  potential {}'.format(
                    stage, len(stage_model_pool), len(potential))
            stage_encod = ''.join([str(code) for code in stage_encod])
            loss = potential[stage_encod] + 4.
            total_loss += loss
        loss_list.append(round(total_loss, 4))
    # print('\nloss:\n', loss_list)
    return loss_list


def str_to_dict(a: str):
    return dict([(b.split(', ')[0].strip('\''), float(b.split(', ')[1]))
                 for b in a.strip('[]()').split('), (')])


if __name__ == '__main__':
    model, acc = get_arch_acc()
    dna_loss = extract_score_list(model)
    TrueAcc = np.array(acc)
    SSDNA = np.array(dna_loss)
    ssdnatau = stats.kendalltau(TrueAcc, SSDNA)
    pearsonr = stats.pearsonr(TrueAcc, SSDNA)
    spearmanr = stats.spearmanr(TrueAcc, SSDNA)
    print("BossNAS: {}\n{}\n{}\n".format(ssdnatau, pearsonr, spearmanr))
