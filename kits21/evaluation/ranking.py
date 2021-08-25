from typing import Callable
import numpy as np
import scipy.stats as ss


def rank_then_aggregate(data: np.ndarray, aggr_fn: Callable = np.mean):
    """
    data must be (algos x metrics). Higher values must mean better result (so Dice is OK, but not HD95!).
    If you want to use this code with HD95 you need to invert it as a preprocessing step
    :param data:
    :param aggr_fn:
    :return:
    """
    ranked = np.apply_along_axis(ss.rankdata, 0, -data, 'min')
    aggregated = aggr_fn(ranked, axis=-1)
    final_rank = ss.rankdata(aggregated, 'min')
    return final_rank, aggregated


def rank_participants(summary_csv: str, output_csv_file: str) -> None:
    results = np.loadtxt(summary_csv, dtype=str, delimiter=',', skiprows=1)
    teams = results[:, 0]

    assert len(np.unique(teams)) == len(teams), 'Some teams have identical names, please fix'
    metrics = results[:, 1:].astype(float)

    assert metrics.shape[1] == 6, 'expected 6 metrics, got %d' % metrics.shape[1]

    mean_dice = np.mean(metrics[:, :3], axis=1, keepdims=True)
    mean_sd = np.mean(metrics[:, 3:], axis=1, keepdims=True)

    mean_metrics = np.concatenate((mean_dice, mean_sd), axis=1)
    ranks, aggregated = rank_then_aggregate(mean_metrics)

    # now clean up ties. This is not the cleanest implementation, but eh
    rank = 1
    while rank < (len(teams) + 1):
        num_teams_on_that_rank = sum(ranks == rank)
        if num_teams_on_that_rank <= 1:
            rank += 1
            continue
        else:
            # tumor dice is tie breaker
            teams_mask = ranks == rank
            tumor_dice_scores = metrics[teams_mask, 2:3]
            new_ranks = rank_then_aggregate(tumor_dice_scores)[0] - 1 + rank
            if len(np.unique(new_ranks)) == 1:
                print("WARNING: Cannot untie ranks of these teams... tumor_dice_scores and ranks are identical:")
                print('team names:', teams[teams_mask])
                rank += 1
            if len(np.unique(new_ranks)) != sum(teams_mask):
                ranks[teams_mask] = new_ranks
                continue
            ranks[teams_mask] = new_ranks
            rank += 1

    # print ranking
    sorting = np.argsort(ranks)
    with open(output_csv_file, 'w') as f:
        f.write('team_name,final_rank,mean_rank,mean_dice,mean_sd,tumor_dice\n')
        for i in sorting:
            f.write('%s,%d,%.4f,%.8f,%.8f,%.8f\n' % (teams[i], ranks[i], aggregated[i], mean_dice[i], mean_sd[i],
                                                     metrics[i, 2]))


if __name__ == '__main__':
    summary_file = 'nnUNet_summary.csv'  # follow the example csv provided in this folder!
    output_file = 'kits2021_ranking.csv'
    rank_participants(summary_file, output_file)
