import numpy as np
from utils.graph_utils import draw_scatterplot
import matplotlib.pyplot as plt
from scipy.stats import beta


true_rewards = {
    0: 1 / 6,
    1: 1 / 2,
    2: 2 / 3,
    3: 3 / 4,
    4: 5 / 6
}


def get_arm_to_pull(beta_distributions):
    best_arm = None
    best_delta = 0
    for arm, beta_distribution in beta_distributions.items():
        delta_arm = np.random.beta(
            beta_distribution['alpha'],
            beta_distribution['beta']
        )
        if delta_arm > best_delta:
            best_arm = arm
            best_delta = delta_arm
    return best_arm


def pull_arm(arm):
    return int(np.random.sample() < true_rewards[arm])


def perform_thompson_sampling(T):
    beta_distributions = {
        arm: {'beta': 1, 'alpha': 1} for arm in range(len(true_rewards))
    }
    reward = 0
    average_rewards = []
    arm_pulls = []
    for i in range(1, T + 1):
        arm_to_pull = get_arm_to_pull(beta_distributions)
        arm_pulls.append(arm_to_pull)
        result = pull_arm(arm_to_pull)
        if result == 1:
            beta_distributions[arm_to_pull]['alpha'] += 1
            reward += 1
        else:
            beta_distributions[arm_to_pull]['beta'] += 1
        average_rewards.append(reward / i)
    return average_rewards, beta_distributions, arm_pulls


def part_3():
    average_rewards, _, _ = perform_thompson_sampling(300)
    draw_scatterplot(
        list(range(1, 301)),
        average_rewards,
        xlabel='t',
        ylabel='Average Reward',
        title='Average Reward vs t',
        save_location='/Users/stewart/Desktop/hw2_problem2_pt_3.png')


def draw_confidence_interval_plot(t):
    _, beta_distributions, _ = perform_thompson_sampling(t)
    xlabels = ['Arm {}'.format(i) for i in range(1, len(true_rewards) + 1)]
    mean_values = []
    yerr_lower = []
    yerr_upper = []
    true_reward_means = []
    for i in range((len(true_rewards))):
        beta_dist = beta_distributions[i]
        mu_hat = beta_dist['alpha'] / (beta_dist['alpha'] + beta_dist['beta'])
        mean_values.append(mu_hat)
        ucb = beta.ppf(.975, beta_dist['alpha'], beta_dist['beta']) - mu_hat
        lcb = mu_hat - beta.ppf(.025, beta_dist['alpha'], beta_dist['beta'])
        yerr_upper.append(ucb)
        yerr_lower.append(lcb)
        true_reward_means.append(true_rewards[i])
    plt.errorbar(
        xlabels,
        mean_values,
        yerr=[yerr_lower, yerr_upper],
        fmt='o',
        label='Posterior Distribution')
    plt.scatter(
        xlabels,
        true_reward_means,
        label="True Mean",
        color='red'
        )
    plt.legend(loc='best')
    plt.ylabel('Posterior Distribution')
    plt.title('Posterior Distribution After {} Pulls'.format(t))
    plt.savefig('/Users/stewart/Desktop/hw2_problem2_pt_4_{}.png'.format(t))
    plt.clf()


def part_4():
    draw_confidence_interval_plot(10)
    draw_confidence_interval_plot(100)
    draw_confidence_interval_plot(300)
    draw_confidence_interval_plot(2000)


def get_proportion_vs_t(arm_pulls, arm_idx):
    return [
        (arm_pulls[:t] == arm_idx).sum() / t
        for t in range(1, len(arm_pulls) + 1)
    ]


def part_5():
    _, _, arm_pulls = perform_thompson_sampling(1000)
    arm_pulls = np.array(arm_pulls)
    x_data = list(range(1, len(arm_pulls) + 1))
    for arm in range(len(true_rewards)):
        plt.plot(
            x_data,
            get_proportion_vs_t(arm_pulls, arm),
            label='Arm {}'.format(arm + 1)
        )
    plt.xlabel('t')
    plt.ylabel('Fraction of time arm is pulled')
    plt.title('Fraction of time each arm is pulled vs t')
    plt.legend(loc="best")
    plt.savefig('/Users/stewart/Desktop/hw2_problem2_pt_5.png')
    plt.clf()


def part_6():
    _, _, arm_pulls = perform_thompson_sampling(10000)
    arm_pulls = np.array(arm_pulls)
    proportions = np.array(get_proportion_vs_t(arm_pulls, 4))
    for idx, proportion in enumerate(proportions):
        if proportion > 0.95 and idx > 10:
            last_ten = proportions[idx - 10: idx]
            if (last_ten > 0.95).sum() == 10:
                print(idx - 10)
                break


if __name__ == '__main__':
    # part_3()
    # part_4()
    # part_5()
    part_6()
