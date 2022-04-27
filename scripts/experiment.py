#!/usr/bin/env python
import gym 
import safety_gym
import safe_rl
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork
from safe_rl.utils.threshold_sche import ThresholdScheduler


def main(robot, task, algo, seed, num_steps, epoch_per_threshold, cpu):
    # Verify experiment
    robot_list = ['point', 'car', 'doggo']
    task_list = ['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']
    algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']

    algo = algo.lower()
    task = task.capitalize()
    robot = robot.capitalize()
    assert algo in algo_list, "Invalid algo"
    assert task.lower() in task_list, "Invalid task"
    assert robot.lower() in robot_list, "Invalid robot"

    # Hyperparameters
    exp_name = algo + '_' + robot + task
    if robot=='Doggo':
        num_steps = 1e8
        steps_per_epoch = 60000
    else:
        # num_steps = 1e7
        steps_per_epoch = 30000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01
    thre_sche = ThresholdScheduler(epoch_per_threshold)

    # Fork for parallelizing
    mpi_fork(cpu)

    # Prepare Logger
    exp_name = exp_name or (algo + '_' + robot.lower() + task.lower())
    logger_kwargs = setup_logger_kwargs(exp_name, seed)

    # Algo and Env
    algo = eval('safe_rl.'+algo)
    env_name = 'Safexp-'+robot+task+'-v0'

    algo(env_fn=lambda: gym.make(env_name),
         ac_kwargs=dict(
             hidden_sizes=(256, 256),
            ),
         epochs=epochs,
         steps_per_epoch=steps_per_epoch,
         save_freq=save_freq,
         target_kl=target_kl,
         cost_lim=25,  # useless
         seed=seed,
         logger_kwargs=logger_kwargs,
         thre_sche=thre_sche
        )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='Point')
    parser.add_argument('--task', type=str, default='Goal1')
    parser.add_argument('--algo', type=str, default='cpo')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--num_steps', type=float, default=3.3e7)
    parser.add_argument('--epoch_per_threshold', type=int, default=20)
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()
    # 当只训练1个threshold时，我们的方法就退化成了RCPO
    # 当训练N个threshold，epoch_per_threshold=num_steps/3e4时，我们的方法相当于每个task按顺序训练
    main(args.robot, args.task, args.algo, args.seed,
         args.num_steps, args.epoch_per_threshold, args.cpu)
