import collections
import numpy as np

class BroomManager:
    def __init__(self, env, agent):
        self._agent = agent

        self._win = 0
        self._lose = 0
        self._win_loss = collections.deque(maxlen=100)
    
    def TrainNoYield(self):
        self._agent.TrainNoYield()

    def Train(self):
        for episode_info in self._agent.Train():
            self.PrintEpisodeInfo(episode_info)

    def Evaluate(self):
        pass

    def PrintEpisodeInfo(self, episode_info):
        name = episode_info['agent_name']
        ep = episode_info['episode']
        ep_steps = episode_info['steps']
        ep_score = episode_info['episode_score']
        mean_score = episode_info['mean_score']
        ep_loss = round(episode_info['loss'], 2)
        total_steps = episode_info['total_steps']

        avg_steps = round(total_steps / ep, 0)

        ep_msg = name
        ep_msg += ' episode: {:<4}'.format(ep)
        ep_msg += ' loss: {:<6}'.format(ep_loss)
        ep_msg += ' steps: {:<3}'.format(ep_steps)
        ep_msg += ' avg steps: {:<3}'.format(avg_steps)
        ep_msg += ' episode score: {:<6}'.format(ep_score)
        ep_msg += ' mean score: {:<6}'.format(mean_score)
        ep_msg += ' total steps: {:<6}'.format(total_steps)

        if 'epsilon' in episode_info:
            ep_msg += ' epsilon: {:<3}'.format(episode_info['epsilon'])

        if 'win' in episode_info:
            if episode_info['win']:
                self._win += 1
                self._win_loss.append(1)
            else:
                self._lose += 1
                self._win_loss.append(0)
            
            ep_msg += ' win: {:<5}'.format(self._win)
            ep_msg += ' lose: {:<5}'.format(self._lose)
            ep_msg += ' win-lose: {:<3}%'.format(np.sum(self._win_loss))

        print(ep_msg)