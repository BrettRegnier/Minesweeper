from torchdrl.agents.markov.DoubleDQL import DoubleDQL
from torchdrl.agents.markov.RainbowDQL import RainbowDQL

class Broom(DoubleDQL):
    def __init__(self, env, **kwargs):
        super(Broom, self).__init__(env, **kwargs)

        if self._plotter:
            self._plotter.AddFigure("Progress", "Mean Score", "blue")
            self._plotter.AddFigure("Progress", "Loss", "purple")

    def Act(self, state):
        state_t = self.ConvertStateToTensor(state)

        q_values = self._net(state_t)


        # # find which tiles are already revealed.
        # unrevealed = 9
        # rows = 1
        # for r in range(len(state[unrevealed])):
        #     for c in range(len(state[unrevealed][r])):
        #         if not state[unrevealed][r][c]:
        #             idx = r * state.shape[rows] + c
        #             q_values[0][idx] = -float("inf")

        # action = q_values.argmax().item()

        return action
