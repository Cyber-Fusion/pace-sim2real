from __future__ import annotations

import cmaes
import torch
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
from datetime import datetime
import os


class CMAESOptimizer:
    def __init__(self, bounds, population_size, log_dir, joint_names, max_iteration, data, device, epsilon=None, sigma=0.5, save_interval=10):

        self.joint_names = joint_names
        self.max_iteration = max_iteration
        self.epsilon = epsilon
        self.save_interval = save_interval
        self.device = device

        # create log_dir in YY_MM_DD_hh-mm-ss format
        folder_time = datetime.now().strftime("%y_%m_%d_%H-%M-%S")
        # create string with time and date
        log_dir = os.path.join(log_dir, folder_time)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = TensorboardSummaryWriter(log_dir=log_dir)
        torch.save({"bounds": bounds,
                    "joint_names": joint_names,
                    "dof_pos": data["dof_pos"],
                    "des_dof_pos": data["des_dof_pos"]}, log_dir + "/config.pt")

        self.bounds = bounds

        bounds_normalized = torch.ones_like(bounds)
        bounds_normalized[:, 0] *= -1
        mean_normalized = torch.zeros_like(bounds[:, 0])

        self.optimizer = cmaes.CMA(mean=mean_normalized.cpu().numpy(), sigma=sigma, bounds=bounds_normalized.cpu().numpy(), seed=0, population_size=population_size)

        self.scores_counter = 0
        self.iteration_counter = 0

        self.scores = torch.zeros(population_size, device=device)
        self.scores_buffer = torch.zeros((max_iteration, population_size), device=device)
        self.sim_dof_pos_buffer = torch.zeros((population_size, data["dof_pos"].shape[0], len(joint_names)), device=device)  # TODO this is not really a nice solution

        self.params = torch.zeros((population_size, bounds.shape[0]), device=device)
        self.sim_params = torch.zeros_like(self.params)
        self.sim_params_buffer = torch.zeros((max_iteration, population_size, bounds.shape[0]), device=device)
        self.armature_idx = slice(0, 12)
        self.damping_idx = slice(12, 24)
        self.friction_idx = slice(24, 36)
        self.bias_idx = slice(36, 48)
        self.delay_idx = 48

        self._reset_population()
        print("CMA-ES optimizer initialized.")
        print("Current iteration: ", self.iteration_counter)

    def ask(self):
        return self.optimizer.ask()

    def tell(self, sim_dof_pos, real_dof_pos):
        self.scores += torch.sum(torch.square(sim_dof_pos - real_dof_pos), dim=1)
        self.sim_dof_pos_buffer[:, self.scores_counter, :] = sim_dof_pos
        self.scores_counter += 1

    def evolve(self):
        self.scores /= self.scores_counter
        self.scores_buffer[self.iteration_counter, :] = self.scores
        solutions = []
        for i in range(self.optimizer.population_size):
            solutions.append((self.params[i].cpu().numpy(), self.scores[i].item()))
        self.optimizer.tell(solutions)
        if self.save_interval > 0 and self.iteration_counter % self.save_interval == 0:
            self.save_checkpoint(torch.tensor(self.optimizer._mean), self.sim_params_buffer, self.scores_buffer, self.iteration_counter)
        self._print_iteration()

        self._reset_population()

        # TODO check if epsilon reached and stop optimization
        if self.epsilon is not None:
            pass

        self.scores = torch.zeros_like(self.scores)
        self.scores_counter = 0
        self.iteration_counter += 1
        print("CMA-ES optimizer iteration: ", self.iteration_counter)

    def finished(self):
        finished = self.max_iteration <= self.iteration_counter
        diff_score = (self.scores_buffer[self.iteration_counter - 1, :].max() - self.scores_buffer[self.iteration_counter - 1, :].min()) / self.scores_buffer[self.iteration_counter - 1, :].min()
        finished = finished or (self.epsilon is not None and diff_score < self.epsilon)
        if finished:
            print("CMA-ES optimization finished.")
            self.save_checkpoint(torch.tensor(self.optimizer._mean), self.sim_params_buffer, self.scores_buffer, self.iteration_counter - 1)
        return finished

    def _reset_population(self):
        for i in range(self.optimizer.population_size):
            self.params[i, :] = torch.tensor(self.optimizer.ask(), device=self.device)
        self.sim_params = self._params_to_sim_params(self.params)

    def update_simulator(self, articulation, joint_ids, initial_position):
        env_ids = torch.arange(len(self.sim_params[:, self.armature_idx]))
        articulation.write_joint_armature_to_sim(self.sim_params[:, self.armature_idx], joint_ids, env_ids=env_ids)
        articulation.data.default_joint_armature[:, joint_ids] = self.sim_params[:, self.armature_idx]
        articulation.write_joint_viscous_friction_coefficient_to_sim(self.sim_params[:, self.damping_idx], joint_ids, env_ids=env_ids)
        articulation.data.default_joint_viscous_friction_coeff[:, joint_ids] = self.sim_params[:, self.damping_idx]
        articulation.write_joint_friction_coefficient_to_sim(self.sim_params[:, self.friction_idx], joint_ids, env_ids=env_ids)
        articulation.data.default_joint_friction_coeff[:, joint_ids] = self.sim_params[:, self.friction_idx]
        # articulation.write_joint_dynamic_friction_coefficient_to_sim(self.sim_params[:, self.friction_idx], joint_ids, env_ids=env_ids)
        # articulation.data.default_joint_dynamic_friction_coeff[:, joint_ids] = self.sim_params[:, self.friction_idx]
        articulation.write_joint_position_to_sim(initial_position, joint_ids)
        articulation.write_joint_velocity_to_sim(torch.zeros_like(initial_position), joint_ids)
        # TODO add joint bias and delay

    def _print_iteration(self):
        min_score = torch.min(self.scores)
        max_score = torch.max(self.scores)
        min_index = torch.argmin(self.scores)
        print("Max score: ", max_score.item())
        print("Min score: ", min_score.item(), " at index: ", min_index.item())
        print("Armature: ", self.sim_params[min_index, self.armature_idx].tolist())
        print("Damping: ", self.sim_params[min_index, self.damping_idx].tolist())
        print("Friction: ", self.sim_params[min_index, self.friction_idx].tolist())
        print("Bias: ", self.sim_params[min_index, self.bias_idx].tolist())
        print("Delay: ", self.sim_params[min_index, self.delay_idx].tolist())
        self._log()

    def _params_to_sim_params(self, params):
        sim_params = (params + 1.0) / 2.0  # change range from 0 to 1
        sim_params = self.bounds[:, 0] + sim_params * (self.bounds[:, 1] - self.bounds[:, 0])  # range from lower to upper bound
        return sim_params

    def get_best_sim_params(self):
        best_params = torch.tensor(self.optimizer._mean)
        return self._params_to_sim_params(best_params)

    def _log(self):
        min_score, min_score_index = torch.min(self.scores, dim=0)
        max_score, _ = torch.max(self.scores, dim=0)
        for i in range(len(self.joint_names)):
            self.writer.add_histogram("4_Bias/distribution_" + self.joint_names[i], self.sim_params[:, self.bias_idx][:, i], self.iteration_counter)
            self.writer.add_histogram("3_Friction/distribution_" + self.joint_names[i], self.sim_params[:, self.friction_idx][:, i], self.iteration_counter)
            self.writer.add_histogram("2_Damping/distribution_" + self.joint_names[i], self.sim_params[:, self.damping_idx][:, i], self.iteration_counter)
            self.writer.add_histogram("1_Armature/distribution_" + self.joint_names[i], self.sim_params[:, self.armature_idx][:, i], self.iteration_counter)

            self.writer.add_scalar("4_Bias/best_" + self.joint_names[i], self.sim_params[min_score_index, self.bias_idx][i].item(), self.iteration_counter)
            self.writer.add_scalar("3_Friction/best_" + self.joint_names[i], self.sim_params[min_score_index, self.friction_idx][i].item(), self.iteration_counter)
            self.writer.add_scalar("2_Damping/best_" + self.joint_names[i], self.sim_params[min_score_index, self.damping_idx][i].item(), self.iteration_counter)
            self.writer.add_scalar("1_Armature/best_" + self.joint_names[i], self.sim_params[min_score_index, self.armature_idx][i].item(), self.iteration_counter)
        self.writer.add_histogram("0_Delay/distribution", self.sim_params[:, self.delay_idx], self.iteration_counter)
        self.writer.add_scalar("0_Delay/best", self.sim_params[min_score_index, self.delay_idx].item(), self.iteration_counter)

        self.writer.add_scalar("0_Episode/score", min_score.item(), self.iteration_counter)
        self.writer.add_scalar("0_Episode/max_score", max_score.item(), self.iteration_counter)
        self.writer.add_scalar("0_Episode/diff_score", (max_score - min_score) / min_score, self.iteration_counter)

    def save_checkpoint(self, mean, params_buffer, scores_buffer, iteration):
        min_index = torch.argmin(self.scores_buffer[iteration, :])
        torch.save({"mean": mean,
                    "params_buffer": params_buffer,
                    "scores_buffer": scores_buffer,
                    "best_trajectory": self.sim_dof_pos_buffer[min_index, :]},
                   os.path.join(self.writer.log_dir, "params_" + f"{iteration:03}" + ".pt"))

    def close(self):
        self.writer.close()
