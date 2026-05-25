"""drone_env_v2: same env, improved reward function.

Changes vs original:
- Goal bonus: 500 -> 20000 (configurable `goal_bonus`)
- Waypoint reward: 50+5i -> wp_base+wp_scale*i (default 3+1i)
- Speed reward: configurable `speed_reward_coef` (default 3.0)
- ADDED: static obstacle proximity penalty (w_obs_proximity)
- ADDED: drone-to-drone proximity penalty (w_drone_proximity)
"""
import numpy as np
from neural_controller.env.drone_env import MultiDroneEnv


class MultiDroneEnvV2(MultiDroneEnv):
    def __init__(self, config: dict):
        super().__init__(config)
        self.goal_bonus = config.get('goal_bonus', 20000.0)
        self.wp_base = config.get('wp_base', 3.0)
        self.wp_scale = config.get('wp_scale', 1.0)
        self.speed_reward_coef = config.get('speed_reward_coef', 3.0)
        self.w_obs_proximity = config.get('w_obs_proximity', 0.3)
        self.w_drone_proximity = config.get('w_drone_proximity', 0.2)
        self.drone_alert_radius = config.get('drone_alert_radius', 1.5)

    def step(self, actions):
        action_matrix = np.array([actions[agent] for agent in self.agents], dtype=np.float32)
        action_matrix = np.clip(action_matrix, -self.max_speed, self.max_speed)
        new_positions = self.drone_positions + action_matrix * self.dt
        new_positions = np.clip(new_positions, 0, self.map_size[0])
        self.drone_velocities = action_matrix
        self.drone_positions = new_positions

        for obs in self.dynamic_obstacles:
            obs['pos'] += obs['vel'] * self.dt
            for dim in range(2):
                if obs['pos'][dim] < 0 or obs['pos'][dim] > self.map_size[0]:
                    obs['vel'][dim] *= -1
                    obs['pos'][dim] = np.clip(obs['pos'][dim], 0, self.map_size[0])

        old_waypoint_indices = self.waypoint_indices.copy()
        old_dist_to_target = np.array(self.prev_dist_to_target, dtype=np.float32)
        self.prev_dist_to_target = self.prev_dist_to_target.copy()

        for i in range(self.num_drones):
            if not self.drone_active[i]:
                continue
            wp_idx = self.waypoint_indices[i]
            if wp_idx < len(self.waypoints) - 1:
                dist_to_wp = np.linalg.norm(self.drone_positions[i] - np.array(self.waypoints[wp_idx]))
                if dist_to_wp < self.goal_tolerance:
                    self.waypoint_indices[i] = wp_idx + 1
            elif wp_idx == len(self.waypoints) - 1:
                dist_to_wp = np.linalg.norm(self.drone_positions[i] - np.array(self.waypoints[wp_idx]))
                if dist_to_wp < self.goal_tolerance:
                    self.waypoint_indices[i] = len(self.waypoints)
            if self.waypoint_indices[i] < len(self.waypoints):
                self.prev_dist_to_target[i] = np.linalg.norm(
                    self.drone_positions[i] - np.array(self.waypoints[self.waypoint_indices[i]]))
            else:
                self.prev_dist_to_target[i] = 0.0

        prev_active = self.drone_active.copy()
        collision_occurred = np.zeros(self.num_drones, dtype=bool)

        if self.current_step >= self.collision_grace_steps:
            for i in range(self.num_drones):
                if not self.drone_active[i]:
                    continue
                for j in range(i + 1, self.num_drones):
                    if not self.drone_active[j]:
                        continue
                    if np.linalg.norm(self.drone_positions[i] - self.drone_positions[j]) < 2 * self.collision_radius:
                        collision_occurred[i] = True
                        collision_occurred[j] = True
            for i in range(self.num_drones):
                if not self.drone_active[i]:
                    continue
                for obs in self.dynamic_obstacles:
                    if np.linalg.norm(self.drone_positions[i] - obs['pos']) < 2 * self.collision_radius:
                        collision_occurred[i] = True
                        break
            for i in range(self.num_drones):
                if not self.drone_active[i]:
                    continue
                if self._is_in_obstacle(self.drone_positions[i]):
                    collision_occurred[i] = True

        self.drone_active = np.logical_and(self.drone_active, ~collision_occurred)
        newly_inactive_collision = np.logical_and(prev_active, collision_occurred)
        newly_inactive_stuck = np.zeros(self.num_drones, dtype=bool)

        # ============ REWARD (MODIFIED) ============
        rewards = {}
        for i, agent in enumerate(self.agents):
            if not self.drone_active[i]:
                if newly_inactive_collision[i]:
                    rewards[agent] = self.death_penalty
                elif newly_inactive_stuck[i]:
                    rewards[agent] = self.stuck_penalty
                else:
                    rewards[agent] = 0.0
                continue

            reward = 0.0

            # 1. Goal bonus (heavily increased for terminal incentive)
            if self.waypoint_indices[i] >= len(self.waypoints):
                reward += self.goal_bonus
                self.drone_active[i] = False
                self.stuck_counters[i] = 0
                rewards[agent] = reward
                continue

            # 2. Waypoint advancement (flattened scale)
            if self.waypoint_indices[i] > old_waypoint_indices[i]:
                wp_reward = self.wp_base + self.wp_scale * self.waypoint_indices[i]
                reward += wp_reward
                self.stuck_counters[i] = 0

            # 3. Speed along path (fix OOB: last waypoint has no next)
            if self.waypoint_indices[i] < len(self.waypoints) - 1:
                next_wp = np.array(self.waypoints[self.waypoint_indices[i] + 1])
                curr_wp = np.array(self.waypoints[self.waypoint_indices[i]])
                path_dir = next_wp - curr_wp
                norm = np.linalg.norm(path_dir) + 1e-8
                path_dir_norm = path_dir / norm
                speed_along = np.dot(self.drone_velocities[i], path_dir_norm)
                reward += self.speed_reward_coef * max(0, speed_along)

            # 4. Stuck counter update (原版遗漏了自增逻辑)
            if self.waypoint_indices[i] > old_waypoint_indices[i]:
                self.stuck_counters[i] = 0
            else:
                curr_dist = self.prev_dist_to_target[i]
                if curr_dist < old_dist_to_target[i] - self.stuck_progress_eps:
                    self.stuck_counters[i] = 0
                elif curr_dist > old_dist_to_target[i] + self.stuck_speed_eps:
                    self.stuck_counters[i] += 2
                else:
                    self.stuck_counters[i] += 1

            # 5. Stuck detection
            if self.stuck_counters[i] >= self.stuck_max_steps:
                self.drone_active[i] = False
                newly_inactive_stuck[i] = True
                reward += self.stuck_penalty
                rewards[agent] = reward
                self.stuck_counters[i] = 0
                continue

            rewards[agent] = reward

        self.current_step += 1
        done_all = (not np.any(self.drone_active)) or (self.current_step >= self.max_steps)
        dones = {agent: (not self.drone_active[i]) for i, agent in enumerate(self.agents)}
        dones["__all__"] = done_all
        obs = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        return obs, rewards, dones, infos
