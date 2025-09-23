import pickle
import random
from typing import Any, Literal

import numpy as np
import gymnasium as gym
import pygame
from gymnasium import spaces
from gymnasium.core import ObsType, ActType, RenderFrame
from gymnasium.wrappers import TimeLimit

from ltl.logic import FrozenAssignment, Assignment


class LetterEnv(gym.Env):
    """
    This environment is a grid with randomly located letters on it.
    We ensure that there is a clean path to any of the letters (a path that includes no passing by any letter).
    Note that stepping outside the map causes the agent to appear on the other side.
    """
    metadata = {'render_modes': ['human', 'rgb_array', 'path'], 'render_fps': 10}

    def __init__(
            self,
            grid_size: int,
            letters: str,
            use_fixed_map: bool,
            use_agent_centric_view: bool,
            render_mode: str | None = None,
            map: dict[tuple[int, int], str] = None
    ):
        if use_agent_centric_view and grid_size % 2 == 0:
            raise ValueError("Agent-centric view is only available for odd grid-sizes")
        self.render_mode = render_mode
        self.grid_size = grid_size
        self.letters = letters
        self.use_fixed_map = use_fixed_map
        self.use_agent_centric_view = use_agent_centric_view
        self.letter_types = sorted(list(set(letters)))
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size, len(self.letter_types) + 1),
                                            dtype=np.uint8)
        self.map = map
        self.agent = (0, 0)
        self.locations = [(i, j) for i in range(grid_size) for j in range(grid_size) if (i, j) != (0, 0)]
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self.rng = random.Random()
        if render_mode is not None:
            self.renderer = LetterEnvRenderer(self.grid_size, render_mode=render_mode,
                                              render_fps=self.metadata['render_fps'])

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        di, dj = self.actions[action]
        agent_i = (self.agent[0] + di + self.grid_size) % self.grid_size
        agent_j = (self.agent[1] + dj + self.grid_size) % self.grid_size
        self.agent = agent_i, agent_j

        if self.render_mode == "human":
            self._render_frame()

        return self._get_observation(), 0.0, False, False, {'propositions': self.get_active_propositions()}

    def wait_for_input(self) -> bool:
        if self.render_mode not in ["human", "path"]:
            return False
        return self.renderer.wait_for_input()

    def _get_observation(self) -> ObsType:
        obs = np.zeros(shape=(self.grid_size, self.grid_size, len(self.letter_types) + 1), dtype=np.uint8)

        # Getting agent-centric view (if needed)
        c_map, agent = self.map, self.agent
        if self.use_agent_centric_view:
            c_map, agent = self._get_centric_map()

        # adding objects
        for loc in c_map:
            letter_id = self.letter_types.index(c_map[loc])
            obs[loc[0], loc[1], letter_id] = 1

        # adding agent
        obs[agent[0], agent[1], len(self.letter_types)] = 1
        return obs

    def _get_centric_map(self):
        center = self.grid_size // 2
        agent = (center, center)
        delta = center - self.agent[0], center - self.agent[1]
        c_map = {}
        for loc in self.map:
            new_loc_i = (loc[0] + delta[0] + self.grid_size) % self.grid_size
            new_loc_j = (loc[1] + delta[1] + self.grid_size) % self.grid_size
            c_map[(new_loc_i, new_loc_j)] = self.map[loc]
        return c_map, agent

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        if seed is not None:
            self.rng = random.Random(seed)
        if not self.use_fixed_map:
            self.map = None
        # Sampling a new map
        while self.map is None:
            # Sampling a random map
            self.map = {}
            self.rng.shuffle(self.locations)
            for i in range(len(self.letters)):
                self.map[self.locations[i]] = self.letters[i]
            # Checking that the map is valid
            if _is_valid_map(self.map, self.grid_size, self.actions):
                break
            self.map = None

        # Locating the agent into (0,0)
        self.agent = (0, 0)
        return self._get_observation(), {'propositions': set()}

    def print(self):
        c_map, agent = self.map, self.agent
        if self.use_agent_centric_view:
            c_map, agent = self._get_centric_map()
        print("*" * (self.grid_size + 2))
        for i in range(self.grid_size):
            line = "*"
            for j in range(self.grid_size):
                if (i, j) == agent:
                    line += "A"
                elif (i, j) in c_map:
                    line += c_map[(i, j)]
                else:
                    line += " "
            print(line + "*")
        print("*" * (self.grid_size + 2))
        print("Active:", self.get_active_propositions())

    def print_features(self):
        obs = self._get_observation()
        print("*" * (self.grid_size + 2))
        for i in range(self.grid_size):
            line = "*"
            for j in range(self.grid_size):
                if np.max(obs[i, j, :]) > 0:
                    line += str(np.argmax(obs[i, j, :]))
                else:
                    line += " "
            print(line + "*")
        print("*" * (self.grid_size + 2))

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def render_path(self, actions: list[int]):
        return self.renderer.render_path(self, [self.actions[a] for a in actions])

    def _render_frame(self):
        return self.renderer.render(self)

    def close(self):
        if self.render_mode is not None:
            self.renderer.close()

    def get_active_propositions(self) -> set[str]:
        if self.agent in self.map:
            letter = self.map[self.agent]
            return {letter}
        return set()

    def get_propositions(self) -> list[str]:
        return self.letter_types

    def get_possible_assignments(self) -> list[Assignment]:
        return Assignment.zero_or_one_propositions(set(self.get_propositions()))

    def save_world_info(self, path: str):
        with open(path, 'wb+') as f:
            pickle.dump(self.map, f)

    def load_world_info(self, path: str):
        with open(path, 'rb') as f:
            self.map = pickle.load(f)
        self.use_fixed_map = True


class LetterSafetyEnv(gym.Env):
    """
    This environment is a grid with randomly located letters on it.
    We ensure that there is a clean path to any of the letters (a path that includes no passing by any letter).
    Note that stepping outside the map causes the agent to appear on the other side.
    """
    metadata = {'render_modes': ['human', 'rgb_array', 'path'], 'render_fps': 10}

    def __init__(
            self,
            grid_size: int,
            letters: str,
            use_fixed_map: bool,
            use_agent_centric_view: bool,
            render_mode: str | None = None,
            map: dict[tuple[int, int], str] = None
    ):
        if use_agent_centric_view and grid_size % 2 == 0:
            raise ValueError("Agent-centric view is only available for odd grid-sizes")
        self.render_mode = render_mode
        self.grid_size = grid_size
        self.letters = letters
        self.use_fixed_map = use_fixed_map
        self.use_agent_centric_view = use_agent_centric_view
        self.letter_types = sorted(list(set(letters)))
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size, len(self.letter_types) + 1),
                                            dtype=np.uint8)
        self.map = map
        self.agent = (0, 0)
        self.original_obs = None
        self.locations = [(i, j) for i in range(grid_size) for j in range(grid_size) if (i, j) != (0, 0)]
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self.rng = random.Random()
        if render_mode is not None:
            self.renderer = LetterEnvRenderer(self.grid_size, render_mode=render_mode,
                                              render_fps=self.metadata['render_fps'])

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        di, dj = self.actions[action]
        agent_i = (self.agent[0] + di + self.grid_size) % self.grid_size
        agent_j = (self.agent[1] + dj + self.grid_size) % self.grid_size
        self.agent = agent_i, agent_j

        if self.render_mode == "human":
            self._render_frame()
        self.original_obs = self._get_observation()
        return self.original_obs, 0.0, False, False, {'propositions': self.get_active_propositions()}

    def wait_for_input(self) -> bool:
        if self.render_mode not in ["human", "path"]:
            return False
        return self.renderer.wait_for_input()

    def _get_observation(self) -> ObsType:
        obs = np.zeros(shape=(self.grid_size, self.grid_size, len(self.letter_types) + 1), dtype=np.uint8)

        # Getting agent-centric view (if needed)
        c_map, agent = self.map, self.agent
        if self.use_agent_centric_view:
            c_map, agent = self._get_centric_map()

        # adding objects
        for loc in c_map:
            letter_id = self.letter_types.index(c_map[loc])
            obs[loc[0], loc[1], letter_id] = 1

        # adding agent
        obs[agent[0], agent[1], len(self.letter_types)] = 1
        return obs

    def _get_centric_map(self):
        center = self.grid_size // 2
        agent = (center, center)
        delta = center - self.agent[0], center - self.agent[1]
        c_map = {}
        for loc in self.map:
            new_loc_i = (loc[0] + delta[0] + self.grid_size) % self.grid_size
            new_loc_j = (loc[1] + delta[1] + self.grid_size) % self.grid_size
            c_map[(new_loc_i, new_loc_j)] = self.map[loc]
        return c_map, agent

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        if seed is not None:
            self.rng = random.Random(seed)
        if not self.use_fixed_map:
            self.map = None
        # Sampling a new map
        while self.map is None:
            # Sampling a random map
            self.map = {}
            self.rng.shuffle(self.locations)
            for i in range(len(self.letters)):
                self.map[self.locations[i]] = self.letters[i]
            # Checking that the map is valid
            if _is_valid_map(self.map, self.grid_size, self.actions):
                break
            self.map = None

        # Locating the agent into (0,0)
        self.agent = (0, 0)
        self.original_obs = self._get_observation()
        return self.original_obs, {'propositions': set()}

    def print(self):
        c_map, agent = self.map, self.agent
        if self.use_agent_centric_view:
            c_map, agent = self._get_centric_map()
        print("*" * (self.grid_size + 2))
        for i in range(self.grid_size):
            line = "*"
            for j in range(self.grid_size):
                if (i, j) == agent:
                    line += "A"
                elif (i, j) in c_map:
                    line += c_map[(i, j)]
                else:
                    line += " "
            print(line + "*")
        print("*" * (self.grid_size + 2))
        print("Active:", self.get_active_propositions())

    def print_features(self):
        obs = self._get_observation()
        print("*" * (self.grid_size + 2))
        for i in range(self.grid_size):
            line = "*"
            for j in range(self.grid_size):
                if np.max(obs[i, j, :]) > 0:
                    line += str(np.argmax(obs[i, j, :]))
                else:
                    line += " "
            print(line + "*")
        print("*" * (self.grid_size + 2))

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def render_path(self, actions: list[int]):
        return self.renderer.render_path(self, [self.actions[a] for a in actions])

    def _render_frame(self):
        return self.renderer.render(self)

    def close(self):
        if self.render_mode is not None:
            self.renderer.close()

    def get_active_propositions(self) -> set[str]:
        if self.agent in self.map:
            letter = self.map[self.agent]
            return {letter}
        return set()

    def get_propositions(self) -> list[str]:
        return self.letter_types

    def get_possible_assignments(self) -> list[Assignment]:
        return Assignment.zero_or_one_propositions(set(self.get_propositions()))

    def save_world_info(self, path: str):
        with open(path, 'wb+') as f:
            pickle.dump(self.map, f)

    def load_world_info(self, path: str):
        with open(path, 'rb') as f:
            self.map = pickle.load(f)
        self.use_fixed_map = True


def _is_valid_map(map, grid_size, actions):
    open_list = [(0, 0)]
    closed_list = set()
    while open_list:
        s = open_list.pop()
        closed_list.add(s)
        if s not in map:
            for di, dj in actions:
                si = (s[0] + di + grid_size) % grid_size
                sj = (s[1] + dj + grid_size) % grid_size
                if (si, sj) not in closed_list and (si, sj) not in open_list:
                    open_list.append((si, sj))
    return len(closed_list) == grid_size * grid_size


class LetterEnvRenderer:
    def __init__(self, grid_size: int, render_mode: Literal["human", "path", "rgb_array"] = "human", render_fps=1,
                 cell_size=120):
        self.grid_size = grid_size
        self.cell_size = cell_size  # Size of each cell in pixels
        self.screen_size = self.grid_size * self.cell_size
        self.render_mode = render_mode
        self.render_fps = render_fps
        if render_mode == "human" or render_mode == "path":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption('LetterWorld')
            self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, int(self.cell_size * 0.75))  # Font size adjusted to cell size
        self.bg_color = (240, 240, 240)  # Soft background color
        self.cell_color = (200, 200, 200)
        self.agent_color = (0, 128, 0)  # Green transparent color for the agent
        self.border_color = (0, 0, 0)  # Black color for borders
        self.arrow_color = (47, 79, 79, 160)  # Gray color for arrows

    def render(self, env: LetterEnv):
        canvas = self.draw_canvas(env)
        return self.update(canvas)

    def draw_canvas(self, env: LetterEnv):
        canvas = pygame.Surface((self.screen_size, self.screen_size))
        canvas.fill(self.bg_color)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                if (i, j) == env.agent:
                    bg_color = self.agent_color
                    fg_color = (255, 255, 255)
                elif (i, j) in env.map:
                    bg_color = self.cell_color
                    fg_color = (0, 0, 0)
                else:
                    bg_color = self.bg_color
                pygame.draw.rect(canvas, bg_color, rect)
                if (i, j) in env.map:
                    text_surface = self.font.render(env.map[(i, j)], True, fg_color)
                    canvas.blit(text_surface, text_surface.get_rect(center=rect.center))
                pygame.draw.rect(canvas, self.border_color, rect, 2)  # Draw cell border in black
        return canvas

    def update(self, canvas):
        if self.render_mode == "human" or self.render_mode == "path":
            self.screen.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.render_fps)
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def render_path(self, env, actions: list[tuple[int, int]]):
        canvas = self.draw_canvas(env)
        path_canvas = pygame.Surface((self.screen_size, self.screen_size), pygame.SRCALPHA)
        path_canvas.fill((0, 0, 0, 0))  # Transparent background

        current_pos = (0, 0)
        for action in actions:
            next_pos = ((current_pos[0] + action[0]) % env.grid_size, (current_pos[1] + action[1]) % env.grid_size)
            self.draw_arrow(path_canvas, current_pos, next_pos)
            current_pos = next_pos

        canvas.blit(path_canvas, (0, 0))
        return self.update(canvas)

    def wait_for_input(self) -> bool:
        pygame.event.clear()
        while True:
            event = pygame.event.wait()
            if event.type == pygame.KEYDOWN:
                return event.key == pygame.K_q

    def draw_arrow(self, canvas, start_pos, end_pos):
        start_x = start_pos[1] * self.cell_size + self.cell_size // 2
        start_y = start_pos[0] * self.cell_size + self.cell_size // 2
        end_x = end_pos[1] * self.cell_size + self.cell_size // 2
        end_y = end_pos[0] * self.cell_size + self.cell_size // 2

        def draw_arrow_segment(sx, sy, ex, ey):
            pygame.draw.line(canvas, self.arrow_color, (sx, sy), (ex, ey), 5)
            angle = np.arctan2(ey - sy, ex - sx)
            arrowhead_size = 10
            arrowhead_angle = np.pi / 6
            arrowhead_points = [
                (ex, ey),
                (ex - arrowhead_size * np.cos(angle - arrowhead_angle),
                 ey - arrowhead_size * np.sin(angle - arrowhead_angle)),
                (ex - arrowhead_size * np.cos(angle + arrowhead_angle),
                 ey - arrowhead_size * np.sin(angle + arrowhead_angle))
            ]
            pygame.draw.polygon(canvas, self.arrow_color, arrowhead_points)

        # Check for wrapping and draw segments
        if abs(start_pos[1] - end_pos[1]) > 1:
            if start_pos[1] > end_pos[1]:  # Wrapping horizontally left to right
                mid_x = self.screen_size
                draw_arrow_segment(start_x, start_y, mid_x, start_y)
                draw_arrow_segment(0, end_y, end_x, end_y)
            elif start_pos[1] < end_pos[1]:  # Wrapping horizontally right to left
                mid_x = 0
                draw_arrow_segment(start_x, start_y, mid_x, start_y)
                draw_arrow_segment(self.screen_size, end_y, end_x, end_y)
        elif abs(start_pos[0] - end_pos[0]) > 1:
            if start_pos[0] > end_pos[0]:  # Wrapping vertically bottom to top
                mid_y = self.screen_size
                draw_arrow_segment(start_x, start_y, start_x, mid_y)
                draw_arrow_segment(end_x, 0, end_x, end_y)
            elif start_pos[0] < end_pos[0]:  # Wrapping vertically top to bottom
                mid_y = 0
                draw_arrow_segment(start_x, start_y, start_x, mid_y)
                draw_arrow_segment(end_x, self.screen_size, end_x, end_y)
        else:
            draw_arrow_segment(start_x, start_y, end_x, end_y)

    def close(self):
        if self.render_mode == "human" or self.render_mode == "path":
            pygame.display.quit()
            pygame.quit()


def main():
    # commands
    str_to_action = {"w": 0, "s": 1, "a": 2, "d": 3}
    grid_size = 7
    letters = "aabbccddee"
    use_fixed_map = False
    use_agent_centric_view = True
    timeout = 10

    # play the game!
    game = LetterEnv(grid_size, letters, use_fixed_map, use_agent_centric_view)
    game = TimeLimit(game, timeout)
    while True:
        # Episode
        game.reset()
        while True:
            game.show()
            print("\nAction? ", end="")
            a = input()
            print()
            # Executing action
            if a in str_to_action:
                obs, reward, term, trunc, _ = game.step(str_to_action[a])
                if term or trunc:
                    break
            else:
                print("Forbidden action")
        game.show()

if __name__ == "__main__":
    main()
