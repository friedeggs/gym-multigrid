from gym_multigrid.multigrid import *

import os
import numpy as np
from tqdm import tqdm

class GridLayout:
    def __init__(self, size):
        self.size = size
        self.grid = [[False for _ in range(self.size)] for _ in range(self.size)]
        for i in range(self.size):
            self.grid[i][0] = self.grid[0][i] = self.grid[i][self.size-1] = self.grid[self.size-1][i] = True

    # def _check(self, i, j):
    #   count = 0
    #   count += self.grid[i][j] == self.grid[i+1][j]
    #   count += self.grid[i+1][j] == self.grid[i+1][j+1]
    #   count += self.grid[i+1][j+1] == self.grid[i][j+1]
    #   count += self.grid[i][j+1] == self.grid[i][j]
    #   return count

    def check(self, i, j):
        count = 0
        count += self.grid[i][j] == self.grid[i+1][j]
        count += self.grid[i+1][j] == self.grid[i+1][j+1]
        count += self.grid[i+1][j+1] == self.grid[i][j+1]
        count += self.grid[i][j+1] == self.grid[i][j]
        return count == 2

    def check_connected(self):
        flag = [[False for _ in range(self.size)] for _ in range(self.size)]
        # for i in range(self.size):
        #   flag[i][0] = flag[0][i] = flag[i][self.size-1] = flag[self.size-1][i] = True
        i = 1 
        j = 1 
        while i < self.size-1 and j < self.size-1 and self.grid[i][j]:
            i += 1 
            if i == self.size-1:
                i = 1
                j += 1
        # found empty tile
        flag[i][j] = True
        tiles = [(i+1, j), (i, j+1), (i-1, j), (i, j-1)]
        while tiles:
            i, j = tiles[0]
            tiles = tiles[1:]
            if self.grid[i][j] or flag[i][j]:
                continue
            else:
                flag[i][j] = True
                tiles.extend([(i+1, j), (i, j+1), (i-1, j), (i, j-1)])
        # for x in range(self.size):
        #   for y in range(self.size):
        #       if flag[x][y]:
        #           print('*\t', end='')
        #       else:
        #           print(' \t', end='')
        #   print()
        # print()
        # check for second component
        i = 1 
        j = 1
        while i < self.size-1 and j < self.size-1 and (self.grid[i][j] or flag[i][j]):
            i += 1 
            if i == self.size-1:
                i = 1
                j += 1
        # print(i, j)
        # if i != self.size-1:
        if i < self.size-1 and j < self.size-1 and not self.grid[i][j] and not flag[i][j]:
            return False
        return True

    def done(self):
        for i in range(self.size-1):
            for j in range(self.size-1):
                if not self.check(i, j):
                    return False
        return True

    def place(self):
        # get a free tile
        x = np.random.choice(range(1, self.size-1))
        y = np.random.choice(range(1, self.size-1))
        while self.grid[x][y]:
            x = np.random.choice(range(1, self.size-1))
            y = np.random.choice(range(1, self.size-1))

        # try to place
        self.grid[x][y] = True
        if not all([self.check(x, y), self.check(x-1,y), self.check(x,y-1), self.check(x-1,y-1), self.check_connected()]):
        # if not all(self.check(x, y), self.check(x+1,y), self.check(x,y+1), self.check(x+1,y+1)):
            self.grid[x][y] = False
            return self.place()
        else:
            return x, y

    def generate(self):
        count = 0
        while not self.done() or count < 8:
            try:
                if count == 8:
                    # import pdb; pdb.set_trace()
                    break
                x, y = self.place()
                count += 1
            except RecursionError:
                return False
        # self.grid[x][y] = False
        return True


    def print(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j]:
                    print('*\t', end='')
                else:
                    print(' \t', end='')
            print()
        print()

    def build_tree(self):
        flag = [[False for _ in range(self.size)] for _ in range(self.size)]
        deg = [[0 for _ in range(self.size)] for _ in range(self.size)]
        neighbors = [[[] for _ in range(self.size)] for _ in range(self.size)]
        i = 1 
        j = 1 
        while i < self.size-1 and j < self.size-1 and self.grid[i][j]:
            i += 1 
            if i == self.size-1:
                i = 1
                j += 1
        # found empty tile
        flag[i][j] = True
        tiles = [(i+1, j), (i, j+1), (i-1, j), (i, j-1)]
        prev = [(i, j), (i, j), (i, j), (i, j)]
        while tiles:
            i, j = tiles[0]
            iprev, jprev = prev[0]
            tiles = tiles[1:]
            prev = prev[1:]
            if self.grid[i][j] or flag[i][j]:
                continue
            else:
                # add an edge between (i, j) and (iprev, jprev)
                neighbors[i][j].append((iprev, jprev))
                neighbors[iprev][jprev].append((i, j))
                deg[i][j] += 1
                deg[iprev][jprev] += 1
                flag[i][j] = True
                tiles.extend([(i+1, j), (i, j+1), (i-1, j), (i, j-1)])
                prev.extend([(i, j), (i, j), (i, j), (i, j)])

        for x in range(self.size):
            for y in range(self.size):
                if flag[x][y]:
                    print(f'{deg[x][y]}\t', end='')
                else:
                    print(' \t', end='')
            print()
        print()

        self.flag = flag
        self.deg = deg 
        self.neighbors = neighbors

    def place_objects(self):
        flag = self.flag 
        deg = self.deg 
        # neighbors = self.neighbors

        max_deg = 0
        locs = []
        end_locs = []
        for x in range(self.size):
            for y in range(self.size):
                if flag[x][y]:
                    if deg[x][y] > max_deg:
                        locs = [(x, y)]
                        max_deg = deg[x][y]
                    elif deg[x][y] == max_deg:
                        locs.append((x, y))
                    if deg[x][y] == 1:
                        end_locs.append((x, y))

        # print(locs)
        ball_loc = locs[np.random.choice(np.arange(len(locs)))]

        self.grid[ball_loc[0]][ball_loc[1]] = True

        flag2 = [[False for _ in range(self.size)] for _ in range(self.size)]
        i = 1 
        j = 1 
        while i < self.size-1 and j < self.size-1 and self.grid[i][j]:
            i += 1 
            if i == self.size-1:
                i = 1
                j += 1
        # found empty tile
        flag2[i][j] = True
        tiles = [(i+1, j), (i, j+1), (i-1, j), (i, j-1)]
        while tiles:
            i, j = tiles[0]
            tiles = tiles[1:]
            if self.grid[i][j] or flag2[i][j]:
                continue
            else:
                flag2[i][j] = True
                tiles.extend([(i+1, j), (i, j+1), (i-1, j), (i, j-1)])


        self.grid[ball_loc[0]][ball_loc[1]] = False

        same_component = np.random.random() < 0.5
        poss_locs = []
        goal_locs = []
        for loc in end_locs:
            x, y = loc 
            if same_component and flag2[x][y]:
                poss_locs.append((x, y))
            elif not same_component and not flag2[x][y]:
                poss_locs.append((x, y))
            if same_component and not flag2[x][y]:
                goal_locs.append((x, y))
            elif not same_component and flag2[x][y]:
                goal_locs.append((x, y))

        caregiver_loc = poss_locs[np.random.choice(np.arange(len(poss_locs)))]
        goal_loc = goal_locs[np.random.choice(np.arange(len(goal_locs)))]

        locs = []
        for loc in end_locs:
            if loc != caregiver_loc and loc != goal_loc:
                locs.append(loc)

        agent_loc = locs[np.random.choice(np.arange(len(locs)))]

        self.full_grid = [['' for _ in range(self.size)] for _ in range(self.size)]

        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j]:
                    self.full_grid[i][j] = '*'
                if (i, j) == agent_loc:
                    self.full_grid[i][j] = 'A'
                if (i, j) == ball_loc:
                    self.full_grid[i][j] = 'B'
                if (i, j) == caregiver_loc:
                    self.full_grid[i][j] = 'C'
                if (i, j) == goal_loc:
                    self.full_grid[i][j] = 'X'

        for x in range(self.size):
            for y in range(self.size):
                print(f'{self.full_grid[x][y]}\t', end='')
            print()
        print()

        self.agent_loc = agent_loc
        self.ball_loc = ball_loc
        self.caregiver_loc = caregiver_loc
        self.goal_loc = goal_loc
        agent_start_dir = np.random.choice(4)
        caregiver_start_dir = np.random.choice(4)
        self.agent_start_dir = agent_start_dir
        self.caregiver_start_dir = caregiver_start_dir

    def solve(self):
        DIR_TO_VEC = [
            # Pointing right (positive X)
            np.array((1, 0)),
            # Down (positive Y)
            np.array((0, 1)),
            # Pointing left (negative X)
            np.array((-1, 0)),
            # Up (negative Y)
            np.array((0, -1)),
        ]

        VEC_TO_DIR = {(x[0], x[1]): i for i, x in enumerate(DIR_TO_VEC)}
        
        neighbors = self.neighbors

        start_loc = self.caregiver_loc
        end_loc = self.goal_loc

        # actions = []

        # path = []

        # flag = [[False for _ in range(self.size)] for _ in range(self.size)]

        # DFS
        # cur_loc = start_loc 
        def dfs(cur_loc, path):
            if cur_loc == end_loc:
                return path
            for neighbor in neighbors[cur_loc[0]][cur_loc[1]]:
                if neighbor in path: continue
                path = dfs(neighbor, path + [neighbor])
                if path:
                    return path
            return []

        path = dfs(start_loc, [])

        print(path)

        actions = []

        # turns
        prev_loc = path[0]
        prev_dir = self.caregiver_start_dir
        for cur_loc in path[1:]:
            vec = (cur_loc[0]-prev_loc[0], cur_loc[1]-prev_loc[1])
            cur_dir = VEC_TO_DIR[vec]
            if cur_dir != prev_dir:
                # add turns 
                actions.append(1) # TODO
                actions.append(2)
            else:
                actions.append(2)
            prev_dir = cur_dir
            prev_loc = cur_loc

        print(actions)

    def read(self, f):
        lines = []
        self.full_grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]
        for i in range(self.size):
            line = f.readline()
            lines.append(line)
            for j, char in enumerate(line[:-1]):
                self.full_grid[i][j] = char

                if char == 'A':
                    self.agent_loc = (i, j)
                elif char == 'B':
                    self.ball_loc = (i, j)
                elif char == 'C':
                    self.caregiver_loc = (i, j)
                elif char == 'X':
                    self.goal_loc = (i, j)

                if char == '*':
                    self.grid[i][j] = True
                else:
                    self.grid[i][j] = False


class ProsocialEnv(MultiGridEnv):
    """
    Environment in which the agents have to collect the balls
    """

    def __init__(
        self,
        size=None,
        # width=7,
        # height=4,
        width=7,
        height=7,
        agents_index = [0,1],
        view_size=7,
        render_mode = None,
        reward_model=None,
        reward_mode='default',

    ):
        self.world = World
        self.possible_agents = agents_index
        self.render_mode = render_mode
        self.agents_index = agents_index
        self.reward_model = reward_model
        self.reward_mode = reward_mode

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))
        self.agents = agents

        np.random.seed(1234)

        # layout = GridLayout(7)
        # success = False
        # while not success:
        #     success = layout.generate()
        # self.layout = layout

        # # self.layout.print()
        # self.layout.build_tree()
        # self.layout.place_objects()
        # # self.layout.solve()

        self.reset_counter = 0

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps=100,
            # Set this to True for maximum speed
            see_through_walls=False,
            partial_obs=False,
            agents=agents,
            agent_view_size=view_size
        )

        self.action_space = spaces.Discrete(5, start=1)
        # self.observation_space = spaces.Box(
        #     low=0,
        #     high=255,
        #     shape=(len(self.agents), view_size, view_size, self.objects.encode_dim+1),
        #     dtype='uint8'
        # )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # # Generate the surrounding walls
        # self.grid.horz_wall(self.world, 0, 0)
        # self.grid.horz_wall(self.world, 0, height-1)
        # self.grid.vert_wall(self.world, 0, 0)
        # self.grid.vert_wall(self.world, width-1, 0)

        lenbytes = os.path.getsize("file.txt")
        num_grids = (lenbytes-1) // 57
        grid_idx = np.random.choice(num_grids)
        f = open("file.txt", "r")
        f.seek(grid_idx*((7+1)*7+1))
        layout = GridLayout(7)
        layout.read(f)
        f.close()
        self.layout = layout

        # layout = self.layout
        for i in range(7):
            for j in range(7):
                if layout.grid[i][j]:
                    self.grid.set(i, j, Wall(self.world))

        self.put_obj(Goal(self.world, 1, COLORS['red']), layout.goal_loc[0], layout.goal_loc[1])
        self.put_obj(Ball(self.world, 2, COLORS['blue']), layout.ball_loc[0], layout.ball_loc[1])

        # # x = np.random.choice(range(1, width-1))
        # # y = np.random.choice(range(1, height-1))
        # x = width-2
        # y = height-2
        # self.put_obj(Goal(self.world, 0, COLORS['green']), x, y)
        # # self.put_obj(Goal(self.world, 0), width - 2, height - 2)

        # Randomize the player start position and orientation
        self.agent_pos = []
        self.agent_dir = []
        # # for a in self.agents[:1]:
        # pos = self.place_agent(self.agents[0])
        # self.agent_pos.append(pos)
        # self.agent_dir.append(self.agents[0].dir)

        self.put_obj(self.agents[0], layout.agent_loc[0], layout.agent_loc[1])
        self.agents[0].pos = self.agents[0].init_pos
        self.agents[0].dir = np.random.choice(4)
        self.agent_pos.append(self.agents[0].pos)
        self.agent_dir.append(self.agents[0].dir)

        self.put_obj(self.agents[1], layout.caregiver_loc[0], layout.caregiver_loc[1])
        self.agents[1].pos = self.agents[1].init_pos
        self.agents[1].dir = np.random.choice(4)
        self.agent_pos.append(self.agents[1].pos)
        self.agent_dir.append(self.agents[1].dir)

        # self.put_obj(self.agents[1], 1, height-2)
        # self.agents[1].pos = self.agents[1].init_pos
        # self.agents[1].dir = 0
        # self.agent_pos.append(self.agents[1].pos)
        # self.agent_dir.append(self.agents[1].dir)

        self.step_count = 0

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if self.agents[i].carrying is None:
                    self.agents[i].carrying = fwd_cell
                    self.agents[i].carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
            elif fwd_cell.type=='agent':
                if fwd_cell.carrying:
                    if self.agents[i].carrying is None:
                        self.agents[i].carrying = fwd_cell.carrying
                        fwd_cell.carrying = None      

    def _reward(self, i, rewards, reward=1):
        """
        Compute the reward to be given upon success
        """
        if self.reward_mode == 'none':
            reward = 0.
        else:
            if i == 0:
                # rewards[0]+=reward # TODO remove
                pass
            else:
                for j,a in enumerate(self.agents):
                    if a.index == i:
                        rewards[j]+=0.4*reward
                        rewards[0]+=0.4*reward # shared rewards

    def step(self, actions):
        obs, rewards, done, info = MultiGridEnv.step(self, actions)
        # terminated = [self.step_count == self.max_steps] * len(self.agents)
        truncated = self.step_count >= self.max_steps

        return obs, rewards, done, truncated, info


