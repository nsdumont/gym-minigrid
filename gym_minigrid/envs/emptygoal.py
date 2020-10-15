from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np
from gym import spaces
from gym_minigrid.wrappers import SSPSpace

class EmptyGoalEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        switch_prob = 0.01,
        use_ssp = False
    ):
        
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.size = size
        self.switch_prob = switch_prob
        self.goal_loc_x = np.random.randint(1,size-1)
        self.goal_loc_y = np.random.randint(1,size-1)
        
        self.use_ssp = use_ssp
        if self.use_ssp:
            import nengo_ssp as ssp
            X,Y,_ = ssp.HexagonalBasis(10,10)
            self.X = X
            self.Y = Y
        
        self.goal_locs = []
        self.goal_locs.append([self.goal_loc_x,self.goal_loc_y])
                               
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )
        
        
        
        #xx, yy = np.meshgrid(np.arange(1,size-1),np.arange(1,size-1))
        #self.xy_list = np.vstack([xx.reshape(-1),yy.reshape(-1)]).T
        
        if self.use_ssp:
            self.observation_space.spaces.update({'mission': SSPSpace(basis=[X,Y], radius=1)})
        else:
            self.observation_space.spaces.update({'mission': spaces.Box(low=0,high=self.size,shape=(2,),dtype='uint8')})

        

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square randomly
        r = np.random.rand()
        if r <= self.switch_prob:
            self.goal_loc_x = np.random.randint(1,self.size-1)
            self.goal_loc_y = np.random.randint(1,self.size-1)
            self.goal_locs.append([self.goal_loc_x,self.goal_loc_y])
        self.put_obj(Goal(), self.goal_loc_x, self.goal_loc_y)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        if self.use_ssp:
            self.mission = (self.X**self.goal_loc_x * self.Y**self.goal_loc_y).v.real
        else:
            self.mission = np.array([self.goal_loc_x, self.goal_loc_y])
        #1*np.all(self.xy_list == np.array([self.goal_loc_x,self.goal_loc_y]),axis=1)

class EmptyGoalEnv5x5(EmptyGoalEnv):
    def __init__(self, **kwargs):
        super().__init__(size=5, **kwargs)

class EmptyRandomGoalEnv5x5(EmptyGoalEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)

class EmptyGoalEnv6x6(EmptyGoalEnv):
    def __init__(self, **kwargs):
        super().__init__(size=6, **kwargs)

class EmptyRandomGoalEnv6x6(EmptyGoalEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)
        
class EmptyGoalEnv16x16(EmptyGoalEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, **kwargs)

register(
    id='MiniGrid-EmptyGoal-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyGoalEnv5x5'
)

register(
    id='MiniGrid-EmptyGoal-Random-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyRandomGoalEnv5x5'
)

register(
    id='MiniGrid-EmptyGoal-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyGoalEnv6x6'
)

register(
    id='MiniGrid-EmptyGoal-Random-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyRandomGoalEnv6x6'
)

register(
    id='MiniGrid-EmptyGoal-8x8-v0',
    entry_point='gym_minigrid.envs:EmptyGoalEnv'
)

register(
    id='MiniGrid-EmptyGoal-16x16-v0',
    entry_point='gym_minigrid.envs:EmptyGoalEnv16x16'
)