import math
import operator
from functools import reduce

import numpy as np
import gym
from gym import error, spaces, utils
from .minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX, IDX_TO_OBJECT, IDX_TO_COLOR, IDX_TO_STATE,DIR_TO_VEC
import nengo_spa as spa
import nengo_ssp as ssp

class ReseedWrapper(gym.core.Wrapper):
    """
    Wrapper to always regenerate an environment with the same set of seeds.
    This can be used to force an environment to always keep the same
    configuration when reset.
    """

    def __init__(self, env, seeds=[0], seed_idx=0):
        self.seeds = list(seeds)
        self.seed_idx = seed_idx
        super().__init__(env)

    def reset(self, **kwargs):
        seed = self.seeds[self.seed_idx]
        self.seed_idx = (self.seed_idx + 1) % len(self.seeds)
        self.env.seed(seed)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

class ActionBonus(gym.core.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        tup = (tuple(env.agent_pos), env.agent_dir, action)

        # Get the count for this (s,a) pair
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this (s,a) pair
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class StateBonus(gym.core.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = (tuple(env.agent_pos))

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, obs):
        return obs['image']

class OneHotPartialObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a partially observable
    agent view as observation.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space['image'].shape

        # Number of bits per cell
        num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0], obs_shape[1], num_bits),
            dtype='uint8'
        )

    def observation(self, obs):
        img = obs['image']
        out = np.zeros(self.observation_space.spaces['image'].shape, dtype='uint8')

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                type = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]

                out[i, j, type] = 1
                out[i, j, len(OBJECT_TO_IDX) + color] = 1
                out[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1

        return {
            'mission': obs['mission'],
            'image': out
        }

class RGBImgObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as the only observation output,
    no language/mission. This can be used to have the agent to solve the
    gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width * tile_size, self.env.height * tile_size, 3),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img = env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=self.tile_size
        )

        return {
            'mission': obs['mission'],
            'image': rgb_img
        }


class RGBImgPartialObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as the only observation output
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space.spaces['image'].shape
        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img_partial = env.get_obs_render(
            obs['image'],
            tile_size=self.tile_size
        )

        return {
            'mission': obs['mission'],
            'image': rgb_img_partial
        }

class FullyObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])

        return {
            'mission': obs['mission'],
            'image': full_grid
        }

class FlatObsWrapper(gym.core.ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 27

        imgSpace = env.observation_space.spaces['image']
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize + self.numCharCodes * self.maxStrLen,),
            dtype='uint8'
        )

        self.cachedStr = None
        self.cachedArray = None

    def observation(self, obs):
        image = obs['image']
        mission = obs['mission']

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert len(mission) <= self.maxStrLen, 'mission string too long ({} chars)'.format(len(mission))
            mission = mission.lower()

            strArray = np.zeros(shape=(self.maxStrLen, self.numCharCodes), dtype='float32')

            for idx, ch in enumerate(mission):
                if ch >= 'a' and ch <= 'z':
                    chNo = ord(ch) - ord('a')
                elif ch == ' ':
                    chNo = ord('z') - ord('a') + 1
                assert chNo < self.numCharCodes, '%s : %d' % (ch, chNo)
                strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))

        return obs

class ViewSizeWrapper(gym.core.Wrapper):
    """
    Wrapper to customize the agent field of view size.
    This cannot be used with fully observable wrappers.
    """

    def __init__(self, env, agent_view_size=7):
        super().__init__(env)

        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3

        # Override default view size
        env.unwrapped.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(agent_view_size, agent_view_size, 3),
            dtype='uint8'
        )

        # Override the environment's observation space
        self.observation_space = spaces.Dict({
            'image': observation_space
        })

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

from .minigrid import Goal
class DirectionObsWrapper(gym.core.ObservationWrapper):
    """
    Provides the slope/angular direction to the goal with the observations as modeled by (y2 - y2 )/( x2 - x1)
    type = {slope , angle}
    """
    def __init__(self, env,type='slope'):
        super().__init__(env)
        self.goal_position = None
        self.type = type

    def reset(self):
        obs = self.env.reset()
        if not self.goal_position:
            self.goal_position = [x for x,y in enumerate(self.grid.grid) if isinstance(y,(Goal) ) ]
            if len(self.goal_position) >= 1: # in case there are multiple goals , needs to be handled for other env types
                self.goal_position = (int(self.goal_position[0]/self.height) , self.goal_position[0]%self.width)
        return obs

    def observation(self, obs):
        slope = np.divide( self.goal_position[1] - self.agent_pos[1] ,  self.goal_position[0] - self.agent_pos[0])
        obs['goal_direction'] = np.arctan( slope ) if self.type == 'angle' else slope
        return obs
    

    
    
class SSPGoalBonus(gym.core.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = (tuple(env.agent_pos))

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class SSPWrapper(gym.core.ObservationWrapper):

    def __init__(self, env,d,X=None,Y=None,delta=2,rng=None):
        super().__init__(env)
        
        self.alg = spa.algebras.HrrAlgebra()

        img_shape = env.observation_space['image'].shape
        self.img_shape = img_shape
        self.d = d
        self.X = X or ssp.vector_generation.UnitaryVectors(d)
        self.Y = Y or ssp.vector_generation.UnitaryVectors(d)
        
        colors = [x.upper() for x in list(COLOR_TO_IDX.keys())]
        
        pointer_gen = spa.vector_generation.UnitaryVectors(self.d, self.alg, rng=rng)
        objects = [x.upper() for x in list(OBJECT_TO_IDX.keys())]

        vocab = spa.Vocabulary(d, pointer_gen = pointer_gen)
        vocab.populate(';'.join(colors + objects ))
        vocab.add('NULL', np.zeros(d))
        vocab.add('OPEN',  self.alg.identity_element(d))
        #vocab.populate('CLOSED;LOCKED')

        
        #vocab.populate(';'.join(objects))
        
        #states = [x.upper() for x in list(STATE_TO_IDX.keys())]
        #vocab.populate(';'.join(states))
        #vocab.populate('AGENT')
        self.vocab = vocab
            

        obs_shape = (d,)

        self.observation_space.spaces["image"] = SSPSpace(
            basis=[self.X,self.Y],
            radius=1,
            shape=(obs_shape[0],)
        )
        
        # delta = 4
        # S_rec = ssp.spatial_semantic_pointer.SpatialSemanticPointer(data=np.zeros(d))
        # for i in np.linspace(0,delta,50):
        #     for j in np.linspace(0,delta,50):
        #         S_rec += self.X**i * self.Y**j
        # S_rec= S_rec.normalized()
        # self.S_rec = S_rec
        self.delta = delta
        xi = np.linspace(0,delta,50)
        yi = np.linspace(-(delta//2),delta//2,50)
        xxi,yyi = np.meshgrid(xi,yi)
        Srecs = ssp.utils.ssp_vectorized(np.vstack([self.X.v, self.Y.v]).T, np.vstack([xxi.reshape(-1), yyi.reshape(-1)]).T)
        S_rec = ssp.SpatialSemanticPointer(data= np.sum(Srecs, axis=1))
        S_rec= S_rec.normalized()
        self.S_rec = S_rec

        
        
        xi = np.arange(img_shape[0] - 0.5,0,-1)*delta - (delta//2)
        yi = np.arange(img_shape[1]//2, -img_shape[1]//2, -1)*delta
        xx,yy = np.meshgrid(xi,yi)
        basisv = np.vstack([self.X.v, self.Y.v]).T
        positions = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T
        S_ids = ssp.utils.ssp_vectorized(basisv, positions).T.real
        S_list_rec = np.zeros(S_ids.shape)
        for i in np.arange(S_ids.shape[0]):
            S_list_rec[i,:] = (S_rec * ssp.SpatialSemanticPointer(data=S_ids[i,:])).normalized().v.real
        
        self.S_list = S_ids.reshape(len(xi),len(yi),d)
        self.S_list_rec = S_list_rec.reshape(len(xi),len(yi),d)

    def observation(self, obs):
        img = obs['image']
        
        # M = spa.SemanticPointer(data=np.zeros(self.d))
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         obj = img[i, j, 0]
        #         color = img[i, j, 1]
        #         state = img[i, j, 2]
        #         if obj not in [0,1]:                    
        #             #S = spa.SemanticPointer(data=self.S_list[i,j,:])
        #             S = spa.SemanticPointer(data=self.S_list_rec[i,j,:])
        #             M = M + ( S * self.vocab[IDX_TO_OBJECT[obj].upper()] * self.vocab[IDX_TO_COLOR[color].upper()] * self.vocab[IDX_TO_STATE[state].upper()])
       
        M = spa.SemanticPointer(data=np.zeros(self.d))
        allobjs = np.unique(img[:,:,0])
        # It doesn't save info about tiles that are 'unseen', 'emtpy', or floors
        allobjs = np.delete(allobjs,np.where((allobjs== 0) | (allobjs ==1) | (allobjs ==3)))
        for i in allobjs:
            obj = i
            idxs = np.where(img[:,:,0]==i)
            
            S = spa.SemanticPointer(data = np.sum(self.S_list_rec[idxs[0],idxs[1],:],axis=0))
            S = S.normalized()
            M = M + ( S * self.vocab[IDX_TO_OBJECT[obj].upper()])
        #M = M.normalized()
        
        #agent_ssp = self.X**(self.delta*self.agent_pos[0]) * self.Y**(self.delta*self.agent_pos[1])
        #M = M * agent_ssp
        return {
            'mission': obs['mission'],
            'image': M.v
        }
    
 
class SSPWrapper2(gym.core.ObservationWrapper):

    def __init__(self, env,d,X=None,Y=None,delta=2,rng=None):
        super().__init__(env)
        
        self.alg = spa.algebras.HrrAlgebra()

        img_shape = env.observation_space['image'].shape
        self.img_shape = img_shape
        self.d = d
        self.X = X or ssp.vector_generation.UnitaryVectors(d)
        self.Y = Y or ssp.vector_generation.UnitaryVectors(d)
        
        
        pointer_gen = spa.vector_generation.UnitaryVectors(self.d, self.alg, rng=rng)

        vocab = spa.Vocabulary(d, pointer_gen = pointer_gen)
        vocab.populate('POSITION;DIRECTION')
        self.vocab = vocab
            

        obs_shape = (d,)

        self.observation_space.spaces["image"] = SSPSpace(
            basis=[self.X,self.Y],
            radius=1,
            shape=(obs_shape[0],)
        )
        

        

    def observation(self, obs):
        #img = obs['image']
        
        agent_ssp = self.X**(self.agent_pos[0]) * self.Y**(self.agent_pos[1])
        dir_ssp = self.X**(DIR_TO_VEC[self.agent_dir][0]) * self.Y**(DIR_TO_VEC[self.agent_dir][1])

        M = (self.vocab['POSITION'] * agent_ssp) + (self.vocab['DIRECTION'] * dir_ssp)
        return {
            'mission': obs['mission'],
            'image': M.v
        }
       

class SSPGoalWrapper(SSPWrapper):
    """
    Provides the slope/angular direction to the goal with the observations as modeled by (y2 - y2 )/( x2 - x1)
    type = {slope , angle}
    """
    def __init__(self,  env,d,X=None,Y=None,delta=2,rng=None):
        super().__init__(env,d,X,Y,delta,rng)
        self.goal_position = None

    def reset(self):
        obs = self.env.reset()
        if not self.goal_position:
            self.goal_position = [x for x,y in enumerate(self.grid.grid) if isinstance(y,(Goal) ) ]
            if len(self.goal_position) >= 1: # in case there are multiple goals , needs to be handled for other env types
                self.goal_position = (int(self.goal_position[0]/self.height) , self.goal_position[0]%self.width)
        return obs

    def observation(self, obs):
        goal_ssp = self.X**self.goal_position[0] * self.Y**self.goal_position[1]
        agent_ssp = self.X**self.agent_pos[0] * self.Y**self.agent_pos[1]
        obs['goal_similarity'] = np.sum(goal_ssp.v.real * agent_ssp.v.real)
        return obs
    
class SSPSpace(gym.spaces.Space):
    def __init__(self, basis, radius, shape=None, dtype=complex, alg = spa.algebras.HrrAlgebra()):
        self.nbasis = len(basis)
        self.basis = basis
        self.dim = len(basis[0].v)
        self.shape = basis[0].v.shape
        self.radius = radius
        self.alg = alg
        self.seed()
        self.dist = ssp.dists.UniformSSPs(basis,self.alg,self.radius)

    def sample(self):   
        return self.dist.sample(1)

    
    def samples(self,n):
        return self.dist.sample(n)
              

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        # need more to assure its a real SSP - ie on right torus
        return ((type(x) == spa.SemanticPointer) | (type(x) == ssp.SpatialSemanticPointer) ) & (len(x) == self.dim)