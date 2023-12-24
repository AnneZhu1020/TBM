from gym_minigrid.envs.multiroom import MultiRoomEnv
from gym_minigrid.register import register


class MultiRoomEnvN7S4(MultiRoomEnv):
    def __init__(self):
        super().__init__(minNumRooms=7, maxNumRooms=7, maxRoomSize=4)


class MultiRoomEnvN7S8(MultiRoomEnv):
    def __init__(self):
        super().__init__(minNumRooms=7, maxNumRooms=7, maxRoomSize=8)


class MultiRoomEnvN10S4(MultiRoomEnv):
    def __init__(self):
        super().__init__(minNumRooms=10, maxNumRooms=10, maxRoomSize=4)


class MultiRoomEnvN10S10(MultiRoomEnv):
    def __init__(self):
        super().__init__(minNumRooms=10, maxNumRooms=10, maxRoomSize=10)


class MultiRoomEnvN12S10(MultiRoomEnv):
    def __init__(self):
        super().__init__(minNumRooms=12, maxNumRooms=12, maxRoomSize=10)


register("MiniGrid-MultiRoomN7S4-v0", entry_point="minigrid_env:MultiRoomEnvN7S4")
register("MiniGrid-MultiRoomN7S8-v0", entry_point="minigrid_env:MultiRoomEnvN7S8")
register("MiniGrid-MultiRoomN10S4-v0", entry_point="minigrid_env:MultiRoomEnvN10S4")
register("MiniGrid-MultiRoomN10S10-v0", entry_point="minigrid_env:MultiRoomEnvN10S10")
register("MiniGrid-MultiRoomN12S10-v0", entry_point="minigrid_env:MultiRoomEnvN12S10")
