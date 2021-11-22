import numpy as np

class Team:
    agent_type = 'state'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None
        self.player1 = {}
        self.player2 = {}

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        self.player1['count'] = 0
        self.player2['count'] = 0

        self.player1['recover'] = 0
        self.player2['recover'] = 0

        return ['tux'] * num_players

    def act(self, player_state, opponent_state, soccer_state):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             You can ignore the camera here.
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param opponent_state: same as player_state just for other team

        :param soccer_state: dict  Mostly used to obtain the puck location
                             ball:  Puck information
                               - location: float3 world location of the puck

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        # TODO: Change me. I'm just cruising straight
        actions = [dict()] * self.num_players

        player1State = player_state[0]['kart']
        player2State = player_state[1]['kart']
 

        puck = soccer_state['ball']['location']
        #print(player1State['velocity'])
        cur_speed = abs(player1State['velocity'][0])

        
        if cur_speed < 0.2 and not self.player1['recover']:
          self.player1['count'] += 1
          if self.player1['count'] > 15:
            self.player1['recover'] = 30
            self.player1['count'] = 0
        else:
          self.player1['count'] = 0

        brake = 0
        acceleration = 0
        if self.player1['recover']:
          brake = 1
          self.player1['recover'] -= 1
        else:
          acceleration = 1 if cur_speed < 5 else 0

        print(cur_speed)

        aim_point = puck
        vector_to_point = puck - np.array(player1State['location'])
        front_dir = np.array(player1State['front']) - np.array(player1State['location'])
        front_dir = front_dir/max(np.linalg.norm(front_dir), 1e-10)
        left_dir = np.cross([0, 1, 0], front_dir)
        steer = left_dir.dot(vector_to_point)

        drift = 1 if steer > 0.5 else 0

        actions[0]['acceleration'] = acceleration
        actions[0]['steer'] = steer
        #actions[0]['drift'] = drift
        actions[0]['brake'] = brake

        cur_speed = abs(player2State['velocity'][0])

        acceleration = 1 if cur_speed < 5 else 0

        aim_point = puck
        vector_to_point = puck - np.array(player2State['location'])
        front_dir = np.array(player2State['front']) - np.array(player2State['location'])
        front_dir = front_dir/max(np.linalg.norm(front_dir), 1e-10)
        left_dir = np.cross([0, 1, 0], front_dir)
        steer = left_dir.dot(vector_to_point)

        drift = 1 if steer > 0.5 else 0

        actions[1]['acceleration'] = acceleration
        actions[1]['steer'] = steer
        actions[1]['drift'] = drift

        return actions
