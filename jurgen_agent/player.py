from os import path 
import numpy as np 
import torch 

def limit_period(angle):
    # turn angle into -1 to 1 
    return angle - torch.floor(angle / 2 + 0.5) * 2 

def extract_featuresV2(pstate, soccer_state, opponent_state, team_id):
    # features of ego-vehicle
    kart_front = torch.tensor(pstate['kart']['front'], dtype=torch.float32)[[0, 2]]
    kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
    kart_direction = (kart_front-kart_center) / torch.norm(kart_front-kart_center)
    kart_angle = torch.atan2(kart_direction[1], kart_direction[0])

    # features of soccer 
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center-kart_center)
    kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0]) 

    kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle)/np.pi)

    # features of score-line 
    goal_line_center = torch.tensor(soccer_state['goal_line'][(team_id+1)%2], dtype=torch.float32)[:, [0, 2]].mean(dim=0)

    puck_to_goal_line = (goal_line_center-puck_center) / torch.norm(goal_line_center-puck_center)

    features = torch.tensor([kart_center[0], kart_center[1], kart_angle, kart_to_puck_angle, 
        goal_line_center[0], goal_line_center[1], kart_to_puck_angle_difference, 
        puck_center[0], puck_center[1], puck_to_goal_line[0], puck_to_goal_line[1]], dtype=torch.float32)

    return features 

class Team:
    agent_type = 'state'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None
        self.model = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), 'jurgen_agent.pt'))

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
        return ['sara_the_racer'] * num_players

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
        actions = [] 
        for player_id, pstate in enumerate(player_state):
            features = extract_featuresV2(pstate, soccer_state, opponent_state, self.team)
            acceleration, steer, brake = self.model(features)
            actions.append(dict(acceleration=acceleration, steer=steer, brake=brake))                        
        return actions 
