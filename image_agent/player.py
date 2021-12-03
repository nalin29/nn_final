import numpy as np
import torch
from torch.serialization import load
import torchvision
import time
from PIL import Image
from os import path

from image_agent.models import load_model, Detector

GOALS = np.float32([[0, 75], [0, -75]])

LOST_STATUS_STEPS = 10
LOST_COOLDOWN_STEPS = 10
START_STEPS = 25
LAST_PUCK_DURATION = 4
MIN_SCORE = 0.2
MAX_DET = 15
MAX_DEV = 0.7
MIN_ANGLE = 20
MAX_ANGLE = 120
TARGET_SPEED = 15
STEER_YIELD = 15
DRIFT_THRESH = 0.7
TURN_CONE = 100

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')


def norm(vector):
    return np.linalg.norm(vector)


class Team:
    agent_type = 'image'

    def __init__(self):
        self.kart = 'wilber'
        self.initialize_vars()
        self.model = torch.load(path.join(path.dirname(path.abspath(__file__)), 'detector.pt')).to(device)
        self.model.eval()
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((128, 128)),
                                                         torchvision.transforms.ToTensor()])

    def new_match(self, team: int, num_players: int) -> list:
        self.team, self.num_players = team, num_players
        self.initialize_vars()
        print(f"Using {device} Match Started: {time.strftime('%H-%M-%S')}")
        return [self.kart] * num_players

    def initialize_vars(self):
        self.step = 0
        self.timer1 = 0

        self.puck_prev1 = 0
        self.last_seen1 = 0
        self.recover_steps1 = 0
        self.use_puck1 = True
        self.cooldown1 = 0

        self.timer2 = 0

        self.puck_prev2 = 0
        self.last_seen2 = 0
        self.recover_steps2 = 0
        self.use_puck2 = True
        self.cooldown2 = 0

    def act(self, player_state, player_image):

        player_info = player_state[0]
        image = player_image[0]

        # predict puck position
        img = self.transform(Image.fromarray(image)).to(device)
        pred = self.model.detect(img, max_pool_ks=7, min_score=MIN_SCORE, max_det=MAX_DET)
        puck_found = len(pred) > 0

        # try and detect if goal scored so we can reset (only needs to be done for one of the players)
        if norm(player_info['kart']['velocity']) < 1:
            if self.timer1 == 0:
                self.timer1 = self.step
            elif self.step - self.timer1 > 20:
                self.initialize_vars()
        else:
            self.timer1 = 0

        # get location in game and direct of kart
        front = np.float32(player_info['kart']['front'])[[0, 2]]
        loc = np.float32(player_info['kart']['location'])[[0, 2]]

        # execute when we find puck on screen
        if puck_found:
            # takes avg of peaks
            puck_loc = np.mean([cx[1] for cx in pred])
            puck_loc = puck_loc / 64 - 1

            # ignores puck detections whose change is too much so that we ignore bad detections
            if self.use_puck1 and np.abs(puck_loc - self.puck_prev1) > MAX_DEV:
                puck_loc = self.puck_prev1
                self.use_puck1 = False
            else:
                self.use_puck1 = True

            # update vars
            self.puck_prev1 = puck_loc
            self.last_seen1 = self.step
        # if puck not seen then use prev location or start lost actions
        elif self.step - self.last_seen1 < LAST_PUCK_DURATION:
            self.use_puck1 = False
            puck_loc = self.puck_prev1
        else:
            puck_loc = None
            self.recover_steps1 = LOST_STATUS_STEPS

        # calcualate direction vector
        dir = front - loc
        dir = dir / norm(dir)

        # calculate angle to own goal
        goal_dir = GOALS[self.team - 1] - loc
        dist_own_goal = norm(goal_dir)
        goal_dir = goal_dir / norm(goal_dir)

        goal_angle = np.arccos(np.clip(np.dot(dir, goal_dir), -1, 1))
        signed_own_goal_deg = np.degrees(
            -np.sign(np.cross(dir, goal_dir)) * goal_angle)

        # calculate angle to opp goal
        goal_dir = GOALS[self.team] - loc
        goal_dist = norm(goal_dir)
        goal_dir = goal_dir / np.linalg.norm(goal_dir)

        goal_angle = np.arccos(np.clip(np.dot(dir, goal_dir), -1, 1))
        signed_goal_angle = np.degrees(
            -np.sign(np.cross(dir, goal_dir)) * goal_angle)

        # restrict dist between [1,2] so we can use a weight function
        goal_dist = (
            (np.clip(goal_dist, 10, 100) - 10) / 90) + 1

        # set aim point if not cooldown or in recovery
        if (self.cooldown1 == 0 or puck_found) and self.recover_steps1 == 0:
            # if angle isn't extreme then weight our attack angle by dist
            if MIN_ANGLE < np.abs(signed_goal_angle) < MAX_ANGLE:
                distW = 1 / goal_dist ** 3
                aim_point = puck_loc + \
                    np.sign(puck_loc - signed_goal_angle /
                            TURN_CONE) * 0.3 * distW
            # if two tight then just chase puck
            else:
                aim_point = puck_loc
            # sets the speed as const if found
            if self.last_seen1 == self.step:
                brake = False
                acceleration = 0.75 if norm(
                    player_info['kart']['velocity']) < TARGET_SPEED else 0
            else:
                brake = False
                acceleration = 0
        # cooldown actions
        elif self.cooldown1 > 0:
            self.cooldown1 -= 1
            brake = False
            acceleration = 0.5
            aim_point = signed_goal_angle / TURN_CONE
        # recovery actions
        else:
            # if not a goal keep backing up
            if dist_own_goal > 10:
                aim_point = signed_own_goal_deg / TURN_CONE
                acceleration = 0
                brake = True
                self.recover_steps1 -= 1
            # if at goal then cooldown on reversing
            else:
                self.cooldown1 = LOST_COOLDOWN_STEPS
                aim_point = signed_goal_angle / TURN_CONE
                acceleration = 0.5
                brake = False
                self.recover_steps1 = 0

        # set steering/drift
        steer = np.clip(aim_point * STEER_YIELD, -1, 1)
        drift = np.abs(aim_point) > DRIFT_THRESH

        p1 = {
            'steer': signed_goal_angle if self.step < START_STEPS else steer,
            'acceleration': 1 if self.step < START_STEPS else acceleration,
            'brake': brake,
            'drift': drift,
            'nitro': False, 
            'rescue': False
        }

        # player 2 (same agent for now)

        player_info = player_state[1]
        image = player_image[1]

        img = self.transform(Image.fromarray(image)).to(device)
        pred = self.model.detect(
            img, max_pool_ks=7, min_score=MIN_SCORE, max_det=MAX_DET)

        front = np.float32(player_info['kart']['front'])[[0, 2]]
        loc = np.float32(player_info['kart']['location'])[[0, 2]]

        puck_found = len(pred) > 0
        if puck_found:
            puck_loc = np.mean([cx[1] for cx in pred])
            puck_loc = puck_loc / 64 - 1

            if self.use_puck2 and np.abs(puck_loc - self.puck_prev2) > MAX_DEV:
                puck_loc = self.puck_prev2
                self.use_puck2 = False
            else:
                self.use_puck2 = True

            self.puck_prev2 = puck_loc
            self.last_seen2 = self.step

        elif self.step - self.last_seen2 < LAST_PUCK_DURATION:
            self.use_puck2 = False
            puck_loc = self.puck_prev2
        else:
            puck_loc = None
            self.recover_steps2 = LOST_STATUS_STEPS

        dir = front - loc
        dir = dir / norm(dir)

        goal_dir = GOALS[self.team - 1] - loc
        dist_own_goal = norm(goal_dir)
        goal_dir = goal_dir / norm(goal_dir)

        goal_angle = np.arccos(np.clip(np.dot(dir, goal_dir), -1, 1))
        signed_own_goal_deg = np.degrees(
            -np.sign(np.cross(dir, goal_dir)) * goal_angle)

        goal_dir = GOALS[self.team] - loc
        goal_dist = norm(goal_dir)
        goal_dir = goal_dir / norm(goal_dir)

        goal_angle = np.arccos(np.clip(np.dot(dir, goal_dir), -1, 1))
        signed_goal_angle = np.degrees(
            -np.sign(np.cross(dir, goal_dir)) * goal_angle)

        goal_dist = (
            (np.clip(goal_dist, 10, 100) - 10) / 90) + 1
        if self.recover_steps2 == 0 and (self.cooldown2 == 0 or puck_found):
            if MIN_ANGLE < np.abs(signed_goal_angle) < MAX_ANGLE:
                distW = 1 / goal_dist ** 3
                aim_point = puck_loc + \
                    np.sign(puck_loc - signed_goal_angle /
                            TURN_CONE) * 0.3 * distW
            else:
                aim_point = puck_loc
            if self.last_seen2 == self.step:
                brake = False
                acceleration = 0.75 if norm(
                    player_info['kart']['velocity']) < TARGET_SPEED else 0
            else:
                acceleration = 0
                brake = False
        elif self.cooldown2 > 0:
            self.cooldown2 -= 1
            brake = False
            acceleration = 0.5
            aim_point = signed_goal_angle / TURN_CONE
        else:
            if dist_own_goal > 10:
                acceleration = 0
                brake = True
                aim_point = signed_own_goal_deg / TURN_CONE
                self.recover_steps2 -= 1
            else:
                self.cooldown2 = LOST_COOLDOWN_STEPS
                self.step_back = 0
                aim_point = signed_goal_angle / TURN_CONE
                acceleration = 0.5
                brake = False

        steer = np.clip(aim_point * STEER_YIELD, -1, 1)
        drift = np.abs(aim_point) > DRIFT_THRESH

        p2 = {
            'steer': signed_goal_angle if self.step < START_STEPS else steer,
            'acceleration': 1 if self.step < START_STEPS else acceleration,
            'brake': brake,
            'drift': drift,
            'nitro': False, 
            'rescue': False
        }

        self.step += 1

        return [p1, p2]
