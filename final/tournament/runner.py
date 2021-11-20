import logging
import numpy as np
from collections import namedtuple

TRACK_NAME = 'icy_soccer_field'
MAX_FRAMES = 1000
TIMEOUT_SLACK = 2   # seconds
TIMEOUT_STEP = 0.1  # seconds

RunnerInfo = namedtuple('RunnerInfo', ['agent_type', 'error', 'total_act_time'])


def to_native(o):
    # Super obnoxious way to hide pystk
    import pystk
    _type_map = {pystk.Camera.Mode: int,
                 pystk.Attachment.Type: int,
                 pystk.Powerup.Type: int,
                 float: float,
                 int: int,
                 list: list,
                 bool: bool,
                 str: str,
                 memoryview: np.array,
                 property: lambda x: None}

    def _to(v):
        if type(v) in _type_map:
            return _type_map[type(v)](v)
        else:
            return {k: _to(getattr(v, k)) for k in dir(v) if k[0] != '_'}
    return _to(o)


class AIRunner:
    agent_type = 'state'
    is_ai = True

    def new_match(self, team: int, num_players: int) -> list:
        pass

    def act(self, player_state, opponent_state, world_state):
        return []

    def info(self):
        return RunnerInfo('state', None, 0)


class TeamRunner:
    agent_type = 'state'
    _error = None
    _total_act_time = 0

    def __init__(self, agent_dir):
        try:
            import grader
        except ImportError:
            from . import grader

        self._error = None

        try:
            assignment = grader.load_assignment(agent_dir)
            if assignment is None:
                self._error = 'Failed to load submission.'
            else:
                self._team = assignment.Team()
        except Exception as e:
            self._error = 'Failed to load submission: {}'.format(str(e))
        self.agent_type = self._team.agent_type

    def new_match(self, team: int, num_players: int) -> list:
        self._total_act_time = 0
        self._error = None
        try:
            r = self._team.new_match(team, num_players)
            if isinstance(r, str) or isinstance(r, list) or r is None:
                return r
            self._error = 'new_match needs to return kart names as a str, list, or None. Got {!r}!'.format(r)
        except Exception as e:
            self._error = 'Failed to start new_match: {}'.format(str(e))
        return []

    def act(self, player_state, *args, **kwargs):
        from time import time
        t0 = time()
        try:
            r = self._team.act(player_state, *args, **kwargs)
        except Exception as e:
            self._error = 'Failed to act: {}'.format(str(e))
        else:
            self._total_act_time += time()-t0
            return r
        return []

    def info(self):
        return RunnerInfo(self.agent_type, self._error, self._total_act_time)


class MatchException(Exception):
    def __init__(self, score, msg1, msg2):
        self.score, self.msg1, self.msg2 = score, msg1, msg2


class Match:
    """
        Do not create more than one match per process (use ray to create more)
    """
    def __init__(self, use_graphics=False, logging_level=None):
        # DO this here so things work out with ray
        import pystk
        self._pystk = pystk
        if logging_level is not None:
            logging.basicConfig(level=logging_level)

        # Fire up pystk
        self._use_graphics = use_graphics
        if use_graphics:
            graphics_config = self._pystk.GraphicsConfig.hd()
            graphics_config.screen_width = 400
            graphics_config.screen_height = 300
        else:
            graphics_config = self._pystk.GraphicsConfig.none()

        self._pystk.init(graphics_config)

    def __del__(self):
        if hasattr(self, '_pystk') and self._pystk is not None and self._pystk.clean is not None:  # Don't ask why...
            self._pystk.clean()

    def _make_config(self, team_id, is_ai, kart):
        PlayerConfig = self._pystk.PlayerConfig
        controller = PlayerConfig.Controller.AI_CONTROL if is_ai else PlayerConfig.Controller.PLAYER_CONTROL
        return PlayerConfig(controller=controller, team=team_id, kart=kart)

    @classmethod
    def _r(cls, f):
        if hasattr(f, 'remote'):
            return f.remote
        if hasattr(f, '__call__'):
            if hasattr(f.__call__, 'remote'):
                return f.__call__.remote
        return f

    @staticmethod
    def _g(f):
        # print('_g', f)
        return f

    def _check(self, team1, team2, where, n_iter, timeout_slack, timeout_step):
        _, error, t1 = self._g(self._r(team1.info)())
        if error:
            raise MatchException([0, 3], 'other team crashed', 'crash during {}: {}'.format(where, error))

        _, error, t2 = self._g(self._r(team2.info)())
        if error:
            raise MatchException([3, 0], 'crash during {}: {}'.format(where, error), 'other team crashed')

        logging.debug('timeout {} <? {} {}'.format(timeout_slack + n_iter * timeout_step, t1, t2))

        if max(t1, t2) > timeout_slack + n_iter * timeout_step:
            if t1 > t2:
                # Team 2 wins because of a timeout
                return [0, 3], 'Timeout ({:.4f}/iter > {:.4f}/iter)'.format(t1 / n_iter, timeout_step),\
                       'other team timed out'
            else:
                # Team 1 wins because of a timeout
                return [3, 0], 'other team timed out',\
                       'Timeout ({:.4f}/iter > {:.4f}/iter)'.format(t2 / n_iter, timeout_step)

    def run(self, team1, team2, num_player=1, max_frames=MAX_FRAMES, max_score=3, record_fn=None,
            timeout_slack=TIMEOUT_SLACK, timeout_step=TIMEOUT_STEP, initial_ball_location=[0, 0],
            initial_ball_velocity=[0, 0]):
        RaceConfig = self._pystk.RaceConfig

        logging.info('Creating teams')

        # Start a new match
        t1_cars = self._g(self._r(team1.new_match)(0, num_player)) or ['tux']
        t2_cars = self._g(self._r(team2.new_match)(1, num_player)) or ['tux']

        t1_type, *_ = self._g(self._r(team1.info()))
        t2_type, *_ = self._g(self._r(team2.info()))

        if t1_type == 'image' or t2_type == 'image':
            assert self._use_graphics, 'Need to use_graphics for image agents.'

        # Deal with crashes
        self._check(team1, team2, 'new_match', 0, timeout_slack, timeout_step)

        # Setup the race config
        logging.info('Setting up race')

        race_config = RaceConfig(track=TRACK_NAME, mode=RaceConfig.RaceMode.SOCCER, num_kart=2 * num_player)
        race_config.players.pop()
        for i in range(num_player):
            race_config.players.append(self._make_config(0, hasattr(team1, 'is_ai') and team1.is_ai, t1_cars[i % len(t1_cars)]))
            race_config.players.append(self._make_config(1, hasattr(team2, 'is_ai') and team2.is_ai, t2_cars[i % len(t2_cars)]))

        # Start the match
        logging.info('Starting race')
        race = self._pystk.Race(race_config)
        race.start()
        race.step()

        state = self._pystk.WorldState()
        state.update()
        state.set_ball_location((initial_ball_location[0], 1, initial_ball_location[1]),
                                (initial_ball_velocity[0], 0, initial_ball_velocity[1]))

        for it in range(max_frames):
            logging.debug('iteration {} / {}'.format(it, MAX_FRAMES))
            state.update()

            # Get the state
            team1_state = [to_native(p) for p in state.players[0::2]]
            team2_state = [to_native(p) for p in state.players[1::2]]
            soccer_state = to_native(state.soccer)
            team1_images = team2_images = None
            if self._use_graphics:
                team1_images = [np.array(race.render_data[i].image) for i in range(0, len(race.render_data), 2)]
                team2_images = [np.array(race.render_data[i].image) for i in range(1, len(race.render_data), 2)]

            # Have each team produce actions (in parallel)
            if t1_type == 'image':
                team1_actions_delayed = self._r(team1.act)(team1_state, team1_images)
            else:
                team1_actions_delayed = self._r(team1.act)(team1_state, team2_state, soccer_state)

            if t2_type == 'image':
                team2_actions_delayed = self._r(team2.act)(team2_state, team2_images)
            else:
                team2_actions_delayed = self._r(team2.act)(team2_state, team1_state, soccer_state)

            # Wait for the actions to finish
            team1_actions = self._g(team1_actions_delayed)
            team2_actions = self._g(team2_actions_delayed)

            self._check(team1, team2, 'act', it, timeout_slack, timeout_step)

            # Assemble the actions
            actions = []
            for i in range(num_player):
                a1 = team1_actions[i] if team1_actions is not None and i < len(team1_actions) else {}
                a2 = team2_actions[i] if team2_actions is not None and i < len(team2_actions) else {}
                actions.append(a1)
                actions.append(a2)

            if record_fn:
                self._r(record_fn)(team1_state, team2_state, soccer_state=soccer_state, actions=actions,
                                   team1_images=team1_images, team2_images=team2_images)

            logging.debug('  race.step  [score = {}]'.format(state.soccer.score))
            if (not race.step([self._pystk.Action(**a) for a in actions]) and num_player) or sum(state.soccer.score) >= max_score:
                break

        race.stop()
        del race

        return state.soccer.score

    def wait(self, x):
        return x


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path
    from os import environ
    from . import remote, utils

    parser = ArgumentParser(description="Play some Ice Hockey. List any number of players, odd players are in team 1, even players team 2.")
    parser.add_argument('-r', '--record_video', help="Do you want to record a video?")
    parser.add_argument('-s', '--record_state', help="Do you want to pickle the state?")
    parser.add_argument('-f', '--num_frames', default=1000, type=int, help="How many steps should we play for?")
    parser.add_argument('-p', '--num_players', default=2, type=int, help="Number of players per team")
    parser.add_argument('-m', '--max_score', default=3, type=int, help="How many goal should we play to?")
    parser.add_argument('-j', '--parallel', type=int, help="How many parallel process to use?")
    parser.add_argument('--ball_location', default=[0, 0], type=float, nargs=2, help="Initial xy location of ball")
    parser.add_argument('--ball_velocity', default=[0, 0], type=float, nargs=2, help="Initial xy velocity of ball")
    parser.add_argument('team1', help="Python module name or `AI` for AI players.")
    parser.add_argument('team2', help="Python module name or `AI` for AI players.")
    args = parser.parse_args()

    logging.basicConfig(level=environ.get('LOGLEVEL', 'WARNING').upper())

    if args.parallel is None or remote.ray is None:
        # Create the teams
        team1 = AIRunner() if args.team1 == 'AI' else TeamRunner(args.team1)
        team2 = AIRunner() if args.team2 == 'AI' else TeamRunner(args.team2)

        # What should we record?
        recorder = None
        if args.record_video:
            recorder = recorder & utils.VideoRecorder(args.record_video)

        if args.record_state:
            recorder = recorder & utils.StateRecorder(args.record_state)

        # Start the match
        match = Match(use_graphics=team1.agent_type == 'image' or team2.agent_type == 'image')
        try:
            result = match.run(team1, team2, args.num_players, args.num_frames, max_score=args.max_score,
                               initial_ball_location=args.ball_location, initial_ball_velocity=args.ball_velocity,
                               record_fn=recorder)
        except MatchException as e:
            print('Match failed', e.score)
            print(' T1:', e.msg1)
            print(' T2:', e.msg2)

        print('Match results', result)

    else:
        # Fire up ray
        remote.init(logging_level=getattr(logging, environ.get('LOGLEVEL', 'WARNING').upper()), configure_logging=True,
                    log_to_driver=True, include_dashboard=False)

        # Create the teams
        team1 = AIRunner() if args.team1 == 'AI' else remote.RayTeamRunner.remote(args.team1)
        team2 = AIRunner() if args.team2 == 'AI' else remote.RayTeamRunner.remote(args.team2)

        # What should we record?
        assert args.record_state is None or args.record_video is None, "Cannot record both video and state in parallel mode"

        # Start the match
        results = []
        for i in range(args.parallel):
            recorder = None
            if args.record_video:
                ext = Path(args.record_video).suffix
                recorder = remote.RayVideoRecorder.remote(args.record_video.replace(ext, f'.{i}{ext}'))
            elif args.record_state:
                ext = Path(args.record_state).suffix
                recorder = remote.RayStateRecorder.remote(args.record_state.replace(ext, f'.{i}{ext}'))

            match = remote.RayMatch.remote(logging_level=getattr(logging, environ.get('LOGLEVEL', 'WARNING').upper()),
                                           use_graphics=team1.agent_type == 'image' or team2.agent_type == 'image')
            result = match.run.remote(team1, team2, args.num_players, args.num_frames, max_score=args.max_score,
                                      initial_ball_location=args.ball_location,
                                      initial_ball_velocity=args.ball_velocity,
                                      record_fn=recorder)
            results.append(result)

        for result in results:
            try:
                result = remote.get(result)
            except (remote.RayMatchException, MatchException) as e:
                print('Match failed', e.score)
                print(' T1:', e.msg1)
                print(' T2:', e.msg2)

            print('Match results', result)
