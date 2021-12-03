from .runner import TeamRunner, Match, MatchException
from .grader import Grader, Case


STEPS_PER_MATCH = 1200
MAX_TIME_IMAGE = 0.05 * STEPS_PER_MATCH
MAX_TIME_STATE = 0.01 * STEPS_PER_MATCH


class HockyRunner(TeamRunner):
    """
        Similar to TeamRunner but this module takes Team object as inputs instead of the path to module
    """
    def __init__(self, team):
        self._team = team
        self.agent_type = self._team.agent_type


class FinalGrader(Grader):
    """Match against Instructor/TA's agents"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.student_model = HockyRunner(self.module.Team())
        self.match = Match(use_graphics=self.student_model.agent_type == 'image')

    def _test(self, agent_name):
        time_limit = MAX_TIME_STATE if self.student_model.agent_type == 'state' else MAX_TIME_IMAGE

        test_model = TeamRunner(agent_name)
        ball_locations = [
            [0, 1],
            [0, -1],
            [1, 0],
            [-1, 0],
        ]
        scores = []
        results = []

        try:
            for bl in ball_locations:
                result = self.match.run(self.student_model, test_model, 2, STEPS_PER_MATCH, max_score=3,
                                   initial_ball_location=bl, initial_ball_velocity=[0, 0],
                                   record_fn=None, timeout=time_limit, verbose=self.verbose)
                scores.append(result[0])
                results.append(f'{result[0]}:{result[1]}')

            for bl in ball_locations:
                result = self.match.run(test_model, self.student_model, 2, STEPS_PER_MATCH, max_score=3,
                                   initial_ball_location=bl, initial_ball_velocity=[0, 0],
                                   record_fn=None, timeout=time_limit, verbose=self.verbose)
                scores.append(result[1])
                results.append(f'{result[1]}:{result[0]}')
        except MatchException as e:
            print('Match failed', e.score)
            print(' T1:', e.msg1)
            print(' T2:', e.msg2)
            assert 0
        return sum(scores), results

    @Case(score=25)
    def test_geoffrey(self):
        """geoffrey agent"""
        scores, results = self._test('geoffrey_agent')
        return min(scores / len(results), 1), "{} goals scored in {} games ({})".format(scores, len(results), '  '.join(results))

    @Case(score=25)
    def test_yann(self):
        """yann agent"""
        scores, results = self._test('yann_agent')
        return min(scores / len(results), 1), "{} goals scored in {} games ({})".format(scores, len(results), '  '.join(results))

    @Case(score=25)
    def test_yoshua(self):
        """yoshua agent"""
        scores, results = self._test('yoshua_agent')
        return min(scores / len(results), 1), "{} goals scored in {} games ({})".format(scores, len(results), '  '.join(results))

    @Case(score=25)
    def test_jurgen(self):
        """jurgen agent"""
        if self.student_model.agent_type == 'state':
            scores, results = self._test('jurgen_agent')
        else:
            scores, results = self._test('image_jurgen_agent')
        return min(scores / len(results), 1), "{} goals scored in {} games ({})".format(scores, len(results), '  '.join(results))
