"""Read visible Pong objects and project the ball's next interception."""

import numpy as np


class VisualCoach:
    """The coach has access to the same screen as the agent, plus the previous screen."""

    def __init__(self, aim_offset=5.0):
        self.aim_offset = aim_offset
        self.previous_ball = None

    def reset(self):
        self.previous_ball = None

    @staticmethod
    def reflect(value, low, high):
        # Unfold the ball's path into a straight line, then fold it at each wall.
        height = high - low
        return low + height - abs((value - low) % (2 * height) - height)

    def observe(self, rgb):
        # The installed Pong ROM uses distinct fixed colours. Crop out the scoreboard
        # and the white walls so their pixels never enter the object measurements.
        red = rgb[34:194, :, 0]
        paddle_rows = np.flatnonzero(np.any(red[:, 140:144] == 92, axis=1))
        opponent_rows = np.flatnonzero(np.any(red[:, 16:20] == 213, axis=1))
        paddle_y = float(paddle_rows.mean() + 34) if len(paddle_rows) else 113.5
        opponent_y = float(opponent_rows.mean() + 34) if len(opponent_rows) else 113.5
        ball_rows, ball_cols = np.nonzero(red[:, 5:155] == 236)
        ball = (float(ball_cols.mean() + 5), float(ball_rows.mean() + 34)) if len(ball_rows) else None

        # Re-centre between rallies. A large apparent jump is a serve/reset, not velocity.
        goal = 113.5
        x, y, dx, dy = 80.0, 113.5, 0.0, 0.0
        if ball is not None:
            x, y = ball
            if self.previous_ball is not None:
                dx = x - self.previous_ball[0]
                dy = y - self.previous_ball[1]
                if abs(dx) > 20 or abs(dy) > 30:
                    dx = dy = 0.0
            if dx > 0 and x <= 139:
                remaining = (138.5 - x) / dx
                goal = self.reflect(y + remaining * dy, 34.5, 192.5)
                # Hit slightly off centre to vary the bounce away from the opponent.
                if remaining > 1.0 or abs(goal - paddle_y) < 6.0:
                    goal += self.aim_offset if opponent_y > 113.5 else -self.aim_offset
                goal = float(np.clip(goal, 34.5, 192.5))
        self.previous_ball = ball
        error = goal - paddle_y
        cues = np.array([error / 80, (paddle_y - 113.5) / 80, (x - 80) / 80,
                         (y - 113.5) / 80, dx / 12, dy / 12,
                         (opponent_y - 113.5) / 80, float(ball is not None)], dtype=np.float32)
        return cues


def demonstration_action(cues):
    """Training label: choose the direction that reduces the interception error."""
    # ALE calls the up/down paddle actions RIGHT/LEFT in this Pong ROM.
    tolerance = np.float32(3 / 80)
    if cues[0] < -tolerance:
        return 2
    if cues[0] > tolerance:
        return 3
    return 0
