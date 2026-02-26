export interface Ball {
  x: number;
  y: number;
  dx: number;
  dy: number;
}

export interface Paddle {
  y: number;
}

export interface GameState {
  ball: Ball;
  p1: Paddle;
  p2: Paddle;
  p1Score: number;
  p2Score: number;
  status: 'waiting' | 'playing' | 'ended';
  playersJoined: Set<1 | 2>;  // Track which players have joined
  servingPlayer: 1 | 2 | null;  // Who's serving (ball attached to their paddle)
  playersReady: Set<1 | 2>;  // Track which players are ready (for post-game restart)
}

export interface PaddleUpdate {
  y: number;
  timestamp: number;
}

export interface BallUpdate {
  x: number;
  y: number;
  dx: number;
  dy: number;
  timestamp: number;
}

export interface StateUpdate {
  status: 'waiting' | 'playing' | 'ended';
  p1Score: number;
  p2Score: number;
  servingPlayer: 1 | 2 | null;
  timestamp: number;
}

export const GAME_CONSTANTS = {
  COURT_WIDTH: 40,
  COURT_HEIGHT: 12,
  PADDLE_HEIGHT: 3,
  BALL_SPEED: 0.3,  // Slightly slower for better playability
  FRAME_RATE: 60,
  WINNING_SCORE: 5,
};
