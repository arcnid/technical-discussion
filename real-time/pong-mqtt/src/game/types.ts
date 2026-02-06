export type PlayerNumber = 1 | 2;

export type GameStatus = 'waiting' | 'playing' | 'paused' | 'ended';

export interface Ball {
  x: number;
  y: number;
  dx: number;
  dy: number;
}

export interface Player {
  y: number;
  score: number;
}

export interface GameState {
  ball: Ball;
  p1: Player;
  p2: Player;
  gameId: string;
  status: GameStatus;
  timestamp: number;
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
  status: GameStatus;
  p1Score: number;
  p2Score: number;
  servingPlayer: 1 | 2 | null;
  timestamp: number;
}

export interface GameConfig {
  gameId: string;
  playerNumber: PlayerNumber;
  isHost: boolean;
  brokerUrl: string;
}

export const GAME_CONSTANTS = {
  COURT_WIDTH: 40,  // Lower resolution for better performance
  COURT_HEIGHT: 12,
  PADDLE_HEIGHT: 3,
  BALL_SPEED: 0.5,  // Smooth, visible speed
  PADDLE_SPEED: 1.0,  // Faster paddle response
  FRAME_RATE: 60,
  RENDER_FPS: 60,  // Smooth 60fps rendering
  WINNING_SCORE: 5,
};
