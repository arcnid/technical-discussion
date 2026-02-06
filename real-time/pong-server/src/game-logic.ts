// Pure functions for game logic - NO MUTATION, NO SIDE EFFECTS

import type { Ball, GameState, Paddle } from './types.js';
import { GAME_CONSTANTS } from './types.js';

// Pure function: Create new game state
export const createGameState = (): GameState => ({
  ball: {
    x: GAME_CONSTANTS.COURT_WIDTH / 2,
    y: GAME_CONSTANTS.COURT_HEIGHT / 2,
    dx: GAME_CONSTANTS.BALL_SPEED * (Math.random() > 0.5 ? 1 : -1),
    dy: GAME_CONSTANTS.BALL_SPEED * (Math.random() > 0.5 ? 1 : -1),
  },
  p1: { y: GAME_CONSTANTS.COURT_HEIGHT / 2 - GAME_CONSTANTS.PADDLE_HEIGHT / 2 },
  p2: { y: GAME_CONSTANTS.COURT_HEIGHT / 2 - GAME_CONSTANTS.PADDLE_HEIGHT / 2 },
  p1Score: 0,
  p2Score: 0,
  status: 'waiting',
  playersJoined: new Set(),
  servingPlayer: null,  // No serving
});

// Pure function: Update paddle position
export const updatePaddle = (
  state: GameState,
  player: 1 | 2,
  y: number
): GameState => {
  if (player === 1) {
    return { ...state, p1: { ...state.p1, y } };
  }
  return { ...state, p2: { ...state.p2, y } };
};

// Pure function: Check wall collision and return new ball velocity
const checkWallCollision = (ball: Ball): { dy: number; y: number } => {
  if (ball.y <= 0) {
    // Hit top wall - bounce down (positive dy)
    return { dy: Math.abs(ball.dy), y: 0 };
  }
  if (ball.y >= GAME_CONSTANTS.COURT_HEIGHT - 1) {
    // Hit bottom wall - bounce up (negative dy)
    return { dy: -Math.abs(ball.dy), y: GAME_CONSTANTS.COURT_HEIGHT - 1 };
  }
  return { dy: ball.dy, y: ball.y };
};

// Pure function: Check paddle collision
const checkPaddleCollision = (
  ball: Ball,
  p1Y: number,
  p2Y: number
): { dx: number; x: number } => {
  const ballRounded = Math.round(ball.y);
  const p1YRounded = Math.round(p1Y);
  const p2YRounded = Math.round(p2Y);

  // P1 paddle (left side at x=1)
  // Ball moving LEFT (dx < 0) toward left wall - check for paddle hit
  if (ball.x <= 2.0 && ball.dx < 0) {
    if (ballRounded >= p1YRounded && ballRounded < p1YRounded + GAME_CONSTANTS.PADDLE_HEIGHT) {
      // Hit paddle! Bounce ball back to the RIGHT
      return { dx: Math.abs(ball.dx), x: 2.0 };
    }
  }

  // P2 paddle (right side at x=38, which is COURT_WIDTH-2)
  // Ball moving RIGHT (dx > 0) toward right wall - check for paddle hit
  if (ball.x >= GAME_CONSTANTS.COURT_WIDTH - 3 && ball.dx > 0) {
    if (ballRounded >= p2YRounded && ballRounded < p2YRounded + GAME_CONSTANTS.PADDLE_HEIGHT) {
      // Hit paddle! Bounce ball back to the LEFT
      return { dx: -Math.abs(ball.dx), x: GAME_CONSTANTS.COURT_WIDTH - 3 };
    }
  }

  return { dx: ball.dx, x: ball.x };
};

// Pure function: Reset ball to center (stationary, ready to serve)
export const resetBall = (): Ball => ({
  x: GAME_CONSTANTS.COURT_WIDTH / 2,
  y: GAME_CONSTANTS.COURT_HEIGHT / 2,
  dx: 0,  // Stationary until served
  dy: 0,
});

// Pure function: Update ball physics (returns new ball or scoring event)
export const updateBall = (
  ball: Ball,
  p1Y: number,
  p2Y: number
): { ball: Ball; scored: null | 1 | 2 } => {
  // Calculate new position
  let newX = ball.x + ball.dx;
  let newY = ball.y + ball.dy;
  let newDx = ball.dx;
  let newDy = ball.dy;

  // Check wall collision
  const wallResult = checkWallCollision({ ...ball, y: newY, dy: newDy });
  newY = wallResult.y;
  newDy = wallResult.dy;

  // Check paddle collision
  const paddleResult = checkPaddleCollision({ ...ball, x: newX, dx: newDx }, p1Y, p2Y);
  newX = paddleResult.x;
  newDx = paddleResult.dx;

  // Check scoring
  if (newX < 0) {
    return { ball: resetBall(), scored: 2 }; // P2 scores
  }
  if (newX > GAME_CONSTANTS.COURT_WIDTH) {
    return { ball: resetBall(), scored: 1 }; // P1 scores
  }

  // Return new ball state (no scoring)
  return {
    ball: { x: newX, y: newY, dx: newDx, dy: newDy },
    scored: null,
  };
};

// Pure function: Update score
export const updateScore = (
  state: GameState,
  scoringPlayer: 1 | 2
): GameState => {
  if (scoringPlayer === 1) {
    return { ...state, p1Score: state.p1Score + 1 };
  }
  return { ...state, p2Score: state.p2Score + 1 };
};

// Pure function: Add player to game
export const addPlayer = (state: GameState, player: 1 | 2): GameState => {
  const newPlayersJoined = new Set(state.playersJoined);
  newPlayersJoined.add(player);
  return {
    ...state,
    playersJoined: newPlayersJoined,
  };
};

// Pure function: Check if both players have joined
export const bothPlayersJoined = (state: GameState): boolean => {
  return state.playersJoined.has(1) && state.playersJoined.has(2);
};

// Pure function: Start game
export const startGame = (state: GameState): GameState => ({
  ...state,
  status: 'playing',
});

// Pure function: Reset game (for restart)
export const resetGame = (state: GameState): GameState => ({
  ...createGameState(),
  playersJoined: state.playersJoined,  // Keep the players joined
  status: 'playing',  // Start immediately since players are already here
});

// Pure function: Attach ball to serving player's paddle
export const attachBallToPaddle = (state: GameState): GameState => {
  if (state.servingPlayer === null) return state;

  const paddleY = state.servingPlayer === 1 ? state.p1.y : state.p2.y;
  const ballX = state.servingPlayer === 1 ? 3 : GAME_CONSTANTS.COURT_WIDTH - 4;

  return {
    ...state,
    ball: {
      x: ballX,
      y: paddleY + GAME_CONSTANTS.PADDLE_HEIGHT / 2,  // Center of paddle
      dx: 0,
      dy: 0,
    },
  };
};

// Pure function: Serve the ball (release from paddle)
export const serveBall = (state: GameState): GameState => {
  if (state.servingPlayer === null) return state;

  const direction = state.servingPlayer === 1 ? 1 : -1;  // Serve toward opponent

  return {
    ...state,
    ball: {
      ...state.ball,
      dx: GAME_CONSTANTS.BALL_SPEED * direction,
      dy: GAME_CONSTANTS.BALL_SPEED * (Math.random() > 0.5 ? 1 : -1),
    },
    servingPlayer: null,  // Ball is now in play
  };
};

// Pure function: Set serving player after a point
export const setServingPlayer = (state: GameState, scoringPlayer: 1 | 2): GameState => {
  // The player who got scored ON serves next (like real Pong)
  const servingPlayer = scoringPlayer === 1 ? 2 : 1;
  return {
    ...state,
    servingPlayer,
  };
};

// Pure function: End game
export const endGame = (state: GameState): GameState => ({
  ...state,
  status: 'ended',
});

// Pure function: Check if game should end
export const shouldEndGame = (state: GameState): boolean => {
  return (
    state.p1Score >= GAME_CONSTANTS.WINNING_SCORE ||
    state.p2Score >= GAME_CONSTANTS.WINNING_SCORE
  );
};

// Pure function: Game tick (main update function)
export const gameTick = (state: GameState): GameState => {
  if (state.status !== 'playing') {
    return state;
  }

  // Update ball physics
  const { ball, scored } = updateBall(state.ball, state.p1.y, state.p2.y);

  // Handle scoring
  if (scored !== null) {
    let newState = updateScore(state, scored);
    // Reset ball with random direction
    const newBall = {
      x: GAME_CONSTANTS.COURT_WIDTH / 2,
      y: GAME_CONSTANTS.COURT_HEIGHT / 2,
      dx: GAME_CONSTANTS.BALL_SPEED * (Math.random() > 0.5 ? 1 : -1),
      dy: GAME_CONSTANTS.BALL_SPEED * (Math.random() > 0.5 ? 1 : -1),
    };
    newState = { ...newState, ball: newBall };

    // Check if game should end
    if (shouldEndGame(newState)) {
      return endGame(newState);
    }

    return newState;
  }

  // Just update ball position
  return { ...state, ball };
};
