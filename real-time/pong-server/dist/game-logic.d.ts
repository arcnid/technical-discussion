import type { Ball, GameState } from './types.js';
export declare const createGameState: () => GameState;
export declare const updatePaddle: (state: GameState, player: 1 | 2, y: number) => GameState;
export declare const resetBall: () => Ball;
export declare const updateBall: (ball: Ball, p1Y: number, p2Y: number) => {
    ball: Ball;
    scored: null | 1 | 2;
};
export declare const updateScore: (state: GameState, scoringPlayer: 1 | 2) => GameState;
export declare const addPlayer: (state: GameState, player: 1 | 2) => GameState;
export declare const bothPlayersJoined: (state: GameState) => boolean;
export declare const startGame: (state: GameState) => GameState;
export declare const resetGame: (state: GameState) => GameState;
export declare const attachBallToPaddle: (state: GameState) => GameState;
export declare const serveBall: (state: GameState) => GameState;
export declare const setServingPlayer: (state: GameState, scoringPlayer: 1 | 2) => GameState;
export declare const endGame: (state: GameState) => GameState;
export declare const shouldEndGame: (state: GameState) => boolean;
export declare const gameTick: (state: GameState) => GameState;
//# sourceMappingURL=game-logic.d.ts.map