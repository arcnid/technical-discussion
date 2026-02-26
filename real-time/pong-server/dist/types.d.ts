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
    playersJoined: Set<1 | 2>;
    servingPlayer: 1 | 2 | null;
    playersReady: Set<1 | 2>;
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
export declare const GAME_CONSTANTS: {
    COURT_WIDTH: number;
    COURT_HEIGHT: number;
    PADDLE_HEIGHT: number;
    BALL_SPEED: number;
    FRAME_RATE: number;
    WINNING_SCORE: number;
};
//# sourceMappingURL=types.d.ts.map