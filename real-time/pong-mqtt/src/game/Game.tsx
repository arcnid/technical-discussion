import React, { useState, useEffect, useMemo } from 'react';
import { Box, Text, useInput, useApp } from 'ink';
import type { GameConfig, Ball, Player, PaddleUpdate, BallUpdate, StateUpdate } from './types';
import { GAME_CONSTANTS } from './types';
import type { FunctionalMqttClient } from '../mqtt/client';
import { topics, parseTopic } from '../mqtt/topics';

interface GameProps {
  config: GameConfig;
  mqttClient: FunctionalMqttClient;
}

// Immutable game state (server is authoritative)
interface GameDisplayState {
  readonly ball: Ball;
  readonly p1: Player;
  readonly p2: Player;
  readonly status: 'waiting' | 'playing' | 'ended';
  readonly servingPlayer: 1 | 2 | null;  // Who's currently serving
}

// Pure function: Create initial state
const createInitialState = (): GameDisplayState => {
  const initialBall = {
    x: GAME_CONSTANTS.COURT_WIDTH / 2,
    y: GAME_CONSTANTS.COURT_HEIGHT / 2,
    dx: 0,
    dy: 0,
  };
  const initialPaddle = {
    y: GAME_CONSTANTS.COURT_HEIGHT / 2 - GAME_CONSTANTS.PADDLE_HEIGHT / 2,
    score: 0,
  };
  return {
    ball: initialBall,
    p1: initialPaddle,
    p2: initialPaddle,
    status: 'waiting',
    servingPlayer: null,
  };
};

// Pure function: Update paddle (returns new state)
const updatePaddleState = (
  state: GameDisplayState,
  player: 1 | 2,
  y: number
): GameDisplayState => {
  if (player === 1) {
    return { ...state, p1: { ...state.p1, y } };
  }
  return { ...state, p2: { ...state.p2, y } };
};

// Pure function: Update ball from server
const updateBallState = (state: GameDisplayState, ball: Ball): GameDisplayState => ({
  ...state,
  ball: { ...ball },
});

// Pure function: Update scores (returns new state)
const updateScoresState = (
  state: GameDisplayState,
  p1Score: number,
  p2Score: number
): GameDisplayState => ({
  ...state,
  p1: { ...state.p1, score: p1Score },
  p2: { ...state.p2, score: p2Score },
});

// Pure function: Update status (returns new state)
const updateStatusState = (
  state: GameDisplayState,
  status: GameDisplayState['status']
): GameDisplayState => ({
  ...state,
  status,
});

// Pure function: Render single court line
const renderCourtLine = (
  row: number,
  ballX: number,
  ballY: number,
  p1Y: number,
  p2Y: number
): string => {
  const ballRoundX = Math.round(ballX);
  const ballRoundY = Math.round(ballY);
  const p1RoundY = Math.round(p1Y);
  const p2RoundY = Math.round(p2Y);

  let line = '';
  for (let col = 0; col < GAME_CONSTANTS.COURT_WIDTH; col++) {
    if (col === ballRoundX && row === ballRoundY) {
      line += '●';
    } else if (col === 1 && row >= p1RoundY && row < p1RoundY + GAME_CONSTANTS.PADDLE_HEIGHT) {
      line += '█';
    } else if (col === GAME_CONSTANTS.COURT_WIDTH - 2 && row >= p2RoundY && row < p2RoundY + GAME_CONSTANTS.PADDLE_HEIGHT) {
      line += '█';
    } else {
      line += ' ';
    }
  }
  return line;
};

// Pure component: Static header
const Header = React.memo(() => (
  <Box justifyContent="center" paddingY={1}>
    <Text bold color="cyan">
      PONG - MQTT EDITION (Functional)
    </Text>
  </Box>
));

// Pure component: Score board
const ScoreBoard = React.memo(({
  p1Score,
  p2Score,
  gameId,
  playerNum,
  servingPlayer
}: {
  p1Score: number;
  p2Score: number;
  gameId: string;
  playerNum: number;
  servingPlayer: number | null;
}) => (
  <Box flexDirection="column">
    <Box justifyContent="space-between" paddingX={2} paddingBottom={1}>
      <Text>
        P1: <Text color="green">{p1Score}</Text>
        {servingPlayer === 1 && <Text color="cyan"> [SERVING]</Text>}
      </Text>
      <Text color="gray">
        Game: {gameId} | You: P{playerNum}
      </Text>
      <Text>
        P2: <Text color="green">{p2Score}</Text>
        {servingPlayer === 2 && <Text color="cyan"> [SERVING]</Text>}
      </Text>
    </Box>
  </Box>
));

// Pure component: Footer
const Footer = React.memo(({
  status,
  servingPlayer,
  playerNumber
}: {
  status: string;
  servingPlayer: number | null;
  playerNumber: number;
}) => (
  <Box paddingTop={1} paddingX={2} justifyContent="space-between">
    <Text color="gray">Arrow keys: UP/DOWN | Q to quit{status === 'ended' ? ' | R to restart' : ''}</Text>
    {status === 'waiting' && <Text color="yellow">Waiting for both players...</Text>}
    {status === 'playing' && servingPlayer === playerNumber && (
      <Text bold color="cyan">Press SPACEBAR to serve!</Text>
    )}
    {status === 'playing' && servingPlayer !== null && servingPlayer !== playerNumber && (
      <Text color="yellow">Opponent serving...</Text>
    )}
    {status === 'ended' && <Text bold color="green">Game Over! Press R to restart</Text>}
  </Box>
));

// Pure component: Game court (memoized with pure rendering logic)
const GameCourt = React.memo(({
  ballX,
  ballY,
  p1Y,
  p2Y
}: {
  ballX: number;
  ballY: number;
  p1Y: number;
  p2Y: number;
}) => {
  // Pure function: Render all court lines
  const courtContent = useMemo(() => {
    const lines: string[] = [];
    for (let row = 0; row < GAME_CONSTANTS.COURT_HEIGHT; row++) {
      lines.push(renderCourtLine(row, ballX, ballY, p1Y, p2Y));
    }
    return lines;
  }, [ballX, ballY, p1Y, p2Y]);

  return (
    <Box
      borderStyle="single"
      borderColor="white"
      width={GAME_CONSTANTS.COURT_WIDTH + 2}
      height={GAME_CONSTANTS.COURT_HEIGHT + 2}
      flexDirection="column"
    >
      {courtContent.map((line, idx) => (
        <Text key={idx}>{line}</Text>
      ))}
    </Box>
  );
}, (prev, next) => {
  // Only re-render if rounded positions changed
  const ballMoved = Math.round(prev.ballX) !== Math.round(next.ballX) ||
                    Math.round(prev.ballY) !== Math.round(next.ballY);
  const p1Moved = Math.round(prev.p1Y) !== Math.round(next.p1Y);
  const p2Moved = Math.round(prev.p2Y) !== Math.round(next.p2Y);
  return !ballMoved && !p1Moved && !p2Moved;
});

export function Game({ config, mqttClient }: GameProps) {
  const { exit } = useApp();

  // Immutable state (functional approach)
  const [gameState, setGameState] = useState<GameDisplayState>(createInitialState);

  // Handle keyboard input - pure function for new position
  useInput((input, key) => {
    if (input === 'q' || input === 'Q') {
      exit();
      return;
    }

    // Handle restart
    if ((input === 'r' || input === 'R') && gameState.status === 'ended') {
      mqttClient.publish(
        topics.restart(config.gameId),
        JSON.stringify({ timestamp: Date.now() })
      );
      return;
    }

    // Handle serve (spacebar)
    if (input === ' ' && gameState.status === 'playing' && gameState.servingPlayer === config.playerNumber) {
      mqttClient.publish(
        topics.serve(config.gameId),
        JSON.stringify({ player: config.playerNumber, timestamp: Date.now() })
      );
      return;
    }

    // Only allow paddle movement during play
    if (gameState.status !== 'playing') {
      return;
    }

    const currentY = config.playerNumber === 1 ? gameState.p1.y : gameState.p2.y;

    if (key.upArrow) {
      const newY = Math.max(0, currentY - GAME_CONSTANTS.PADDLE_SPEED);

      // CLIENT-SIDE PREDICTION: Update local state immediately for responsiveness
      setGameState(state => updatePaddleState(state, config.playerNumber, newY));

      // Also send to server (server is authoritative, will correct if needed)
      publishPaddle(newY);
    } else if (key.downArrow) {
      const maxY = GAME_CONSTANTS.COURT_HEIGHT - GAME_CONSTANTS.PADDLE_HEIGHT;
      const newY = Math.min(maxY, currentY + GAME_CONSTANTS.PADDLE_SPEED);

      // CLIENT-SIDE PREDICTION: Update local state immediately for responsiveness
      setGameState(state => updatePaddleState(state, config.playerNumber, newY));

      // Also send to server (server is authoritative, will correct if needed)
      publishPaddle(newY);
    }
  });

  // Side effect: Publish paddle position
  const publishPaddle = (y: number): void => {
    const update: PaddleUpdate = { y, timestamp: Date.now() };
    mqttClient.publish(
      topics.paddle(config.gameId, config.playerNumber),
      JSON.stringify(update)
    );
  };

  // Side effect: Subscribe to MQTT topics
  useEffect(() => {
    const setupMqtt = async () => {
      try {
        await mqttClient.subscribe(topics.all(config.gameId));

        // Announce join (and resend periodically while waiting)
        const announceJoin = () => {
          mqttClient.publish(
            topics.join(config.gameId),
            JSON.stringify({ player: config.playerNumber, timestamp: Date.now() })
          );
        };

        // Initial join
        announceJoin();

        // Don't use automatic rejoin - it causes issues during gameplay
        // Just send join once on connect

        // Listen for server updates (all state updates are pure)
        const unsubscribe = mqttClient.onMessage((topic, message) => {
          const parsed = parseTopic(topic);
          if (!parsed || parsed.gameId !== config.gameId) return;

          try {
            const data = JSON.parse(message.toString());

            // Server is authoritative - accept ALL updates immediately
            if (parsed.messageType === 'paddle' && parsed.player) {
              const paddleUpdate = data as PaddleUpdate;
              if (parsed.player === config.playerNumber) {
                // Server echo of OUR paddle - only accept if significantly different (desync correction)
                setGameState(state => {
                  const currentY = parsed.player === 1 ? state.p1.y : state.p2.y;
                  const diff = Math.abs(currentY - paddleUpdate.y);
                  if (diff > 1) {
                    // Significant desync - correct to server position
                    return updatePaddleState(state, parsed.player!, paddleUpdate.y);
                  }
                  return state;
                });
              } else {
                // OTHER player's paddle - always use server position
                setGameState(state => updatePaddleState(state, parsed.player!, paddleUpdate.y));
              }
            }
            else if (parsed.messageType === 'ball') {
              const ballUpdate = data as BallUpdate;
              // Use server position directly - NO interpolation
              setGameState(state => updateBallState(state, ballUpdate));
            }
            else if (parsed.messageType === 'state') {
              const stateUpdate = data as StateUpdate;
              setGameState(state => {
                let newState = updateScoresState(state, stateUpdate.p1Score, stateUpdate.p2Score);
                newState = updateStatusState(newState, stateUpdate.status);
                newState = { ...newState, servingPlayer: stateUpdate.servingPlayer };
                return newState;
              });
            }
          } catch (err) {
            console.error('Error parsing message:', err);
          }
        });

        return () => {
          unsubscribe();
        };
      } catch (err) {
        console.error('MQTT setup error:', err);
      }
    };

    setupMqtt();
  }, [config.gameId, config.playerNumber, mqttClient]);

  // Pure rendering
  return (
    <Box flexDirection="column">
      <Header />
      <ScoreBoard
        p1Score={gameState.p1.score}
        p2Score={gameState.p2.score}
        gameId={config.gameId}
        playerNum={config.playerNumber}
        servingPlayer={gameState.servingPlayer}
      />
      <GameCourt
        ballX={gameState.ball.x}
        ballY={gameState.ball.y}
        p1Y={gameState.p1.y}
        p2Y={gameState.p2.y}
      />
      <Footer
        status={gameState.status}
        servingPlayer={gameState.servingPlayer}
        playerNumber={config.playerNumber}
      />
    </Box>
  );
}
