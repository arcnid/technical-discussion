import type { PlayerNumber } from '../game/types';

/**
 * MQTT topic structure for the Pong game
 *
 * Topic hierarchy:
 *   pong/game/{gameId}/p1/paddle - Player 1 paddle position
 *   pong/game/{gameId}/p2/paddle - Player 2 paddle position
 *   pong/game/{gameId}/ball - Ball position (published by host)
 *   pong/game/{gameId}/state - Game state (scores, status)
 *   pong/game/{gameId}/join - Player join notifications
 */

export const topics = {
  /**
   * Get the paddle topic for a specific player
   */
  paddle: (gameId: string, player: PlayerNumber): string => {
    return `pong/game/${gameId}/p${player}/paddle`;
  },

  /**
   * Get the ball position topic
   */
  ball: (gameId: string): string => {
    return `pong/game/${gameId}/ball`;
  },

  /**
   * Get the game state topic (scores, status, etc)
   */
  state: (gameId: string): string => {
    return `pong/game/${gameId}/state`;
  },

  /**
   * Get the join topic for player notifications
   */
  join: (gameId: string): string => {
    return `pong/game/${gameId}/join`;
  },

  /**
   * Get the restart topic for restarting the game
   */
  restart: (gameId: string): string => {
    return `pong/game/${gameId}/restart`;
  },

  /**
   * Get the serve topic for serving the ball
   */
  serve: (gameId: string): string => {
    return `pong/game/${gameId}/serve`;
  },

  /**
   * Get wildcard topic to subscribe to all game messages
   */
  all: (gameId: string): string => {
    return `pong/game/${gameId}/#`;
  },
};

/**
 * Parse a topic string to extract game ID and message type
 */
export function parseTopic(topic: string): {
  gameId: string;
  messageType: 'paddle' | 'ball' | 'state' | 'join' | 'unknown';
  player?: PlayerNumber;
} | null {
  const parts = topic.split('/');

  // Expected format: pong/game/{gameId}/{type}
  if (parts.length < 4 || parts[0] !== 'pong' || parts[1] !== 'game') {
    return null;
  }

  const gameId = parts[2];
  const type = parts[3];

  if (type === 'p1' || type === 'p2') {
    // Paddle message: pong/game/{gameId}/p{1|2}/paddle
    const player = type === 'p1' ? 1 : 2;
    return { gameId, messageType: 'paddle', player: player as PlayerNumber };
  }

  if (type === 'ball') {
    return { gameId, messageType: 'ball' };
  }

  if (type === 'state') {
    return { gameId, messageType: 'state' };
  }

  if (type === 'join') {
    return { gameId, messageType: 'join' };
  }

  return { gameId, messageType: 'unknown' };
}
