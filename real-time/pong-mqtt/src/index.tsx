#!/usr/bin/env node
import React, { useState, useEffect } from 'react';
import { render, Box, Text } from 'ink';
import { Game } from './game/Game';
import { getMqttClient } from './mqtt/client';
import type { GameConfig, PlayerNumber } from './game/types';

// Parse command-line arguments
function parseArgs(): {
  gameId: string;
  playerNumber: PlayerNumber;
  brokerUrl: string;
  username?: string;
  password?: string;
} {
  const args = process.argv.slice(2);

  let gameId = 'demo';
  let playerNumber: PlayerNumber = 1;
  // Default to Raptor cloud broker
  let brokerUrl = 'mqtt://3.141.116.27:1883';
  let username = 'raptor';
  let password = 'raptorMQTT2025';

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    if (arg === '--game' && args[i + 1]) {
      gameId = args[i + 1];
      i++;
    } else if (arg === '--player' && args[i + 1]) {
      const num = parseInt(args[i + 1], 10);
      if (num === 1 || num === 2) {
        playerNumber = num;
      }
      i++;
    } else if (arg === '--broker' && args[i + 1]) {
      brokerUrl = args[i + 1];
      i++;
    } else if (arg === '--username' && args[i + 1]) {
      username = args[i + 1];
      i++;
    } else if (arg === '--password' && args[i + 1]) {
      password = args[i + 1];
      i++;
    } else if (arg === '--local') {
      // Quick flag to use local broker
      brokerUrl = 'mqtt://localhost:1883';
      username = undefined;
      password = undefined;
    }
  }

  return { gameId, playerNumber, brokerUrl, username, password };
}

function App() {
  const [mqttClient] = useState(() => {
    const { brokerUrl, username, password } = parseArgs();
    return getMqttClient({ brokerUrl, username, password });
  });

  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const connect = async () => {
      try {
        await mqttClient.connect();
        setConnected(true);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    };

    connect();

    return () => {
      mqttClient.disconnect();
    };
  }, [mqttClient]);

  if (error) {
    return (
      <Box flexDirection="column" padding={1}>
        <Text color="red" bold>
          Error connecting to MQTT broker:
        </Text>
        <Text color="red">{error}</Text>
        <Text color="gray" marginTop={1}>
          Make sure the MQTT broker is running at the specified URL.
        </Text>
      </Box>
    );
  }

  if (!connected) {
    return (
      <Box padding={1}>
        <Text color="yellow">Connecting to MQTT broker...</Text>
      </Box>
    );
  }

  const { gameId, playerNumber, brokerUrl } = parseArgs();

  const config: GameConfig = {
    gameId,
    playerNumber,
    isHost: false,  // No longer used - server is authoritative
    brokerUrl,
  };

  return <Game config={config} mqttClient={mqttClient} />;
}

// Render the app
render(<App />);

// Handle cleanup on exit
process.on('SIGINT', () => {
  process.exit(0);
});

process.on('SIGTERM', () => {
  process.exit(0);
});
