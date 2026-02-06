import mqtt, { type MqttClient } from 'mqtt';

export interface MqttClientConfig {
  brokerUrl: string;
  clientId?: string;
  username?: string;
  password?: string;
}

export type MessageHandler = (topic: string, message: Buffer) => void;

// Immutable client interface (functional API)
export interface FunctionalMqttClient {
  readonly connect: () => Promise<void>;
  readonly subscribe: (topic: string) => Promise<void>;
  readonly publish: (topic: string, message: string, qos?: 0 | 1 | 2) => void;
  readonly onMessage: (handler: MessageHandler) => () => void;
  readonly disconnect: () => Promise<void>;
  readonly isConnected: () => boolean;
}

// Pure function: Generate client ID
const generateClientId = (): string =>
  `pong-${Math.random().toString(16).substr(2, 8)}`;

// Pure function: Create MQTT client options
const createMqttOptions = (config: MqttClientConfig) => ({
  clientId: config.clientId || generateClientId(),
  clean: true,
  keepalive: 60,
  reconnectPeriod: 5000,
  username: config.username,
  password: config.password,
});

// Factory function: Create functional MQTT client (closure-based state)
export const createMqttClient = (config: MqttClientConfig): FunctionalMqttClient => {
  // Encapsulated mutable state (hidden in closure)
  let client: MqttClient | null = null;
  let connected = false;
  const messageHandlers: Set<MessageHandler> = new Set();

  // Pure function: Connect to broker
  const connect = (): Promise<void> => {
    return new Promise((resolve, reject) => {
      const options = createMqttOptions(config);

      client = mqtt.connect(config.brokerUrl, options);

      client.on('connect', () => {
        connected = true;
        resolve();
      });

      client.on('error', (err) => {
        reject(err);
      });

      client.on('message', (topic, message) => {
        // Notify all handlers (functional iteration)
        messageHandlers.forEach((handler) => handler(topic, message));
      });

      client.on('offline', () => {
        connected = false;
      });
    });
  };

  // Pure function: Subscribe to topic
  const subscribe = (topic: string): Promise<void> => {
    return new Promise((resolve, reject) => {
      if (!client) {
        reject(new Error('Client not connected'));
        return;
      }

      client.subscribe(topic, { qos: 0 }, (err) => {
        if (err) {
          reject(err);
        } else {
          resolve();
        }
      });
    });
  };

  // Pure function: Publish message
  const publish = (topic: string, message: string, qos: 0 | 1 | 2 = 0): void => {
    if (!client || !connected) {
      console.warn('Cannot publish: client not connected');
      return;
    }

    client.publish(topic, message, { qos }, (err) => {
      if (err) {
        console.error('Publish error:', err);
      }
    });
  };

  // Pure function: Register message handler (returns unsubscribe function)
  const onMessage = (handler: MessageHandler): (() => void) => {
    messageHandlers.add(handler);

    // Return pure unsubscribe function (closure over handler)
    return () => {
      messageHandlers.delete(handler);
    };
  };

  // Pure function: Disconnect from broker
  const disconnect = (): Promise<void> => {
    return new Promise((resolve) => {
      if (!client) {
        resolve();
        return;
      }

      client.end(false, {}, () => {
        connected = false;
        resolve();
      });
    });
  };

  // Pure function: Check connection status
  const isConnected = (): boolean => connected;

  // Return immutable interface (all functions are closures)
  return Object.freeze({
    connect,
    subscribe,
    publish,
    onMessage,
    disconnect,
    isConnected,
  });
};

// Singleton pattern (functional approach)
let globalClient: FunctionalMqttClient | null = null;

// Pure function: Get or create singleton client
export const getMqttClient = (config: MqttClientConfig): FunctionalMqttClient => {
  if (!globalClient) {
    globalClient = createMqttClient(config);
  }
  return globalClient;
};
