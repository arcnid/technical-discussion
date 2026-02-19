#!/usr/bin/env node
import mqtt from 'mqtt';
import { createGameState, updatePaddle, gameTick, startGame, addPlayer, bothPlayersJoined, resetGame, serveBall, } from './game-logic.js';
// Configuration
const BROKER_URL = process.env.MQTT_BROKER || 'mqtt://localhost:1883';
const MQTT_USER = process.env.MQTT_USER || 'raptor';
const MQTT_PASS = process.env.MQTT_PASS || 'raptorMQTT2025';
const FRAME_RATE = 60;
console.log(`🎮 Pong Game Server Starting (Functional)...`);
console.log(`📡 MQTT Broker: ${BROKER_URL}`);
// Connect to MQTT with more aggressive keepalive settings
const client = mqtt.connect(BROKER_URL, {
    clientId: `pong-server-${Date.now()}`,
    clean: true,
    keepalive: 30, // More frequent keepalive (30s instead of 60s)
    reconnectPeriod: 1000, // Faster reconnect (1s instead of 5s)
    connectTimeout: 30 * 1000,
    username: MQTT_USER,
    password: MQTT_PASS,
});
// Immutable game storage (functional approach)
let games = new Map();
// Pure function: Get or create game state
const getOrCreateGame = (gameId) => {
    const existing = games.get(gameId);
    if (existing) {
        return existing;
    }
    console.log(`🎮 New game started: ${gameId}`);
    const newGame = createGameState();
    games = new Map(games).set(gameId, newGame);
    startGameLoop(gameId);
    return newGame;
};
// Pure function: Update game in immutable map
const setGame = (gameId, state) => {
    games = new Map(games).set(gameId, state);
};
// Pure function: Delete game from immutable map
const deleteGame = (gameId) => {
    const newGames = new Map(games);
    newGames.delete(gameId);
    games = newGames;
    console.log(`🗑️  Game ${gameId} cleaned up`);
};
client.on('connect', () => {
    console.log('✅ Connected to MQTT broker');
    client.subscribe('pong/game/+/p1/paddle', { qos: 0 });
    client.subscribe('pong/game/+/p2/paddle', { qos: 0 });
    client.subscribe('pong/game/+/join', { qos: 0 });
    client.subscribe('pong/game/+/restart', { qos: 0 });
    client.subscribe('pong/game/+/serve', { qos: 0 });
    console.log('🎯 Subscribed to game topics');
    console.log('🚀 Server ready - waiting for players...\n');
});
client.on('error', (err) => {
    console.error('❌ MQTT Error:', err.message);
});
client.on('message', (topic, message) => {
    try {
        const parts = topic.split('/');
        if (parts[0] !== 'pong' || parts[1] !== 'game')
            return;
        const gameId = parts[2];
        const messageType = parts[3];
        // Get or create game (pure function)
        let state = getOrCreateGame(gameId);
        const data = JSON.parse(message.toString());
        // Handle paddle updates (pure function)
        if (messageType === 'p1' || messageType === 'p2') {
            const paddleData = data;
            const player = messageType === 'p1' ? 1 : 2;
            state = updatePaddle(state, player, paddleData.y);
            setGame(gameId, state);
        }
        // Handle join messages
        if (messageType === 'join') {
            const joinData = data;
            state = addPlayer(state, joinData.player);
            setGame(gameId, state);
            console.log(`👋 Player ${joinData.player} joined game ${gameId}`);
            // Always publish current state so the joining client knows the game status
            // (including 'ended', so the client can immediately send a restart)
            if (state.status === 'waiting' && bothPlayersJoined(state)) {
                state = startGame(state);
                setGame(gameId, state);
                console.log(`▶️  Game ${gameId} started! Both players ready.`);
            }
            else if (state.status === 'waiting') {
                console.log(`⏳ Game ${gameId} waiting for ${state.playersJoined.has(1) ? 'player 2' : 'player 1'}...`);
            }
            publishGameState(gameId, state);
        }
        // Handle restart messages
        if (messageType === 'restart' && state.status === 'ended') {
            console.log(`🔄 Restarting game ${gameId}...`);
            state = resetGame(state);
            setGame(gameId, state);
            publishGameState(gameId, state);
            console.log(`▶️  Game ${gameId} restarted!`);
        }
        // Handle serve messages
        if (messageType === 'serve' && state.status === 'playing' && state.servingPlayer !== null) {
            const serveData = data;
            // Only allow the serving player to serve
            if (serveData.player === state.servingPlayer) {
                console.log(`🎾 Player ${serveData.player} serves!`);
                state = serveBall(state);
                setGame(gameId, state);
                publishGameState(gameId, state);
            }
        }
    }
    catch (err) {
        console.error('Error processing message:', err);
    }
});
// Game loop using pure functions
function startGameLoop(gameId) {
    const interval = setInterval(() => {
        const state = games.get(gameId);
        if (!state) {
            clearInterval(interval);
            return;
        }
        if (state.status !== 'playing') {
            return;
        }
        // Pure function: Update game state
        const newState = gameTick(state);
        // Check if score changed (for logging)
        if (newState.p1Score !== state.p1Score) {
            console.log(`🎯 Player 1 scores! (${newState.p1Score} - ${newState.p2Score})`);
        }
        if (newState.p2Score !== state.p2Score) {
            console.log(`🎯 Player 2 scores! (${newState.p1Score} - ${newState.p2Score})`);
        }
        // Update immutable state
        setGame(gameId, newState);
        // Publish ball updates every frame (60/sec) for smooth gameplay
        publishBall(gameId, newState.ball);
        // Publish state on score change
        if (newState.p1Score !== state.p1Score || newState.p2Score !== state.p2Score) {
            publishGameState(gameId, newState);
        }
        // Handle game end
        if (newState.status === 'ended' && state.status === 'playing') {
            const winner = newState.p1Score > newState.p2Score ? 'Player 1' : 'Player 2';
            console.log(`🏆 Game ${gameId} ended! ${winner} wins! (${newState.p1Score} - ${newState.p2Score})`);
            publishGameState(gameId, newState);
            clearInterval(interval);
            // Clean up after 30 seconds
            setTimeout(() => deleteGame(gameId), 30000);
        }
    }, 1000 / FRAME_RATE);
}
// Side effect: Publish paddle to MQTT
function publishPaddle(gameId, player, y) {
    const update = {
        y,
        timestamp: Date.now(),
    };
    client.publish(`pong/game/${gameId}/p${player}/paddle`, JSON.stringify(update), { qos: 0 });
}
// Side effect: Publish ball to MQTT
function publishBall(gameId, ball) {
    const update = {
        x: ball.x,
        y: ball.y,
        dx: ball.dx,
        dy: ball.dy,
        timestamp: Date.now(),
    };
    client.publish(`pong/game/${gameId}/ball`, JSON.stringify(update), { qos: 0 });
}
// Side effect: Publish game state to MQTT
function publishGameState(gameId, state) {
    const update = {
        status: state.status,
        p1Score: state.p1Score,
        p2Score: state.p2Score,
        servingPlayer: state.servingPlayer,
        timestamp: Date.now(),
    };
    client.publish(`pong/game/${gameId}/state`, JSON.stringify(update), { qos: 0 });
}
// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n👋 Server shutting down...');
    client.end();
    process.exit(0);
});
process.on('SIGTERM', () => {
    console.log('\n👋 Server shutting down...');
    client.end();
    process.exit(0);
});
//# sourceMappingURL=index.js.map