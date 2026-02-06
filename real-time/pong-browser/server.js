#!/usr/bin/env node

const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

// Parse command line args
const args = process.argv.slice(2);
const openPlayer = args.find(arg => arg.startsWith('--open='))?.split('=')[1];

// Try ports starting from 3000
let port = 3000;
const maxPort = 3010;

function tryPort(portToTry) {
    const server = http.createServer((req, res) => {
        // Parse URL
        const url = new URL(req.url, `http://localhost:${portToTry}`);

        // Serve index.html
        if (url.pathname === '/' || url.pathname === '/index.html') {
            fs.readFile(path.join(__dirname, 'index.html'), (err, data) => {
                if (err) {
                    res.writeHead(500);
                    res.end('Error loading index.html');
                    return;
                }
                res.writeHead(200, { 'Content-Type': 'text/html' });
                res.end(data);
            });
        } else {
            res.writeHead(404);
            res.end('Not found');
        }
    });

    server.on('error', (err) => {
        if (err.code === 'EADDRINUSE') {
            console.log(`Port ${portToTry} is in use, trying ${portToTry + 1}...`);
            if (portToTry < maxPort) {
                tryPort(portToTry + 1);
            } else {
                console.error(`All ports from 3000-${maxPort} are in use!`);
                process.exit(1);
            }
        } else {
            console.error('Server error:', err);
            process.exit(1);
        }
    });

    server.on('listening', () => {
        const actualPort = server.address().port;
        console.log('\n🎮 MQTT Pong Browser Client\n');
        console.log(`📡 Server running at: http://localhost:${actualPort}\n`);
        console.log(`👥 Player 1: http://localhost:${actualPort}?player=1`);
        console.log(`👥 Player 2: http://localhost:${actualPort}?player=2\n`);
        console.log('🌐 Share with your friend:');
        console.log('   git clone https://github.com/arcnid/pong.git');
        console.log('   cd pong-browser');
        console.log('   npm run dev\n');
        console.log('Press Ctrl+C to stop\n');

        // Open browser if requested
        if (openPlayer) {
            const url = `http://localhost:${actualPort}?player=${openPlayer}`;
            console.log(`🚀 Opening Player ${openPlayer}: ${url}\n`);

            // Cross-platform open
            const cmd = process.platform === 'darwin' ? 'open' :
                       process.platform === 'win32' ? 'start' : 'xdg-open';
            spawn(cmd, [url], { detached: true, stdio: 'ignore' }).unref();
        }
    });

    server.listen(portToTry);
}

// Start trying ports
tryPort(port);
