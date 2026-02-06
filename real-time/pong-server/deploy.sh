#!/bin/bash
set -e

SSH_KEY="$HOME/raptor-server/raptor.pem"
SERVER="ec2-user@3.141.116.27"
REMOTE_DIR="/home/ec2-user/pong-server"

echo "🚀 Deploying Pong Game Server..."

# Build locally
echo "📦 Building server..."
npm run build

# Create deployment package
echo "📦 Creating deployment package..."
tar -czf pong-server.tar.gz dist package.json package-lock.json

# Upload to server
echo "📤 Uploading to server..."
scp -i "$SSH_KEY" pong-server.tar.gz "$SERVER:/tmp/"

# Install and setup on server
echo "🔧 Setting up on server..."
ssh -i "$SSH_KEY" "$SERVER" << 'ENDSSH'
set -e

# Install Node.js if not present
if ! command -v node &> /dev/null; then
    echo "📥 Installing Node.js..."
    curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo bash -
    sudo yum install -y nodejs
fi

# Create directory
sudo mkdir -p /home/ec2-user/pong-server
sudo chown ec2-user:ec2-user /home/ec2-user/pong-server

# Extract package
cd /home/ec2-user/pong-server
tar -xzf /tmp/pong-server.tar.gz
rm /tmp/pong-server.tar.gz

# Install dependencies
npm install --production

echo "✅ Server files deployed"
ENDSSH

# Install systemd service
echo "🔧 Installing systemd service..."
scp -i "$SSH_KEY" pong-server.service "$SERVER:/tmp/"
ssh -i "$SSH_KEY" "$SERVER" << 'ENDSSH'
set -e

# Install service
sudo mv /tmp/pong-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable pong-server
sudo systemctl restart pong-server

echo "✅ Service installed and started"
ENDSSH

# Cleanup
rm -f pong-server.tar.gz

echo ""
echo "✅ Deployment complete!"
echo "📊 Check status: ssh -i $SSH_KEY $SERVER 'sudo systemctl status pong-server'"
echo "📝 View logs:    ssh -i $SSH_KEY $SERVER 'sudo journalctl -u pong-server -f'"
