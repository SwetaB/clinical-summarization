#!/bin/bash

set -e  # exit immediately if a command fails

echo "Installing Docker..."
sudo apt update
sudo apt install -y docker.io unzip curl

echo "Fixing Docker permissions..."
sudo usermod -aG docker $USER

echo "Installing AWS CLI..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
sudo ./aws/install
aws --version

echo "Authenticating Docker to AWS ECR..."
aws configure
# Replace REGION and ACCOUNT_ID below
REGION="us-east-1"
ACCOUNT_ID="your-aws-account-id"

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

echo "Setup complete! Please run 'exec \$SHELL' or re-login to apply Docker permissions."
