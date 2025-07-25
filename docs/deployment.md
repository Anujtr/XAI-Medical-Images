# Deployment Guide

## Overview

This guide covers various deployment options for the XAI Medical Images application, from local development to production environments.

## Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- Git
- Sufficient storage for model checkpoints and data

## Local Development Deployment

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd XAI-Medical-Images
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   make install-dev
   # or
   pip install -r requirements-dev.txt
   ```

4. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env file with your settings
   ```

5. **Run the application:**
   ```bash
   make run-dev
   # or
   ENVIRONMENT=development python run.py
   ```

### Development Server Configuration

Create a development configuration file at `config/development.yaml`:

```yaml
flask:
  host: "127.0.0.1"
  port: 5000
  debug: true

logging:
  level: "DEBUG"

security:
  enable_csrf: false
```

## Docker Deployment

### Basic Docker Deployment

1. **Build the Docker image:**
   ```bash
   make docker-build
   # or
   docker build -t xai-medical-images .
   ```

2. **Run the container:**
   ```bash
   make docker-run
   # or
   docker run -p 5000:5000 xai-medical-images
   ```

### Docker Compose (Recommended)

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - ENVIRONMENT=production
      - FLASK_HOST=0.0.0.0
      - FLASK_PORT=5000
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - web
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

## Production Deployment

### Server Requirements

**Minimum Requirements:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 10GB free space
- Python 3.10+
- Docker (recommended)

**Recommended Requirements:**
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 50GB+ free space
- GPU: NVIDIA GPU with CUDA support (optional, for faster inference)

### Environment Setup

1. **Create production environment file:**
   ```bash
   cp .env.example .env.production
   ```

   Edit `.env.production`:
   ```bash
   ENVIRONMENT=production
   FLASK_HOST=0.0.0.0
   FLASK_PORT=5000
   SECRET_KEY=your-very-secure-secret-key-here
   MODEL_CHECKPOINT_PATH=/app/models/checkpoints/best_model.pth
   LOG_LEVEL=INFO
   ```

2. **Create production configuration:**
   ```yaml
   # config/production.yaml
   flask:
     host: "0.0.0.0"
     port: 5000
     debug: false
   
   security:
     enable_csrf: true
     max_requests_per_minute: 30
   
   logging:
     level: "INFO"
     log_dir: "/var/log/xai-medical"
   ```

### Using Gunicorn (Production WSGI Server)

1. **Install Gunicorn:**
   ```bash
   pip install gunicorn
   ```

2. **Create Gunicorn configuration file (`gunicorn.conf.py`):**
   ```python
   bind = "0.0.0.0:5000"
   workers = 4
   worker_class = "sync"
   worker_connections = 1000
   max_requests = 1000
   max_requests_jitter = 100
   timeout = 30
   keepalive = 2
   preload_app = True
   ```

3. **Run with Gunicorn:**
   ```bash
   gunicorn --config gunicorn.conf.py run:app
   ```

### Nginx Reverse Proxy

Create `/etc/nginx/sites-available/xai-medical`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration
    ssl_certificate /path/to/your/certificate.crt;
    ssl_certificate_key /path/to/your/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # Upload size limit
    client_max_body_size 16M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Static files (if serving separately)
    location /static {
        alias /path/to/app/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/xai-medical /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Systemd Service

Create `/etc/systemd/system/xai-medical.service`:

```ini
[Unit]
Description=XAI Medical Images Web Application
After=network.target

[Service]
Type=exec
User=www-data
Group=www-data
WorkingDirectory=/opt/xai-medical-images
Environment=PATH=/opt/xai-medical-images/venv/bin
Environment=ENVIRONMENT=production
ExecStart=/opt/xai-medical-images/venv/bin/gunicorn --config gunicorn.conf.py run:app
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable xai-medical.service
sudo systemctl start xai-medical.service
```

## Cloud Deployment

### AWS Deployment

#### Using EC2

1. **Launch EC2 instance:**
   - Ubuntu 20.04 LTS
   - t3.medium or larger
   - Security group: Allow HTTP (80), HTTPS (443), SSH (22)

2. **Install dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3.10 python3.10-venv python3-pip nginx
   ```

3. **Deploy application:**
   ```bash
   git clone <repository-url>
   cd XAI-Medical-Images
   python3.10 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Configure and start services as described above**

#### Using Docker on EC2

```bash
# Install Docker
sudo apt update
sudo apt install docker.io docker-compose
sudo usermod -aG docker $USER

# Deploy application
git clone <repository-url>
cd XAI-Medical-Images
docker-compose -f docker-compose.prod.yml up -d
```

### Google Cloud Platform

#### Using Cloud Run

1. **Build and push image:**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/xai-medical-images
   ```

2. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy xai-medical-images \
     --image gcr.io/PROJECT-ID/xai-medical-images \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 2Gi \
     --cpu 2
   ```

### Azure Deployment

#### Using Container Instances

```bash
az container create \
  --resource-group myResourceGroup \
  --name xai-medical-images \
  --image your-registry/xai-medical-images:latest \
  --dns-name-label xai-medical \
  --ports 5000 \
  --memory 2 \
  --cpu 2
```

## Monitoring and Logging

### Application Logs

Configure structured logging in production:

```python
# In your configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  log_dir: "/var/log/xai-medical"
```

### Health Monitoring

Set up monitoring for the `/health` endpoint:

```bash
# Simple health check script
#!/bin/bash
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/health)
if [ $response != "200" ]; then
    echo "Health check failed with status: $response"
    # Add alerting logic here
fi
```

### Log Rotation

Configure log rotation with logrotate:

```
# /etc/logrotate.d/xai-medical
/var/log/xai-medical/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 www-data www-data
    postrotate
        systemctl reload xai-medical
    endscript
}
```

## Security Considerations

### HTTPS/TLS

- Always use HTTPS in production
- Use valid SSL certificates (Let's Encrypt is free)
- Configure strong cipher suites

### Firewall Configuration

```bash
# UFW firewall rules
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

### Regular Updates

- Keep system packages updated
- Update Python dependencies regularly
- Monitor security advisories

### Environment Variables

- Never commit secrets to version control
- Use environment variables or secret management systems
- Rotate secrets regularly

## Backup and Recovery

### Database Backups (if applicable)

```bash
# Backup script
#!/bin/bash
backup_dir="/backups/xai-medical/$(date +%Y%m%d)"
mkdir -p $backup_dir

# Backup models
cp -r /opt/xai-medical-images/models $backup_dir/

# Backup configuration
cp -r /opt/xai-medical-images/config $backup_dir/
```

### Model Checkpoints

- Store model checkpoints in persistent storage
- Consider using cloud storage for redundancy
- Version your models appropriately

## Troubleshooting

### Common Issues

1. **Out of Memory:**
   - Reduce batch size in configuration
   - Increase server memory
   - Use CPU instead of GPU if memory-constrained

2. **Slow Response Times:**
   - Check model size and complexity
   - Optimize image preprocessing
   - Use GPU acceleration if available

3. **File Upload Issues:**
   - Check file size limits in nginx and Flask
   - Verify file permissions
   - Check disk space

### Log Analysis

```bash
# View application logs
sudo journalctl -u xai-medical.service -f

# Check nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Performance Monitoring

Use tools like:
- `htop` for system resources
- `nvidia-smi` for GPU usage
- Application performance monitoring (APM) tools

## Scaling

### Horizontal Scaling

- Use load balancers (nginx, HAProxy)
- Deploy multiple application instances
- Consider container orchestration (Kubernetes)

### Vertical Scaling

- Increase server resources (CPU, memory)
- Use GPU acceleration for inference
- Optimize model architecture

### Caching

- Implement Redis for session storage
- Cache frequently accessed models
- Use CDN for static assets