"""
Cloudflare Python Workers entrypoint.

Deployment (Cloudflare):
1. pip install -r requirements.txt
2. wrangler login
3. wrangler deploy
4. Optional local run: wrangler dev

This file is a compatibility layer only. It does not change Flask routes,
SSE behavior, threading logic, JSON storage, or Gunicorn deployment.
"""

from asgiref.wsgi import WsgiToAsgi
from app import app

asgi_app = WsgiToAsgi(app)
