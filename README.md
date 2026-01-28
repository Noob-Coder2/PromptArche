# PromptArche

> **Analyze your AI conversation history** — Extract patterns, discover habits, and get brutally honest insights about how you prompt.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Supabase](https://img.shields.io/badge/Supabase-Backend-orange)

## Overview

PromptArche ingests your exported conversations from ChatGPT, Claude, and Grok, then uses semantic clustering and AI analysis to reveal patterns in how you interact with AI assistants.

### Key Features

- **Multi-Provider Import** — Upload exports from ChatGPT, Claude, or Grok
- **Semantic Clustering** — UMAP + HDBSCAN groups similar prompts automatically
- **Brutally Honest Insights** — AI-powered analysis of your prompting habits
- **Async Processing** — Background task queue with job tracking for large uploads
- **Progress Tracking** — Real-time job status with retry logic and error handling
- **Secure Auth** — HTTP-only cookies with Supabase JWT validation

---

## Architecture

```
app/
├── core/
│   ├── config.py          # Environment settings
│   ├── security.py        # OAuth2 + JWT authentication
│   ├── error_handler.py   # Global error handling
│   ├── exceptions.py      # Custom exception classes
│   ├── rate_limiter.py    # Rate limiting middleware
│   ├── repository.py      # Data access layer
│   └── validators.py      # Input validation
├── db/
│   ├── schema.sql         # Main database schema
│   ├── ingestion_jobs.sql # Job tracking schema
│   └── supabase.py        # Database client
├── routers/
│   ├── auth.py            # Authentication endpoints
│   ├── ingestion.py       # File upload & job management
│   ├── dashboard.py       # Dashboard statistics & data
│   ├── pages.py           # HTML page rendering
│   └── health.py          # Health check endpoints
├── services/
│   ├── embeddings.py      # HuggingFace embedding generation
│   ├── ingestion.py       # File parsing & import
│   ├── clustering.py      # UMAP + HDBSCAN clustering
│   ├── adaptive_clustering.py  # Dynamic parameter tuning
│   ├── insights.py        # Groq LLM insight generation
│   ├── job_service.py     # Ingestion job management
│   ├── task_queue.py      # Async background task queue
│   ├── health_check.py    # System health monitoring
│   └── embedding_cache.py # Embedding cache management
├── templates/             # Jinja2 HTML templates
│   ├── base.html
│   ├── login.html
│   ├── dashboard.html
│   └── upload.html
├── schemas.py             # Pydantic models
└── main.py                # FastAPI application
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Supabase project (free tier works)
- HuggingFace API token
- Groq API key

### Installation

```bash
# Clone and setup
git clone https://github.com/your-repo/PromptArche.git
cd PromptArche

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_JWT_SECRET=your-jwt-secret
HF_TOKEN=hf_your_huggingface_token
GROQ_API_KEY=gsk_your_groq_key
```

### Database Setup

Run these SQL files in your Supabase SQL Editor:

1. `app/db/schema.sql` — Core tables (prompts, clusters, insights)
2. `app/db/ingestion_jobs.sql` — Job tracking table

### Run

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000

---

## API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/login` | Set HTTP-only auth cookie |
| POST | `/api/logout` | Clear auth cookie |

### Ingestion
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/ingest` | Upload file for processing |
| GET | `/api/jobs/{id}` | Get job status |
| GET | `/api/jobs` | List recent jobs |
| GET | `/api/jobs/active` | Get active job |

### Pages
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Login page |
| GET | `/dashboard` | Main dashboard |
| GET | `/upload` | File upload page |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Uvicorn |
| Database | Supabase (PostgreSQL + pgvector) |
| Embeddings | HuggingFace BGE-Large-EN-v1.5 |
| Clustering | UMAP + HDBSCAN |
| Insights | Groq (Qwen3-32B) |
| Auth | Supabase Auth + JWT |

---

## Architecture Highlights

### Recent Improvements
- ✅ **Async Task Queue** — Background processing for large file ingestion
- ✅ **Job Management** — Persistent job tracking with retry logic
- ✅ **Embedding Cache** — Reduces API calls with intelligent caching
- ✅ **Error Handling** — Comprehensive exception framework with user-friendly messages
- ✅ **Rate Limiting** — Per-endpoint request throttling
- ✅ **Repository Pattern** — Clean data access abstraction layer
- ✅ **Health Monitoring** — System health checks and diagnostics

### Performance Features
- Concurrent embedding generation with connection pooling
- Streaming JSON parsing for memory-efficient large file handling
- Adaptive clustering with automatic parameter tuning
- HTTP-only secure cookies for authentication

---

## Docker

```bash
docker-compose up --build
```

---

## License

MIT
