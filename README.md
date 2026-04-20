# SmartSelf AI

**Intelligent Self-Learning Chatbot with Continuous Knowledge Acquisition**

## Overview

SmartSelf AI is a professional-grade chatbot that continuously learns from the internet, expanding its knowledge base through automated web crawling, content processing, and semantic search. Built following senior software engineering best practices with a clean, modular architecture.

## Features

- **Continuous Learning**: Automatically crawls and learns from websites, news feeds, and public APIs
- **Knowledge Integration**: Smart processing and storage of learned information in vector databases
- **Semantic Search**: Advanced semantic search using embeddings for intelligent information retrieval
- **Free API Integration**: 20+ free public APIs with no authentication required
- **Scalable Architecture**: Professional structure designed for growth and maintainability
- **Type Safety**: Pydantic models for configuration validation
- **Async Performance**: Built on async/await for high-performance operations

## Project Structure

```
smartself/
├── src/                      # Source code
│   ├── config/              # Configuration management
│   ├── chatbot/             # Core chatbot logic
│   ├── learning/            # Learning systems
│   ├── knowledge/           # Knowledge management
│   ├── api/                 # API integrations
│   ├── crawler/             # Web crawling
│   ├── processor/           # Content processing
│   └── utils/               # Utilities
├── tests/                   # Test suite
├── docs/                    # Documentation
├── config/                  # Configuration files
├── data/                    # Data storage
├── frontend/                # Frontend application
└── scripts/                 # Utility scripts
```

## Quick Start

### Prerequisites

- Python 3.11+
- pip or poetry

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/smartself-ai.git
cd smartself-ai
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Running the Application

```bash
python -m src.main
```

## Configuration

Edit `.env` file to configure:

- **OPENAI_API_KEY**: OpenAI API key for LLM functionality
- **DEEPSEEK_API_KEY**: DeepSeek API key (alternative to OpenAI)
- **DEBUG**: Enable debug mode (default: false)
- **HOST**: Server host (default: 0.0.0.0)
- **PORT**: Server port (default: 8000)

## Architecture

### Design Principles

- **Separation of Concerns**: Each module has a single, well-defined responsibility
- **Dependency Injection**: Components receive dependencies rather than creating them
- **Interface-Based Design**: Clear interfaces between components
- **Testability**: All components are easily testable in isolation
- **Scalability**: Structure supports horizontal and vertical scaling
- **Maintainability**: Code is organized for easy maintenance and updates

### Core Components

#### src/config/settings.py
Configuration management with Pydantic validation

#### src/chatbot/
Core chatbot logic and conversation management

#### src/learning/
Continuous learning pipeline and knowledge acquisition

#### src/knowledge/
Knowledge base management and vector store operations

#### src/api/
External API integrations including 20+ free public APIs

#### src/crawler/
Web crawling and RSS feed processing

#### src/processor/
Content processing and text normalization

#### src/utils/
Logging utilities and helper functions

## Free APIs Included

The system includes 20+ free public APIs requiring no authentication:

1. Wikipedia API - Knowledge articles
2. Hacker News API - Tech news
3. GitHub API - Public events
4. Chuck Norris API - Jokes
5. Official Jokes API - Programming jokes
6. Quotable API - Inspirational quotes
7. Advice Slip API - Life advice
8. Bored API - Activity suggestions
9. Open Trivia DB - Quiz questions
10. Rest Countries API - Country data
11. Dog CEO API - Dog images
12. Cat Facts API - Cat facts
13. Agify API - Age prediction
14. Genderize API - Gender prediction
15. Nationalize API - Nationality prediction
16. CoinGecko API - Crypto prices
17. Open Library API - Book data
18. IP API - Geolocation
19. Fake Store API - E-commerce data
20. JSONPlaceholder - Fake test data

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black src/
flake8 src/
mypy src/
```

### Adding New Features

1. Create feature branch
2. Implement changes in appropriate module
3. Add tests
4. Update documentation
5. Submit pull request

## Deployment

### Docker

```bash
docker build -t smartself-ai .
docker run -p 8000:8000 smartself-ai
```

### Docker Compose

```bash
docker-compose up -d
```

See `DEPLOYMENT.md` for detailed deployment instructions.

## Documentation

- `PROJECT_STRUCTURE.md` - Detailed project structure and design
- `ARCHITECTURE.md` - System architecture and component interaction
- `DEPLOYMENT.md` - Deployment guide
- `docs/` - Additional documentation

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details

## Support

- Documentation: [docs.smartself.ai](https://docs.smartself.ai)
- Issues: [GitHub Issues](https://github.com/your-org/smartself-ai/issues)
- Email: support@smartself.ai

---

Built with ❤️ by SmartSelf AI Team
