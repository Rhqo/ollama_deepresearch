# ollama deepresearch

### How to start
```bash
    cd ollama_deepresearch

    # Firecrawl api key (https://www.firecrawl.dev/app)
    touch .env
    echo "FIRECRAWL_API_KEY="fc-..."" > .env

    # Download Ollama (https://ollama.com/download)
    ollama pull gemma3
    ollama pull deepseek-r1:1.5b

    # install uv
    pip install uv

    # start survey agent    
    uv run main.py
```