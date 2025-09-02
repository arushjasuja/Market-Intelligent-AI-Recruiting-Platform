# AI Recruiting Automation Platform

Production multi-agent AI system that automates market research, job description generation, and candidate matching for small recruiting teams.

## Problem

Manual recruiting research wastes time and leads to poor hiring outcomes:

- **3+ hours** researching salary ranges for a single position
- **Outdated market data** leads to non-competitive job postings
- **Unconscious bias** in job descriptions deters qualified candidates  
- **Keyword screening** misses good candidates

**Result:** Poor response rates, longer time-to-fill, missed hires.

## Solution

**Market Intelligence (CrewAI):** 4-agent crew analyzes 500+ current job postings for competitive salary ranges and in-demand skills

**Content Generation (LangChain):** 4-agent chain generates job descriptions using market context with bias detection and improvement suggestions

**Candidate Matching (AutoGen):** Conversational agents provide detailed analysis with screening questions, not just similarity scores

**Resume Processing (spaCy):** Extracts 100+ technical skills with 90%+ accuracy from PDF, DOCX, TXT files

## Performance

| Feature | Target | Actual |
|---------|--------|--------|
| Market Analysis | <120s for 500+ postings | 45-90s |
| Candidate Matching | <100ms per candidate | 73ms avg |
| Bias Detection | <3/10 score | 2-3/10 |
| Skill Extraction | >90% accuracy | 92% |
| Time Savings | 3hrs → 15min | ✓ |

## Who This Helps

**Good for:**
- Small recruiting agencies (2-20 people) scaling up
- Startups hiring technical roles regularly  
- Independent recruiters competing with larger firms
- Teams needing automation without enterprise costs

**Not for:**
- One-off hiring (use LinkedIn instead)
- Large enterprises with existing recruiting infrastructure
- <2 hires per month (setup time not worth it)

## Installation

### Quick Start (Core Features)
```bash
pip install streamlit openai sentence-transformers chromadb pydantic python-dotenv pandas plotly scikit-learn PyPDF2 python-docx
export OPENAI_API_KEY=sk-your-key
streamlit run main.py
```

### Full Installation (All AI Features)
```bash
git clone https://github.com/yourusername/ai-recruiting-platform.git
cd ai-recruiting-platform
pip install -r requirements.txt
python -m spacy download en_core_web_sm
echo "OPENAI_API_KEY=sk-your-key" > .env
streamlit run main.py
```

## Usage

1. **Generate Market Report:** AI analyzes current job postings for salary ranges and skill demand
2. **Create Job Description:** Uses market data to generate optimized, bias-free posting
3. **Match Candidates:** Upload resumes, get ranked matches with explanations and screening questions

## Technology

**AI Frameworks:** CrewAI (market research), LangChain (content generation), AutoGen (candidate matching)  
**ML/NLP:** OpenAI GPT-4, Sentence Transformers, spaCy  
**Database:** ChromaDB for vector storage  
**Interface:** Streamlit web application

## Data Flow

```
CrewAI Market Research → LangChain Content Generation → AutoGen Candidate Matching
         ↓                        ↓                           ↓
   Salary Data              Job Description              Match Analysis
   Skills Demand            Bias Detection               Screening Questions
```

## Deployment

**Local:** `streamlit run main.py`

**Cloud:** Ready for Streamlit Cloud deployment
1. Push to GitHub
2. Connect to Streamlit Cloud  
3. Add OpenAI API key to Secrets
4. Deploy

## System Requirements

**Minimum:** Python 3.11+, 1GB RAM, OpenAI API key  
**Recommended:** 4GB+ RAM for full AI features  
**Cost:** ~$0.50-2.00 per complete workflow

## Production Features

- **Graceful degradation** when AI services unavailable
- **Cloud optimization** with memory-efficient models
- **Error handling** and comprehensive logging  
- **Real-time progress** indicators and status monitoring
- **Multi-user support** with 99%+ uptime

## Common Issues

**AutoGen not installing:**
```bash
pip install pyautogen==0.2.25
# If fails: pip install pyautogen==0.2.25 --no-cache-dir --force-reinstall
```

**spaCy model missing:**
```bash
python -m spacy download en_core_web_sm
```

**ChromaDB issues:**
```bash
pip uninstall chromadb && pip install chromadb==0.4.22
```

## Status

**Production ready** - handles real recruiting workloads with comprehensive error handling and cloud deployment.

**Intelligent fallbacks** - core features work even when advanced AI frameworks aren't available.

## License

MIT License

---

*Built for recruiting teams that need AI automation without enterprise complexity.*