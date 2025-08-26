# 🤖 Agentic BI Platform

An AI-powered Business Intelligence platform that automatically analyzes your data and provides actionable insights using OpenAI GPT models, LangGraph workflow orchestration, and Streamlit.

## 🚀 Features

- **🔍 Automatic Pattern Detection**: AI identifies trends, correlations, and anomalies
- **🧹 Intelligent Data Cleaning**: AI-powered data preprocessing and quality improvement
- **📊 Smart Visualizations**: Automatically generated charts based on data characteristics
- **💡 Business Insights**: AI-generated actionable recommendations and executive summaries
- **🔄 Workflow Orchestration**: LangGraph manages the entire analysis pipeline
- **📱 Interactive Dashboard**: Real-time progress tracking and results visualization

## 🏗️ Architecture

The platform uses an agentic architecture with specialized AI agents:

- **Data Collector Agent**: Validates and processes uploaded files
- **Data Cleaner Agent**: AI-powered data cleaning and preprocessing
- **Pattern Detector Agent**: Identifies statistical patterns and correlations
- **Visualizer Agent**: Generates intelligent chart recommendations
- **Insight Generator Agent**: Creates business insights and recommendations

All agents are orchestrated using LangGraph for seamless workflow management.

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- 4GB+ RAM recommended for larger datasets

## ⚡ Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd agentic_bi_streamlit
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## 📁 Project Structure

```
agentic_bi_streamlit/
├── app.py                          # Main Streamlit application
├── .env.example                    # Environment variables template
├── requirements.txt                # Python dependencies
├── config/
│   ├── settings.py                 # Application configuration
│   └── openai_config.py           # OpenAI client setup
├── agents/                         # AI agent implementations
│   ├── base_agent.py              # Base agent class
│   ├── data_collector.py          # File upload & validation
│   ├── data_cleaner.py            # AI-powered data cleaning
│   ├── pattern_detector.py        # Pattern analysis agent
│   ├── visualizer.py              # Chart generation agent
│   └── insight_generator.py       # Business insights agent
├── workflows/
│   ├── bi_workflow.py             # LangGraph workflow orchestration
│   └── workflow_state.py          # State management
├── components/                     # UI components
│   ├── file_uploader.py           # File upload interface
│   ├── progress_tracker.py        # Real-time progress tracking
│   ├── dashboard.py               # Interactive dashboard
│   └── insights_panel.py          # Insights display
├── utils/                          # Utility functions
│   ├── data_handler.py            # Data processing utilities
│   ├── chart_generator.py         # Chart generation helpers
│   ├── openai_utils.py            # OpenAI API utilities
│   └── session_manager.py         # Session state management
├── prompts/                        # AI prompts
│   ├── data_analysis.txt          # Data analysis prompts
│   ├── insight_generation.txt     # Insight generation prompts
│   └── chart_suggestions.txt      # Visualization prompts
└── .streamlit/
    └── config.toml                # Streamlit configuration
```

## 🎯 Usage

1. **Upload Data**: Upload CSV or Excel files (max 200MB)
2. **AI Analysis**: The system automatically processes your data through 5 stages:
   - Data Collection & Validation
   - AI-Powered Data Cleaning
   - Pattern Detection & Analysis
   - Smart Visualization Generation
   - Business Insights & Recommendations
3. **View Results**: Explore interactive dashboards, insights, and download reports
4. **Export**: Download insights reports and cleaned data

## 📊 Supported Data Formats

- **CSV files** (*.csv)
- **Excel files** (*.xlsx, *.xls)
- Maximum file size: 200MB
- Recommended: 100+ rows for meaningful analysis