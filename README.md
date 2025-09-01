# RAG Coding Assistant

A sophisticated LangGraph-based Retrieval Augmented Generation (RAG) system for intelligent code assistance, powered by Microsoft's Phi-3.5 model and ChromaDB vector storage.

## 🌟 Features

- **🧠 Intelligent Code Generation**: Generate Python code based on natural language descriptions
- **📖 Code Explanation**: Get detailed explanations of existing code snippets
- **💬 Smart Conversational Interface**: Context-aware chat that understands when you need coding help
- **🔍 RAG-Enhanced Responses**: Leverages OpenAI HumanEval dataset for context-aware code generation
- **🌐 Web Interface**: Gradio-based web UI for easy interaction
- **📊 Evaluation System**: Built-in MBPP dataset evaluation for performance testing (*work in progress*)
- **🚀 Multiple Deployment Options**: CLI, web interface

## 🏗️ Architecture

```
📁 RAG Coding Assistant/
├── 📁 src/
│   ├── 📁 rag/              # Vector storage & retrieval
│   ├── 📁 agents/           # LangGraph agent implementation
│   ├── 📁 models/           # Phi-3.5 model handling
│   ├── 📁 ui/               # Gradio web interface
│   └── 📁 evaluation/       # MBPP evaluation system
│   └── main.py
├── 📁 data/                 # Vector databases & datasets
├── 📁 docs/                 # Documentation
├── 📁 configs/              # Configuration files
└── 📁 scripts/              # Setup & utility scripts
└── pyproject.toml
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone <https://github.com/KarimK-x/rag-coding-assistant>
   cd rag-coding-assistant
   ```

2. **Run the setup script**
   ```bash
   python scripts/setup.py
   ```

   Or install manually:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

### Usage

#### � Tutorial Notebook (Recommended Start)
```bash
# Open the comprehensive tutorial notebook
jupyter notebook GradioDeployedCodingAssistant.ipynb
```
> **🎓 New to the project?** Start with the Jupyter notebook! It provides a step-by-step walkthrough of the entire system, from setup to deployment, with detailed explanations and live demonstrations of all features.

#### �🖥️ Command Line Interface
```bash
python main.py --mode cli
```

#### 🌐 Web Interface
```bash
python main.py --mode gradio
```

#### 📊 Run Evaluation
```bash
python main.py --mode evaluation
```

## 💡 How It Works

### 1. **RAG System**
- Uses ChromaDB for efficient vector storage
- HuggingFace embeddings with `all-mpnet-base-v2` model
- Retrieves relevant code examples from OpenAI HumanEval dataset

### 2. **LangGraph Agent**
- **Smart Routing**: Automatically detects intent (chat, explain, generate)
- **Multi-Node Architecture**: Separate nodes for different types of interactions
- **Context Management**: Maintains conversation history and context

### 3. **Phi-3.5 Integration**
- Microsoft's Phi-3.5-mini-instruct model for code generation
- Optimized for code understanding and generation tasks
- Supports both CPU and GPU inference

## 🎯 Use Cases

### Code Generation
```
User: "Write a function that finds the longest common subsequence between two strings"
Assistant: [Generates complete Python function with explanations]
```

### Code Explanation
```
User: "Can you explain what this bubble sort implementation does?"
Assistant: [Provides line-by-line explanation of the algorithm]
```

### General Programming Help
```
User: "What's the difference between list and tuple in Python?"
Assistant: [Provides comprehensive explanation with examples]
```

## 📁 Project Structure

### Core Components

- **`src/rag/vectorstore.py`**: Vector database setup and management
- **`src/rag/retrieval.py`**: Document retrieval functions
- **`src/agents/langgraph_agent.py`**: Main LangGraph agent implementation
- **`src/agents/intent_classifier.py`**: Smart intent classification system
- **`src/models/phi_model.py`**: Phi-3.5 model initialization and inference
- **`src/ui/gradio_interface.py`**: Web interface implementation
- **`src/evaluation/mbpp_evaluator.py`**: MBPP evaluation system (*work in progress*)

### Tutorial & Documentation

- **`GradioDeployedCodingAssistant.ipynb`**: **📓 Complete tutorial notebook** - Interactive walkthrough covering the entire system from setup to deployment. Perfect for understanding the architecture and seeing live demonstrations.

### Configuration Files

- **`requirements.txt`**: Python dependencies
- **`setup_env.py`**: Environment setup for devs
- **`pyproject.toml`**: Modern Python packaging

## 🔧 Configuration

### Model Configuration
The system uses Microsoft's Phi-3.5-mini-instruct model by default. To use a different model:

```python
# In src/models/phi_model.py
model_name = "your-preferred-model"
```
## 🚀 Deployment

### Local Development
```bash
python main.py --mode gradio
```

### Production Deployment
The Gradio interface can be deployed to various platforms:

- **Hugging Face Spaces**: Upload the repository directly
- **Local Server**: Run with public sharing enabled


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📧 Contact

**Karim Khaled** - [karimkhaled2k4@gmail.com]

---