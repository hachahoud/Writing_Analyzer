```markdown
# 📝 Student Writing Development Analyzer

An interactive web application that analyzes student writing samples to track linguistic development over time.
## Live Demo

🌐 [Visit the live app here](https://.streamlit.app)

## License

MIT License

## Author

Hamza Achahoud - Educational Technology Developer

## Features

- ✍️ **Single Text Analysis** - Quick analysis of individual writing samples
- 📊 **Time Series Analysis** - Track progress across multiple writing samples
- 📈 **Visual Progress Tracking** - Interactive charts showing development trends
- 💾 **Export Reports** - Download results as CSV or Word documents

## Metrics Analyzed

- **Type-Token Ratio (TTR)** - Vocabulary diversity
- **Moving Average Type-Token Ratio (MATTR)** - Stable vocabulary diversity measure
- **Dependent Clause Ratio (DCR)** - Syntactic complexity
- **Lexical Density** - Proportion of content words
- **Word Count Statistics** - Total and unique words

## How to Use

### Single Text Analysis
1. Select "Single Text Analysis" mode
2. Paste or type the student's writing
3. Add a date/label (optional)
4. Click "Analyze Text"

### Multiple Files Analysis
1. Select "Multiple Files (Time Series)" mode
2. Upload multiple .txt files (name them with dates, e.g., `2024-01-15.txt`)
3. Click "Analyze All Files"
4. View interactive charts and download reports

## Technology Stack

- **Frontend:** Streamlit
- **NLP Processing:** spaCy
- **Data Analysis:** pandas
- **Visualizations:** Plotly
- **Reports:** python-docx

## Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/writing-analyzer.git
cd writing-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```



## Feedback

Found a bug or have a feature request? Please open an issue!
```
