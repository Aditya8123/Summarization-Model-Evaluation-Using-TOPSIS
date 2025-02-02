# Summarization-Model-Evaluation-Using-TOPSIS

**Text Summarization Model Evaluation Using TOPSIS**

This project aims to evaluate multiple text summarization models using the TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) approach. Evaluation is conducted across four key metrics: Compression, Readability, Similarity, and Inference Time. The results are ranked and presented visually to allow for easy comparison of the models' performances.

### Project Overview
In this project, we compare the summarization performance of four advanced text summarization models on a set of texts. The objective is to assess how effectively these models create concise summaries while retaining the critical information from the original text. The models evaluated are:

**BART (Bidirectional and Auto-Regressive Transformers):**
BART is a transformer-based model fine-tuned for abstractive summarization tasks. It can generate high-quality summaries for long documents, leveraging both bidirectional and autoregressive capabilities.

**T5 (Text-to-Text Transfer Transformer):**
T5 is a versatile model that frames all NLP tasks, including summarization, as text-to-text problems. It can handle various types of tasks, producing a wide range of outputs suitable for summarization and other NLP applications.

**Pegasus (Pretrained Text-to-Text Transformer):**
Pegasus is a transformer model optimized for abstractive summarization. It has been pretrained on large-scale summarization tasks, making it particularly strong in producing coherent and informative summaries.

**LED (Long-Document Encoder-Decoder):**
LED is designed to process and summarize long documents. It features a modified transformer architecture capable of handling long text sequences, making it an ideal choice for summarizing extensive reports or articles.

### Evaluation Metrics
To assess the summarization performance of each model, the following metrics are considered:

**Compression:**  
This metric measures the degree of text reduction in the summary. It is calculated by dividing the summary length by the original text length. A smaller ratio indicates a higher compression.

**Readability:**  
Readability is assessed using the Flesch Reading Ease score. The score evaluates how easy the text is to read based on sentence structure and word complexity. Higher scores indicate better readability.

**Similarity:**  
Cosine similarity is used to compare the generated summary with a reference summary. This metric measures how closely the content of the summary matches the reference. A higher cosine similarity indicates better performance.

**Inference Time:**  
Inference time tracks how long it takes for each model to generate a summary. Lower times are preferred for applications requiring real-time processing.

### Requirements
To run the evaluation script, you will need the following Python libraries:

```
pip install pandas numpy matplotlib seaborn transformers sentence-transformers textstat scikit-learn
```

### Script Overview

**Data Preparation:**  
A long text (e.g., a historical summary) is used to evaluate the models, and a reference summary is provided for calculating similarity.

**Model Setup:**  
The models (BART, T5, Pegasus, LED) are loaded using Hugging Face's `transformers` library.

**Metrics Calculation:**  
Compression is calculated as the ratio of the summary length to the original length. Readability is measured with the `textstat` library. Similarity is computed using `sentence-transformers` to compare the model's output with the reference. Inference time is measured for each model.

**TOPSIS Evaluation:**  
TOPSIS is used to rank the models based on the four metrics. The weights and impacts for each metric are defined as follows:
- **Compression:** Positive impact (shorter summaries are better).
- **Readability:** Positive impact (easier-to-read summaries are better).
- **Similarity:** Positive impact (more similar summaries are better).
- **Inference Time:** Negative impact (faster inference is better).

**Visualization:**  
Various plots are generated to visualize the results:
- **Bar Plot:** Comparing models across each metric.
- **TOPSIS Score Comparison:** Bar plot showing models ranked by their TOPSIS score.
- **Box Plot:** Showing the distribution of model ranks.
- **Heatmap:** Visualizing the model scores across all metrics.

### Usage
To run the script and evaluate the models, simply execute the following:

```
python analysis.py
```

The script will generate:
- **TOPSIS Results:** A list of models ranked by their TOPSIS scores.
- **Visualizations:** A series of visual plots, including bar charts, heatmaps, and box plots.
