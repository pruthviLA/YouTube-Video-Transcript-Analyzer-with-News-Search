import streamlit as st
import requests
import re
import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
import PyPDF2
import io
from bs4 import BeautifulSoup
from newspaper import Article
from datetime import datetime
import time
import random
from pytrends.request import TrendReq
import os
from openai import OpenAI
import concurrent.futures

# Set page title and favicon
st.set_page_config(
    page_title="AI-Enhanced Content Analysis & News Finder",
    page_icon="📰",
    layout="wide"
)

# Initialize NLTK resources
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

download_nltk_resources()

# API Keys and Configurations
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
BING_SEARCH_API_KEY = st.secrets["BING_SEARCH_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
BING_SEARCH_ENDPOINT = "https://api.bing.microsoft.com/v7.0/news/search"

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Advanced Web Scraping Class
class WebScraper:
    """Enhanced web scraper for news sources with proxy rotation and error handling"""
    
    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
        ]
        self.google_news_url = "https://news.google.com/search"
        self.timeout = 10
        self.max_retries = 3
        self.delay_between_requests = (1, 3)  # Random delay range in seconds
        
    def _get_random_user_agent(self):
        """Get a random user agent from the list"""
        return random.choice(self.user_agents)
    
    def _get_headers(self):
        """Generate headers for HTTP requests"""
        return {
            "User-Agent": self._get_random_user_agent(),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0"
        }
    
    def _handle_rate_limiting(self, response):
        """Handle rate limiting and other HTTP errors"""
        if response.status_code == 429:  # Too Many Requests
            retry_after = int(response.headers.get('Retry-After', 30))
            time.sleep(retry_after)
            return True
        return False
    
    def search_google_news(self, query, num_results=5):
        """Search Google News with enhanced error handling and rate limiting awareness"""
        params = {
            "q": query,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en"
        }
        
        for attempt in range(self.max_retries):
            try:
                # Add random delay between requests
                time.sleep(random.uniform(*self.delay_between_requests))
                
                headers = self._get_headers()
                response = requests.get(
                    self.google_news_url, 
                    headers=headers, 
                    params=params,
                    timeout=self.timeout
                )
                
                if self._handle_rate_limiting(response):
                    continue
                    
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                articles = []
                for article_elem in soup.select('div[jscontroller] article')[:num_results]:
                    # Extract article data
                    title_elem = article_elem.select_one('h3 a')
                    time_elem = article_elem.select_one('time')
                    source_elem = article_elem.select_one('div[data-n-tid]')
                    
                    if not title_elem:
                        continue
                    
                    # Get title
                    title = title_elem.text.strip()
                    
                    # Get URL - Transform Google News relative URLs
                    article_url = title_elem.get('href', '')
                    if article_url.startswith('./'):
                        article_url = 'https://news.google.com' + article_url[1:]
                    
                    # Get source
                    source = source_elem.text.strip() if source_elem else "Unknown source"
                    
                    # Get published date
                    published = time_elem.get('datetime', '') if time_elem else ""
                    
                    # Get description (not always present)
                    description_elem = article_elem.select_one('h3 + div') or article_elem.select_one('p')
                    description = description_elem.text.strip() if description_elem else ""
                    
                    articles.append({
                        "title": title,
                        "url": article_url,
                        "source": source,
                        "published": published,
                        "description": description
                    })
                    
                return articles
                
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    st.warning(f"Error accessing Google News: {str(e)}")
                    return []
                    
                # Exponential backoff
                time.sleep(2 ** attempt)
                
        return []
    
    def fetch_article_content(self, url):
        """Fetch and parse article content with enhanced reliability"""
        for attempt in range(self.max_retries):
            try:
                article = Article(url)
                article.download()
                article.parse()
                
                # Check if content was actually retrieved
                if not article.text or len(article.text) < 100:
                    raise ValueError("Article text too short, likely paywall or not properly parsed")
                    
                return {
                    "title": article.title,
                    "authors": article.authors,
                    "publish_date": article.publish_date,
                    "text": article.text[:1000] + "..." if len(article.text) > 1000 else article.text,
                    "url": url
                }
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {
                        "title": "Could not fetch article",
                        "authors": [],
                        "publish_date": None,
                        "text": f"Error: {str(e)}",
                        "url": url
                    }
                
                # Exponential backoff
                time.sleep(2 ** attempt)

# Functions for text processing
def preprocess_text(text):
    """Clean and preprocess text to extract keywords"""
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words and len(word) > 2]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens

def extract_keywords(text, num_keywords=10):
    """Extract most frequent keywords from text"""
    tokens = preprocess_text(text)
    
    # Count frequency of each token
    freq_dist = nltk.FreqDist(tokens)
    
    # Get most common tokens
    keywords = [word for word, _ in freq_dist.most_common(num_keywords)]
    
    return keywords

# Enhanced AI functions using OpenAI
def ai_analyze_content(text, max_length=4000):
    """
    Use OpenAI to analyze content and extract:
    - Key topics
    - Main arguments
    - Sentiment
    - Suggested search terms
    """
    # Truncate text if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert content analyzer. Extract the most important information from the provided text."},
                {"role": "user", "content": f"Analyze the following content and provide: 1) Main topics, 2) Key arguments or points, 3) Overall sentiment, 4) Suggested search terms for finding related content. Keep your response in JSON format. Here's the content: {text}"}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error with OpenAI analysis: {str(e)}")
        return {
            "main_topics": [],
            "key_arguments": [],
            "sentiment": "unknown",
            "suggested_search_terms": []
        }

def ai_enhance_search_query(base_query):
    """Use OpenAI to enhance a search query for better results"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a search optimization expert. Your goal is to transform user queries into more effective search terms."},
                {"role": "user", "content": f"Transform this basic search query into a more effective one that would yield better news search results. Return only the enhanced query text without explanation. Base query: {base_query}"}
            ]
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Error enhancing search query: {str(e)}")
        return base_query

def ai_summarize_news_articles(articles, query):
    """Use OpenAI to summarize and find relationships between news articles and original query"""
    if not articles:
        return {"summary": "No articles found", "related_points": []}
        
    articles_text = "\n\n".join([f"Title: {a['title']}\nSource: {a['source']}\nDescription: {a['description']}" 
                                for a in articles[:5]])
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a news analyst who specializes in finding connections between topics."},
                {"role": "user", "content": f"Here are some news articles related to the query '{query}':\n\n{articles_text}\n\nProvide: 1) A brief summary of how these articles relate to the original query, and 2) A list of key points that connect these articles. Format as JSON with 'summary' and 'related_points' fields."}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error with articles summarization: {str(e)}")
        return {"summary": "Error generating summary", "related_points": []}

# Functions for YouTube data extraction
def extract_youtube_id(url):
    """Extract YouTube video ID from URL"""
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    
    match = re.match(youtube_regex, url)
    if match:
        return match.group(6)
    return None

def get_youtube_video_info(video_id):
    """Get title and description of a YouTube video"""
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    
    # Get video details
    request = youtube.videos().list(
        part="snippet",
        id=video_id
    )
    response = request.execute()
    
    if not response['items']:
        return None, None
    
    snippet = response['items'][0]['snippet']
    
    return snippet['title'], snippet['description']

def get_youtube_transcript(video_id):
    """Get transcript of a YouTube video"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Concatenate all transcript segments
        transcript_text = ' '.join([item['text'] for item in transcript_list])
        
        return transcript_text
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return ""

# Function for PDF extraction
def extract_text_from_pdf(pdf_file):
    """Extract text from an uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    
    return text

# Functions for news search
def search_google_news(query, num_results=5):
    """Search Google News for a query using WebScraper"""
    scraper = WebScraper()
    results = scraper.search_google_news(query, num_results=num_results)
    
    return results

def search_bing_news(query, num_results=5):
    """Search Bing News for a query"""
    headers = {"Ocp-Apim-Subscription-Key": BING_SEARCH_API_KEY}
    params = {"q": query, "count": num_results, "mkt": "en-US", "freshness": "Day"}
    
    response = requests.get(BING_SEARCH_ENDPOINT, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    
    return search_results.get("value", [])

def get_google_trends(keywords, timeframe='today 1-m'):
    """Get Google Trends data for keywords"""
    pytrends = TrendReq(hl='en-US', tz=360)
    
    # Only use up to 5 keywords for trends
    if len(keywords) > 5:
        keywords = keywords[:5]
    
    try:
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')
        trend_data = pytrends.interest_over_time()
        
        if trend_data.empty:
            return pd.DataFrame()
        
        return trend_data
    except Exception as e:
        st.error(f"Error fetching Google Trends: {str(e)}")
        return pd.DataFrame()

def parallel_fetch_content(articles, scraper):
    """Fetch article content in parallel for better performance"""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_article = {executor.submit(scraper.fetch_article_content, article['url']): article 
                            for article in articles}
        for future in concurrent.futures.as_completed(future_to_article):
            article = future_to_article[future]
            try:
                data = future.result()
                results.append((article, data))
            except Exception as exc:
                results.append((article, {
                    "title": article['title'],
                    "authors": [],
                    "publish_date": None,
                    "text": f"Error: {str(exc)}",
                    "url": article['url']
                }))
    
    # Sort results back to original order
    article_dict = {a['url']: i for i, a in enumerate(articles)}
    results.sort(key=lambda x: article_dict.get(x[0]['url'], 0))
    
    return [r[1] for r in results]

# Main App UI Components
def render_header():
    st.title("AI-Enhanced Content Analysis & News Finder")
    st.write("Analyze content from YouTube videos or PDF transcripts and find related news with AI assistance.")

def render_input_selection():
    return st.radio(
        "Select input method:",
        ["YouTube Video", "PDF Transcript"]
    )

def render_ai_settings():
    with st.expander("AI Analysis Settings"):
        use_ai = st.checkbox("Use AI to enhance analysis", value=True)
        if use_ai:
            ai_depth = st.select_slider(
                "AI Analysis Depth",
                options=["Basic", "Standard", "Deep"],
                value="Standard"
            )
        else:
            ai_depth = "None"
    return use_ai, ai_depth

def render_youtube_content(video_id, use_ai=True, ai_depth="Standard"):
    st.write(f"Processing YouTube video...")
    
    # Show YouTube video
    st.video(f"https://www.youtube.com/watch?v={video_id}")
    
    # Progress indicator
    progress_bar = st.progress(0)
    
    # Get video info
    title, description = get_youtube_video_info(video_id)
    progress_bar.progress(20)
    
    transcript = get_youtube_transcript(video_id)
    progress_bar.progress(40)
    
    if title:
        st.subheader("Video Information")
        st.write(f"**Title:** {title}")
        
        with st.expander("Video Description"):
            st.write(description)
        
        with st.expander("Video Transcript"):
            st.write(transcript)
        
        # Process text content
        all_text = f"{title} {description} {transcript}"
        
        # AI Analysis section
        if use_ai:
            with st.spinner("Performing AI analysis..."):
                ai_results = ai_analyze_content(all_text)
                progress_bar.progress(60)
                
                st.subheader("AI Content Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Main Topics**")
                    for topic in ai_results.get("main_topics", []):
                        st.write(f"- {topic}")
                    
                    st.write("**Sentiment**")
                    st.write(ai_results.get("sentiment", "Unknown"))
                
                with col2:
                    st.write("**Key Arguments**")
                    for point in ai_results.get("key_arguments", []):
                        st.write(f"- {point}")
                
                # Use AI-suggested keywords as primary
                keywords = ai_results.get("suggested_search_terms", [])
                
                # Fallback to traditional extraction if AI didn't provide enough
                if len(keywords) < 5:
                    trad_keywords = extract_keywords(all_text, num_keywords=10)
                    keywords.extend([k for k in trad_keywords if k not in keywords])
                    keywords = keywords[:15]  # Limit to 15 total
        else:
            keywords = extract_keywords(all_text, num_keywords=15)
        
        progress_bar.progress(70)
        
        st.subheader("Keywords")
        st.write(", ".join(keywords))
        
        # Show Google Trends
        st.subheader("Google Trends for Keywords")
        with st.spinner("Fetching Google Trends data..."):
            trends_data = get_google_trends(keywords)
            
            if not trends_data.empty:
                st.line_chart(trends_data)
            else:
                st.write("No Google Trends data available for these keywords.")
        
        progress_bar.progress(80)
        
        # Create base query
        if use_ai:
            base_query = title + " " + " ".join(keywords[:5])
            enhanced_query = ai_enhance_search_query(base_query)
            primary_query = enhanced_query
            st.write(f"**AI-enhanced search query:** {enhanced_query}")
        else:
            primary_query = title + " " + " ".join(keywords[:5])
        
        # Fetch news
        news_results = fetch_and_display_news(primary_query, scraper=WebScraper(), use_ai=use_ai)
        
        progress_bar.progress(100)
        return all_text, keywords, news_results
    else:
        st.error("Could not retrieve video information. Please check the URL and try again.")
        progress_bar.progress(100)
        return None, None, None

def render_pdf_content(pdf_file, use_ai=True, ai_depth="Standard"):
    # Extract text from PDF
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(pdf_file)
    
    # Display extracted text
    with st.expander("Extracted Text"):
        st.write(text)
    
    # Process with AI if enabled
    if use_ai:
        with st.spinner("Performing AI analysis..."):
            ai_results = ai_analyze_content(text)
            
            st.subheader("AI Content Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Main Topics**")
                for topic in ai_results.get("main_topics", []):
                    st.write(f"- {topic}")
                
                st.write("**Sentiment**")
                st.write(ai_results.get("sentiment", "Unknown"))
            
            with col2:
                st.write("**Key Arguments**")
                for point in ai_results.get("key_arguments", []):
                    st.write(f"- {point}")
            
            # Use AI-suggested keywords as primary
            keywords = ai_results.get("suggested_search_terms", [])
            
            # Fallback to traditional extraction if AI didn't provide enough
            if len(keywords) < 5:
                trad_keywords = extract_keywords(text, num_keywords=10)
                keywords.extend([k for k in trad_keywords if k not in keywords])
                keywords = keywords[:15]  # Limit to 15 total
    else:
        keywords = extract_keywords(text, num_keywords=15)
    
    st.subheader("Keywords")
    st.write(", ".join(keywords))
    
    # Show Google Trends
    st.subheader("Google Trends for Keywords")
    with st.spinner("Fetching Google Trends data..."):
        trends_data = get_google_trends(keywords)
        
        if not trends_data.empty:
            st.line_chart(trends_data)
        else:
            st.write("No Google Trends data available for these keywords.")
    
    # Create search query
    if use_ai:
        first_line = text.split('\n')[0] if '\n' in text else text[:100]
        base_query = first_line + " " + " ".join(keywords[:5])
        enhanced_query = ai_enhance_search_query(base_query)
        primary_query = enhanced_query
        st.write(f"**AI-enhanced search query:** {enhanced_query}")
    else:
        first_line = text.split('\n')[0] if '\n' in text else text[:100]
        primary_query = first_line + " " + " ".join(keywords[:5])
    
    # Fetch news
    news_results = fetch_and_display_news(primary_query, scraper=WebScraper(), use_ai=use_ai)
    
    return text, keywords, news_results

def fetch_and_display_news(query, scraper, use_ai=True):
    """Fetch and display news results with optional AI summarization"""
    st.subheader("Related News Articles")
    
    # Create tabs for different news sources
    tabs = st.tabs(["Google News", "Bing News", "Summary"])
    
    # Initialize results containers
    google_results = []
    bing_results = []
    
    # Fetch Google News
    with tabs[0]:
        st.write("### Google News Results")
        with st.spinner("Searching Google News..."):
            google_results = search_google_news(query, num_results=8)
            
            if google_results:
                # Fetch article content in parallel for better performance
                article_contents = parallel_fetch_content(google_results, scraper)
                
                for i, (article, content) in enumerate(zip(google_results, article_contents), 1):
                    with st.expander(f"{i}. {article['title']}"):
                        st.write(f"**Source:** {article['source']}")
                        st.write(f"**Published:** {article['published']}")
                        st.write(f"**Description:** {article['description']}")
                        st.write(f"**Link:** [{article['url']}]({article['url']})")
                        
                        st.write("**Article Preview:**")
                        st.write(content['text'])
            else:
                st.write("No Google News results found.")
    
    # Fetch Bing News
    with tabs[1]:
        st.write("### Bing News Results")
        with st.spinner("Searching Bing News..."):
            bing_results = search_bing_news(query, num_results=8)
            
            if bing_results:
                # Fetch article content in parallel
                bing_article_data = [
                    {
                        "title": article['name'],
                        "url": article['url'],
                        "source": article['provider'][0]['name'],
                        "published": article.get('datePublished', 'N/A'),
                        "description": article['description']
                    }
                    for article in bing_results
                ]
                
                article_contents = parallel_fetch_content(bing_article_data, scraper)
                
                for i, (article, content) in enumerate(zip(bing_results, article_contents), 1):
                    with st.expander(f"{i}. {article['name']}"):
                        st.write(f"**Source:** {article['provider'][0]['name']}")
                        st.write(f"**Published:** {article.get('datePublished', 'N/A')}")
                        st.write(f"**Description:** {article['description']}")
                        st.write(f"**Link:** [{article['url']}]({article['url']})")
                        
                        st.write("**Article Preview:**")
                        st.write(content['text'])
            else:
                st.write("No Bing News results found.")
    
    # AI Summary of News Articles
    with tabs[2]:
        st.write("### News Summary")
        
        if use_ai and (google_results or bing_results):
            with st.spinner("Generating AI summary of news articles..."):
                # Combine results for summary
                combined_results = google_results[:5]
                if len(combined_results) < 5 and bing_results:
                    # Add unique bing results to get up to 5 total
                    bing_urls = [b['url'] for b in bing_results]
                    google_urls = [g['url'] for g in google_results]
                    unique_bing = [
                        {
                            "title": b['name'],
                            "url": b['url'],
                            "source": b['provider'][0]['name'],
                            "description": b['description']
                        }
                        for b in bing_results if b['url'] not in google_urls
                    ]
                    combined_results.extend(unique_bing[:5-len(combined_results)])
                
                summary = ai_summarize_news_articles(combined_results, query)
                
                st.write("#### Overview")
                st.write(summary.get("summary", "No summary available"))
                
                st.write("#### Key Connections")
                for point in summary.get("related_points", []):
                    st.write(f"- {point}")
        else:
            st.write("Enable AI analysis or search for news to see a summary.")
    
    # Return combined results
    return {
        "google_results": google_results,
        "bing_results": bing_results
    }

# Main App
def main():
    render_header()
    
    input_method = render_input_selection()
    use_ai, ai_depth = render_ai_settings()
    
    if input_method == "YouTube Video":
        youtube_url = st.text_input("Enter YouTube video URL:")
        
        if youtube_url:
            video_id = extract_youtube_id(youtube_url)
            
            if video_id:
                content, keywords, news = render_youtube_content(video_id, use_ai, ai_depth)
            else:
                st.error("Invalid YouTube URL. Please enter a valid YouTube video URL.")
    
    elif input_method == "PDF Transcript":
        uploaded_file = st.file_uploader("Upload video transcript PDF", type="pdf")
        
        if uploaded_file is not None:
            content, keywords, news = render_pdf_content(uploaded_file, use_ai, ai_depth)
    
    # Include footer with info
    st.markdown("---")
    st.markdown("Built with Streamlit and OpenAI. Analyzes content and finds related news from Google News, Google Trends, and Bing News.")

if __name__ == "__main__":
    main()
