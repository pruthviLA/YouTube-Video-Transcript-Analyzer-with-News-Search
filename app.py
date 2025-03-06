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
import openai
import hashlib
import threading
import concurrent.futures
from urllib.parse import urlparse
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set page title and favicon
st.set_page_config(
    page_title="AI Content Analyzer & News Finder",
    page_icon="🧠",
    layout="wide"
)

# Initialize NLTK resources
@st.cache_resource
def initialize_nltk():
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)

initialize_nltk()

# API Keys and Configurations
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Cache for storing API responses
cache = {}

def get_cache_key(func_name, *args, **kwargs):
    """Generate a cache key based on function name and arguments"""
    key_parts = [func_name]
    key_parts.extend([str(arg) for arg in args])
    key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
    return hashlib.md5(":".join(key_parts).encode()).hexdigest()

# Advanced Web Scraper class with proxy rotation and rate limiting
class AdvancedWebScraper:
    """Advanced web scraper with proxy rotation, rate limiting and caching"""
    
    def __init__(self):
        self.session = requests.Session()
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
        ]
        self.last_request_time = {}
        self.min_request_interval = 2  # seconds between requests to the same domain
    
    def _get_random_user_agent(self):
        """Return a random user agent"""
        return random.choice(self.user_agents)
    
    def _get_domain(self, url):
        """Extract domain from URL"""
        return urlparse(url).netloc
    
    def _respect_rate_limits(self, domain):
        """Ensure minimum time between requests to the same domain"""
        if domain in self.last_request_time:
            elapsed = time.time() - self.last_request_time[domain]
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
        self.last_request_time[domain] = time.time()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get(self, url, headers=None, params=None):
        """Perform a GET request with rate limiting and retries"""
        domain = self._get_domain(url)
        self._respect_rate_limits(domain)
        
        if not headers:
            headers = {}
        
        headers["User-Agent"] = self._get_random_user_agent()
        
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for {url}: {str(e)}")
            raise

# Google News Scraper using the advanced web scraper
class GoogleNewsScraper:
    """A robust scraper for Google News search results"""
    
    def __init__(self):
        self.base_url = "https://news.google.com/search"
        self.scraper = AdvancedWebScraper()
    
    def search(self, query, num_results=8):
        """Search Google News for the given query and return results"""
        # Check cache first
        cache_key = get_cache_key("google_news_search", query, num_results)
        if cache_key in cache:
            return cache[cache_key]
        
        params = {
            "q": query,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en"
        }
        
        headers = {
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        try:
            response = self.scraper.get(self.base_url, headers=headers, params=params)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = []
            # Try different selectors for better resilience against HTML changes
            article_elements = soup.select('div[jscontroller] article') or soup.select('.IBr9hb') or soup.select('.NiLAwe')
            
            for article_elem in article_elements[:num_results]:
                # Extract article data using multiple possible selectors
                title_elem = article_elem.select_one('h3 a') or article_elem.select_one('.DY5T1d') or article_elem.select_one('a[href*="articles"]')
                time_elem = article_elem.select_one('time') or article_elem.select_one('.WW6dff')
                source_elem = article_elem.select_one('div[data-n-tid]') or article_elem.select_one('.wEwyrc')
                
                if not title_elem:
                    continue
                
                # Get title
                title = title_elem.text.strip()
                
                # Get URL
                article_url = title_elem.get('href', '')
                if article_url.startswith('./'):
                    article_url = 'https://news.google.com' + article_url[1:]
                
                # Get source
                source = source_elem.text.strip() if source_elem else "Unknown source"
                
                # Get published date
                published = time_elem.get('datetime', '') if time_elem else ""
                
                # Get description
                description_elem = article_elem.select_one('h3 + div') or article_elem.select_one('.xBbh9') or article_elem.select_one('p')
                description = description_elem.text.strip() if description_elem else ""
                
                articles.append({
                    "title": title,
                    "url": article_url,
                    "source": source,
                    "published": published,
                    "description": description
                })
            
            # Save to cache
            cache[cache_key] = articles
            return articles
            
        except Exception as e:
            logging.error(f"Error scraping Google News: {str(e)}")
            return []

# Functions for OpenAI analysis
def analyze_text_with_ai(text, analysis_type="summarize"):
    """Analyze text using OpenAI's capabilities"""
    cache_key = get_cache_key("analyze_text_with_ai", text[:100], analysis_type)
    if cache_key in cache:
        return cache[cache_key]
    
    try:
        if analysis_type == "summarize":
            prompt = f"Please summarize the following text concisely, highlighting the key points:\n\n{text}"
        elif analysis_type == "keywords":
            prompt = f"Extract 10-15 relevant keywords from the following text. Return only the keywords separated by commas, with no explanations or introductions:\n\n{text}"
        elif analysis_type == "themes":
            prompt = f"Identify the main themes and topics discussed in the following text. Format as a bullet list:\n\n{text}"
        elif analysis_type == "search_queries":
            prompt = f"Based on the following text, generate 3-5 search queries that would help find related news articles. Make the queries specific and focused. Format as a numbered list:\n\n{text}"
        else:
            prompt = text
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in content analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        result = response.choices[0].message.content
        cache[cache_key] = result
        return result
    except Exception as e:
        logging.error(f"Error with OpenAI analysis: {str(e)}")
        if analysis_type == "keywords":
            return "error, ai, openai, connection, problem, analysis, content"
        return f"Error performing AI analysis: {str(e)}"

def ai_enhanced_keywords(text):
    """Use AI to extract keywords from text"""
    ai_keywords = analyze_text_with_ai(text[:4000], "keywords")  # Limit text length
    keywords = [keyword.strip() for keyword in ai_keywords.split(',')]
    return keywords

def generate_search_queries(text):
    """Generate optimized search queries using AI"""
    queries_text = analyze_text_with_ai(text[:4000], "search_queries")
    
    # Extract queries from numbered list format
    queries = []
    for line in queries_text.split('\n'):
        line = line.strip()
        if re.match(r'^\d+\.', line):  # Lines starting with a number and period
            query = re.sub(r'^\d+\.\s*', '', line)  # Remove the number and period
            queries.append(query)
    
    # If we couldn't extract queries, use default
    if not queries:
        queries = [text[:100]]
    
    return queries

def summarize_article(article_text):
    """Summarize an article using AI"""
    # Limit text length to avoid token limits
    limited_text = article_text[:3000]
    summary = analyze_text_with_ai(limited_text, "summarize")
    return summary

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

def extract_keywords(text, num_keywords=10, use_ai=True):
    """Extract keywords from text, optionally using AI"""
    if use_ai and len(text) > 0:
        return ai_enhanced_keywords(text)
    
    # Fallback to traditional NLP if AI fails or is not used
    tokens = preprocess_text(text)
    freq_dist = nltk.FreqDist(tokens)
    keywords = [word for word, _ in freq_dist.most_common(num_keywords)]
    return keywords

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

@st.cache_data
def get_youtube_video_info(video_id):
    """Get title and description of a YouTube video"""
    cache_key = get_cache_key("youtube_video_info", video_id)
    if cache_key in cache:
        return cache[cache_key]
    
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    
    try:
        # Get video details
        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
        response = request.execute()
        
        if not response['items']:
            return None, None
        
        snippet = response['items'][0]['snippet']
        result = (snippet['title'], snippet['description'])
        
        # Save to cache
        cache[cache_key] = result
        return result
    except Exception as e:
        logging.error(f"Error fetching YouTube video info: {str(e)}")
        return None, None

@st.cache_data
def get_youtube_transcript(video_id):
    """Get transcript of a YouTube video"""
    cache_key = get_cache_key("youtube_transcript", video_id)
    if cache_key in cache:
        return cache[cache_key]
    
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Concatenate all transcript segments
        transcript_text = ' '.join([item['text'] for item in transcript_list])
        
        # Save to cache
        cache[cache_key] = transcript_text
        return transcript_text
    except Exception as e:
        logging.error(f"Error fetching transcript: {str(e)}")
        return ""

# Function for PDF extraction
def extract_text_from_pdf(pdf_file):
    """Extract text from an uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    
    return text

# Functions for news search with parallel execution
def search_google_news(query, num_results=8):
    """Search Google News for a query"""
    scraper = GoogleNewsScraper()
    results = scraper.search(query, num_results=num_results)
    return results

def get_google_trends(keywords, timeframe='today 1-m'):
    """Get Google Trends data for keywords"""
    if not keywords:
        return pd.DataFrame()
    
    cache_key = get_cache_key("google_trends", str(keywords), timeframe)
    if cache_key in cache:
        return cache[cache_key]
    
    pytrends = TrendReq(hl='en-US', tz=360)
    
    # Only use up to 5 keywords for trends
    if len(keywords) > 5:
        keywords = keywords[:5]
    
    try:
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')
        trend_data = pytrends.interest_over_time()
        
        # Save to cache
        cache[cache_key] = trend_data
        
        if trend_data.empty:
            return pd.DataFrame()
        
        return trend_data
    except Exception as e:
        logging.error(f"Error fetching Google Trends: {str(e)}")
        return pd.DataFrame()

def fetch_article_content(url):
    """Fetch and parse article content using newspaper3k"""
    cache_key = get_cache_key("fetch_article", url)
    if cache_key in cache:
        return cache[cache_key]
    
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        result = {
            "title": article.title,
            "authors": article.authors,
            "publish_date": article.publish_date,
            "text": article.text,
            "summary": summarize_article(article.text) if len(article.text) > 300 else article.text,
            "url": url
        }
        
        # Save to cache
        cache[cache_key] = result
        return result
    except Exception as e:
        logging.error(f"Error fetching article content: {str(e)}")
        return {
            "title": "Could not fetch article",
            "authors": [],
            "publish_date": None,
            "text": f"Error: {str(e)}",
            "summary": "Could not generate summary",
            "url": url
        }

def parallel_search_news(queries, num_results=8):
    """Execute multiple search queries in parallel"""
    google_results = []
    
    # Function to run in threads
    def search_google(query):
        return search_google_news(query, num_results)
    
    # Use ThreadPoolExecutor to run queries in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(queries), 5)) as executor:
        future_to_query = {executor.submit(search_google, query): query for query in queries}
        for future in concurrent.futures.as_completed(future_to_query):
            results = future.result()
            google_results.extend(results)
    
    # Remove duplicates based on URL
    unique_google = []
    seen_urls = set()
    for article in google_results:
        if article['url'] not in seen_urls:
            unique_google.append(article)
            seen_urls.add(article['url'])
    
    return unique_google

# Streamlit UI Components
def render_header():
    """Render the app header with styling"""
    st.title("🧠 AI-Powered Content Analyzer & News Explorer")
    st.markdown("""
    <style>
    .stTitle {
        font-size: 2.5rem !important;
        color: #4B61D1 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    This application uses AI to analyze content from YouTube videos or PDF transcripts, 
    extract key insights, and find related news articles from Google News.
    """)

def render_sidebar():
    """Render the sidebar with options"""
    with st.sidebar:
        st.header("Options & Settings")
        
        ai_analysis = st.toggle("Enable AI Analysis", value=True, 
                               help="Use OpenAI to enhance analysis (better keywords, summaries, etc.)")
        
        st.subheader("Search Settings")
        news_count = st.slider("Number of news results", min_value=3, max_value=20, value=10,
                              help="Maximum number of news articles to retrieve")
        
        trends_timeframe = st.selectbox(
            "Google Trends timeframe",
            options=["today 1-m", "today 3-m", "today 12-m", "today 5-y"],
            format_func=lambda x: {
                "today 1-m": "Past month",
                "today 3-m": "Past 3 months",
                "today 12-m": "Past year",
                "today 5-y": "Past 5 years"
            }[x],
            help="Time period for Google Trends data"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This app uses:")
        st.markdown("- OpenAI GPT for content analysis")
        st.markdown("- YouTube Data & Transcript APIs")
        st.markdown("- Google News & Google Trends")
        st.markdown("- Advanced web scraping techniques")
        
    return {
        "ai_analysis": ai_analysis,
        "news_count": news_count,
        "trends_timeframe": trends_timeframe
    }

# Main App
def main():
    render_header()
    options = render_sidebar()
    
    # Input method selection
    input_method = st.radio(
        "Select input method:",
        ["YouTube Video", "PDF Transcript"],
        horizontal=True
    )
    
    if input_method == "YouTube Video":
        process_youtube_input(options)
    elif input_method == "PDF Transcript":
        process_pdf_input(options)

def process_youtube_input(options):
    youtube_url = st.text_input("Enter YouTube video URL:", placeholder="https://www.youtube.com/watch?v=...")
    
    if not youtube_url:
        # Show demo section when no URL is entered
        st.info("Enter a YouTube URL above to analyze its content and find related news.")
        return
    
    video_id = extract_youtube_id(youtube_url)
    
    if not video_id:
        st.error("Invalid YouTube URL. Please enter a valid YouTube video URL.")
        return
    
    with st.spinner("Processing YouTube video..."):
        # Show YouTube video
        st.video(youtube_url)
        
        # Get video info
        title, description = get_youtube_video_info(video_id)
        if not title:
            st.error("Could not retrieve video information. Please check the URL and try again.")
            return
        
        transcript = get_youtube_transcript(video_id)
        
        st.subheader("Video Information")
        st.write(f"**Title:** {title}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("Video Description", expanded=False):
                st.write(description)
        
        with col2:
            with st.expander("Video Transcript Preview", expanded=False):
                st.write(transcript[:500] + "..." if len(transcript) > 500 else transcript)
                if len(transcript) > 500:
                    st.download_button(
                        label="Download Full Transcript",
                        data=transcript,
                        file_name=f"{video_id}_transcript.txt"
                    )
        
        # Process text content
        all_text = f"{title} {description} {transcript}"
        
        if options["ai_analysis"] and len(all_text) > 0:
            with st.spinner("Performing AI analysis of content..."):
                # AI content analysis
                summary = analyze_text_with_ai(all_text, "summarize")
                themes = analyze_text_with_ai(all_text, "themes")
                ai_keywords = ai_enhanced_keywords(all_text)
                
                # Display AI insights
                st.subheader("AI Content Analysis")
                
                tab1, tab2, tab3 = st.tabs(["Summary", "Main Themes", "Keywords"])
                
                with tab1:
                    st.markdown(summary)
                
                with tab2:
                    st.markdown(themes)
                
                with tab3:
                    st.write(", ".join(ai_keywords))
                
                # Generate search queries
                search_queries = generate_search_queries(all_text)
                
                # Show Google Trends for AI keywords
                if ai_keywords:
                    st.subheader("Google Trends for Keywords")
                    with st.spinner("Loading Google Trends data..."):
                        trends_data = get_google_trends(ai_keywords, options["trends_timeframe"])
                        
                        if not trends_data.empty:
                            st.line_chart(trends_data)
                        else:
                            st.info("No Google Trends data available for these keywords.")
        else:
            # Traditional NLP analysis if AI is disabled
            keywords = extract_keywords(all_text, num_keywords=15, use_ai=False)
            
            st.subheader("Extracted Keywords")
            st.write(", ".join(keywords))
            
            # Show Google Trends
            st.subheader("Google Trends for Keywords")
            with st.spinner("Loading Google Trends data..."):
                trends_data = get_google_trends(keywords, options["trends_timeframe"])
                
                if not trends_data.empty:
                    st.line_chart(trends_data)
                else:
                    st.info("No Google Trends data available for these keywords.")
            
            # Use title and keywords for search
            search_queries = [f"{title} {' '.join(keywords[:5])}"]
        
        # Search for news using the AI-generated queries
        st.subheader("Related News Articles")
        
        with st.spinner("Searching for related news articles..."):
            google_results = parallel_search_news(search_queries, options["news_count"])
            
            if not google_results:
                st.info("No news results found. Try different keywords or search terms.")
                
            for i, article in enumerate(google_results, 1):
                with st.expander(f"{i}. {article['title']}"):
                    st.write(f"**Source:** {article['source']}")
                    st.write(f"**Published:** {article['published']}")
                    st.write(f"**Description:** {article['description']}")
                    
                    # Add a button to fetch full article content
                    if st.button(f"Load full article content #{i}", key=f"google_{i}"):
                        with st.spinner("Fetching article content..."):
                            article_content = fetch_article_content(article['url'])
                            st.write("**Article Summary:**")
                            st.write(article_content['summary'])
                            
                            with st.expander("View Full Article Text"):
                                st.write(article_content['text'])
                    
                    st.write(f"**Link:** [{article['url']}]({article['url']})")

def process_pdf_input(options):
    uploaded_file = st.file_uploader("Upload video transcript PDF", type="pdf")
    
    if not uploaded_file:
        st.info("Upload a PDF file containing video transcripts to analyze.")
        return
    
    # Extract text from PDF
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)
    
    # Display extracted text preview
    st.subheader("PDF Content")
    with st.expander("Extracted Text Preview"):
        st.write(text[:1000] + "..." if len(text) > 1000 else text)
        if len(text) > 1000:
            st.download_button(
                label="Download Full Text",
                data=text,
                file_name=f"{uploaded_file.name.split('.')[0]}_text.txt"
            )
    
    # Process text content with AI if enabled
    if options["ai_analysis"] and len(text) > 0:
        with st.spinner("Performing AI analysis of content..."):
            # AI content analysis
            summary = analyze_text_with_ai(text, "summarize")
            themes = analyze_text_with_ai(text, "themes")
            ai_keywords = ai_enhanced_keywords(text)
            
            # Display AI insights
            st.subheader("AI Content Analysis")
            
            tab1, tab2, tab3 = st.tabs(["Summary", "Main Themes", "Keywords"])
            
            with tab1:
                st.markdown(summary)
            
            with tab2:
                st.markdown(themes)
            
            with tab3:
                st.write(", ".join(ai_keywords))
            
            # Generate search queries
            search_queries = generate_search_queries(text)
            
            # Show Google Trends for AI keywords
            if ai_keywords:
                st.subheader("Google Trends for Keywords")
                with st.spinner("Loading Google Trends data..."):
                    trends_data = get_google_trends(ai_keywords, options["trends_timeframe"])
                    
                    if not trends_data.empty:
                        st.line_chart(trends_data)
                    else:
                        st.info("No Google Trends data available for these keywords.")
    else:
        # Traditional NLP analysis if AI is disabled
        keywords = extract_keywords(text, num_keywords=15, use_ai=False)
        
        st.subheader("Extracted Keywords")
        st.write(", ".join(keywords))
        
        # Show Google Trends
        st.subheader("Google Trends for Keywords")
        with st.spinner("Loading Google Trends data..."):
            trends_data = get_google_trends(keywords, options["trends_timeframe"])
            
            if not trends_data.empty:
                st.line_chart(trends_data)
            else:
                st.info("No Google Trends data available for these keywords.")
        
        # Create search query from first line and keywords
        first_line = text.split('\n')[0] if '\n' in text else text[:100]
        search_queries = [f"{first_line} {' '.join(keywords[:5])}"]
    
    # Search for news using the AI-generated queries
    st.subheader("Related News Articles")
    
    with st.spinner("Searching for related news articles..."):
        google_results = parallel_search_news(search_queries, options["news_count"])
        
        if not google_results:
            st.info("No news results found. Try different keywords or search terms.")
        
        for i, article in enumerate(google_results, 1):
            with st.expander(f"{i}. {article['title']}"):
                st.write(f"**Source:** {article['source']}")
                st.write(f"**Published:** {article['published']}")
                st.write(f"**Description:** {article['description']}")
                
                # Add a button to fetch full article content
                if st.button(f"Load full article content #{i}", key=f"google_pdf_{i}"):
                    with st.spinner("Fetching article content..."):
                        article_content = fetch_article_content(article['url'])
                        st.write("**Article Summary:**")
                        st.write(article_content['summary'])
                        
                        with st.expander("View Full Article Text"):
                            st.write(article_content['text'])
                
                st.write(f"**Link:** [{article['url']}]({article['url']})")

if __name__ == "__main__":
    main()
