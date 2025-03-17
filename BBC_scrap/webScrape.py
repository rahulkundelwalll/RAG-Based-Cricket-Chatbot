import requests
from bs4 import BeautifulSoup
import os

class WebScraper:
    """
    A class to scrape metadata and content from multiple web pages and store the data in text files.
    """
    def __init__(self, base_url, output_dir, pages, txt_file_name):
        """
        Initializes the scraper with base URL, output directory, page count, and text file name.
        
        :param base_url: The base URL of the website to scrape.
        :param output_dir: The directory where data will be stored.
        :param pages: The number of pages to scrape.
        :param txt_file_name: The file name to append extracted articles.
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.pages = pages
        self.txt_file_name = txt_file_name
        self.list_of_links = []

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory '{output_dir}' created.")
        else:
            print(f"Directory '{output_dir}' already exists.")

        # Full path for the text file
        self.txt_file_path = os.path.join(output_dir, txt_file_name)

    def get_all_pages_links(self):
        """
        Extracts all links from the base URL and stores them in self.list_of_links.
        """
        for i in range(1, self.pages + 1):
            url = f"{self.base_url}{i}"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                links = [a['href'] for a in soup.find_all('a', href=True)]
                new_links = [link for link in links if link.startswith('https://indianexpress.com/article')]

                # Append new unique links to the list
                for link in new_links:
                    if link not in self.list_of_links:
                        self.list_of_links.append(link)
            else:
                print(f"Failed to retrieve {url}. Status code:", response.status_code)

        print(f"Total links found: {len(self.list_of_links)}")

    def get_article_data_append_to_txt(self, url):
        """
        Extracts the heading and body of an article from a given URL and appends it to a text file.

        :param url: The URL of the article to scrape.
        """
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            heading = soup.find('h1', id='main-heading-article')
            body = soup.find('div', id="pcl-full-content")

            if heading and body:
                article_title = heading.text.strip()
                article_body = "\n".join([p.text.strip() for p in body.find_all('p')])

                # Append article data to the text file
                with open(self.txt_file_path, "a", encoding="utf-8") as file:
                    file.write(f"Title: {article_title}\n")
                    file.write(f"URL: {url}\n")
                    file.write(f"Content:\n{article_body}\n")
                    file.write("=" * 80 + "\n\n")  # Separator for readability
                
                print(f"Article saved: {article_title}")

            else:
                print(f"Skipping {url}, article structure not found.")
        else:
            print(f"Failed to fetch {url}. Status code:", response.status_code)

    def scrape(self):
        """
        Main method to scrape all articles and append data to the text file.
        """
        self.get_all_pages_links()
        for link in self.list_of_links:
            self.get_article_data_append_to_txt(link)
        print(f"Scraping completed! Data stored in {self.txt_file_path}")


# Example usage:
scraper = WebScraper("https://indianexpress.com/section/sports/cricket/page/", "BBC_Sport_Output", 30, "data.txt")
scraper.scrape()
