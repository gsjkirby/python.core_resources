"""A very basic intro to scraping information from urls.
"""
#pip install BeautifulSoup4
import urllib2
from bs4 import BeautifulSoup
#import requests
#requests.get('https://www.imdb.com/chart/toptv/?ref_=nv_tp_tv250_2', verify=False)

# Ensure certificate is verified
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

# Scrape infor from imdb top rated TV shows
url = 'https://www.imdb.com/chart/toptv/?ref_=nv_tp_tv250_2'
test_url = urllib2.urlopen(url)
readHtml = test_url.read()
#test_url.close()

bs = BeautifulSoup(readHtml)
print(bs)

# Now you would pull the data out of the html and then export to a CSV
# Follow tutorial here to complete exercise:
# https://medium.freecodecamp.org/how-to-scrape-websites-with-python-and-beautifulsoup-5946935d93fe
