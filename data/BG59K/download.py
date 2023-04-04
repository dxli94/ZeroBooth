import urllib.request
import os

# Create the folder to save images
if not os.path.exists("images"):
    os.makedirs("images")

# Read the list of URLs from a text file
with open("urls.txt") as f:
    urls = f.read().splitlines()

# Download each image and save it to the "images" folder
for i, url in enumerate(urls):
    if i % 1000 == 0:
        print(i)
    filename = os.path.join("images", url.split("=")[-2] + ".jpg")
    if os.path.exists(filename):
        continue
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {url}")
    except Exception as e:
        print(f"Error downloading {url}")
        print(e)
