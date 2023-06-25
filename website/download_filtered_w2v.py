"""
Author: Joshua Ashkinaze
Date: 2023-06-25

Description: Downloads filtered w2v file from Dropbox. This is used for rendering participant feedback
"""
import requests
import os

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

def main():
    url = 'https://www.dropbox.com/scl/fi/uc6yr2ey8arb62vrp17sn/filtered_w2v.wv?dl=1&rlkey=ecp2mb9pjb7nsbljsjxnwr7na'
    save_path = os.path.join('data', 'word_embeddings', 'filtered_w2v.wv')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    download_file(url, save_path)
    print(f"Model downloaded and saved to {save_path}")

if __name__ == "__main__":
    main()
