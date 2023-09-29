import multiprocessing
import re
from queue import Empty

import fasttext
import warcio
from bs4 import BeautifulSoup
from tqdm import tqdm

# from langdetect import detect # it's just too slow 0.1 secs

content_queue = multiprocessing.Queue(maxsize=300)
warc_dir = "warc_files"
# Load the pre-trained FastText language identification model
fasttext.FastText.eprint = lambda x: None  # monkey patch
model = fasttext.load_model("lid.176.ftz")


def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    # Remove script and style tags
    for script in soup(
        ["script", "style", "head", "header", "footer", "a", "ul", "ol"]
    ):
        script.extract()
    # Extract and clean text content
    text_content = soup.get_text()
    # Remove excessive whitespace (including extra newlines and spaces)
    text_content = re.sub(r"[^a-zA-Z0-9\s.,]", " ", text_content)
    cleaned_content = " ".join(text_content.split())
    return cleaned_content


def process_record(record, queue):
    content_type = record.http_headers.get_header("Content-Type")
    if content_type and "text/html" in content_type:
        content = record.content_stream().read()
        cleaned_content = clean_html(content.decode("utf-8", errors="ignore"))
        if cleaned_content and cleaned_content != "":
            predictions = model.predict(
                cleaned_content[0:1000]
            )  # maybe do this with transformers
            lang_code, confidence = predictions
            if lang_code[0] == "__label__en":
                # print(cleaned_content)
                queue.put(cleaned_content, block=True)


# Function to process a single WARC file
def process_warc_file(sem, path, queue, thread):
    with sem:
        # total_iterations = 25000
        # progress_bar = tqdm(
        #     total=total_iterations,
        #     desc=str(path),
        #     position=thread,
        #     unit="record",
        #     unit_scale=True,
        #     smoothing=0.1,
        # )
        with open(path, "rb") as warc_file:
            for record in warcio.archiveiterator.ArchiveIterator(warc_file):
                if record.rec_type == "response":
                    process_record(record, queue)
                    # progress_bar.update(1)
        # Close the progress bar
        # progress_bar.close()


# List of warc.gz file paths
warc_file_paths = [
    warc_dir + "/CC-NEWS-20230904222436-01038.warc.gz",
    warc_dir + "/CC-NEWS-20230904235352-01039.warc.gz",
    warc_dir + "/CC-NEWS-20230905014842-01040.warc.gz",
]


def queue_iterator(queue):
    print("Starting queue iterator")
    max_concurrent_processes = 4
    semaphore = multiprocessing.Semaphore(max_concurrent_processes)
    processes = []
    thread = 0
    for warc_file_path in warc_file_paths:
        thread = thread + 1
        process = multiprocessing.Process(
            target=process_warc_file,
            args=(semaphore, warc_file_path, content_queue, thread),
        )
        processes.append(process)
        process.start()
    while True:
        try:
            yield queue.get(timeout=5)
        except Empty:
            print("Reached end of warc record queue. Exiting")
            break
    for process in processes:
        process.join()


# Create a custom iterator for content_queue
print("Starting warc file queue..")
iterator = queue_iterator(content_queue)
