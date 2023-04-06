import json

def process_posts(posts):
    processed_posts = []
    for post in posts:
        if 'message' in post:
            processed_posts.append(post['message'].replace('\n', ' '))
    return processed_posts

def write_to_txt_file(processed_posts, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        for post in processed_posts:
            file.write(post + "\n")

# Read the posts from the local JSON file
with open('posts.json', 'r', encoding='utf-8') as file:
    posts_data = json.load(file)

# Process the posts
processed_posts = process_posts(posts_data)

# Write the processed posts to a text file
write_to_txt_file(processed_posts, 'train_data.txt')