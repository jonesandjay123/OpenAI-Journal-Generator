import requests
import json

def read_access_token(file_name):
    with open(file_name, 'r') as file:
        return file.read().strip()

# Read the access token from the text file
access_token = read_access_token('access_token.txt')

# Set the API endpoint to get posts
url = 'https://graph.facebook.com/v16.0/me/posts'

# Add the access token and fields to the API endpoint
params = {'access_token': access_token, 'fields': 'message,created_time'}

# Initialize variables for paging
count = 0
next_page = True
all_posts = []  # Add this line to initialize the all_posts list
while next_page:
    # Send a GET request to the API endpoint
    response = requests.get(url, params=params)
    
    # Check if the API call was successful
    if response.status_code != 200:
        print("Error: The API call was not successful.")
        print("Status code:", response.status_code)
        print("Response text:", response.text)
        break

    # Convert the API response to a JSON object
    data = json.loads(response.text)

    # (Continue with the rest of your code)
    # Print the current page number
    print('Page: {}'.format(count))
    count += 1    
    
    # Add the data to the all_posts list
    all_posts.extend(data['data'])
    
    # Check if there is a next page of data
    if 'paging' in data and 'next' in data['paging']:
        # Update the URL and params to get the next page of data
        url = data['paging']['next']
        params = {}
    else:
        # Stop requesting pages if there is no next page
        next_page = False

# Write the JSON object to a file
with open('posts.json', 'w', encoding='utf-8') as f:
    json.dump(all_posts, f, ensure_ascii=False)