import json
import urllib
import requests

url = 'https://parseapi.back4app.com/classes/Color?limit=100000'
headers = {
    'X-Parse-Application-Id': 'vei5uu7QWv5PsN3vS33pfc7MPeOPeZkrOcP24yNX', # This is the fake app's application id
    'X-Parse-Master-Key': 'aImLE6lX86EFpea2nDjq9123qJnG0hxke416U7Je' # This is the fake app's readonly master key
}
data = json.loads(requests.get(url, headers=headers).content.decode('utf-8')) # Here you have the data that you need
# print(json.dumps(data, indent=2))

with open("color_names.json", "w") as twitter_data_file:
    json.dump(data, twitter_data_file, indent=4, sort_keys=True)