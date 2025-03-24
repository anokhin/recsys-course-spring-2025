import json
from io import StringIO, BytesIO
from typing import Dict

from .recommender import Recommender
from ..envs import RemoteRecommenderConfig

from urllib.parse import urlunsplit, urlencode

try:
    import pycurl

    use_pycurl = True
    print("Using pycurl to access botify API")
except ImportError:
    import urllib3

    use_pycurl = False
    print("Using urllib3 to access botify API")


SCHEME = "http"


class RemoteRecommender(Recommender):
    """Call the remote recommender service"""

    def __init__(self, config: RemoteRecommenderConfig):
        self.host = config.host
        self.port = config.port

    def recommend(self, observation: Dict[str, int], reward: float, done: bool) -> int:
        data = {"track": int(observation["track"]), "time": reward}
        endpoint = "next" if not done else "last"
        url = self.get_request_url(f"{endpoint}/{observation['user']}", {})
        if use_pycurl:
            response = self.post_curl(url, data)
        else:
            response = self.post_urllib(url, data)
        return response.get("track")

    def get_request_url(self, path, query_params):
        query = urlencode(query_params)
        return urlunsplit((SCHEME, f"{self.host}:{self.port}", path, query, ""))

    def post_curl(self, url, data):
        self.curl.setopt(pycurl.URL, url)

        body = json.dumps(data)
        self.curl.setopt(pycurl.POSTFIELDSIZE, len(body))
        self.curl.setopt(pycurl.READDATA, StringIO(body))

        response = BytesIO()
        self.curl.setopt(pycurl.WRITEFUNCTION, response.write)

        self.curl.perform()

        status_code = self.curl.getinfo(pycurl.RESPONSE_CODE)
        if status_code != 200:
            raise ValueError(
                "Aww Snap :( Server returned HTTP status code {}".format(status_code)
            )

        return json.loads(response.getvalue().decode("UTF-8"))

    def post_urllib(self, url, data):
        headers = {
            "Content-Type": "application/json",
        }
        response = self.http.request(
            "POST", url, headers=headers, body=json.dumps(data)
        )
        return json.loads(response.data.decode("UTF-8"))

    def __enter__(self):
        if use_pycurl:
            self.curl = pycurl.Curl()
            self.curl.setopt(
                pycurl.HTTPHEADER,
                ["Accept: application/json", "Content-Type: application/json"],
            )
            self.curl.setopt(pycurl.POST, 1)
        else:
            self.http = urllib3.PoolManager()

        return self

    def __exit__(self, type, value, traceback):
        if use_pycurl:
            self.curl.close()
        else:
            self.http.__exit__(type, value, traceback)

    def __repr__(self):
        return f"remote({self.host}, {self.port})"
