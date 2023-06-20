import os
import pickle
import base64
import hashlib

def get_hash_key(thing):
    return hashlib.sha256(str(thing).encode()).hexdigest()

def cache_it(cache_path, key, thing):
    b64_key = base64.b64encode(bytes(key, 'utf-8')).decode('ascii')
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    file_name = os.path.join(cache_path, b64_key)
    pickle.dump(thing, open(file_name, 'wb'))

def get_it(cache_path, key):
    b64_key = base64.b64encode(bytes(key, 'utf-8')).decode('ascii')
    file_name = os.path.join(cache_path, b64_key)
    if os.path.isfile(file_name):
        return True, pickle.load(open(file_name, 'rb'))
    else:
        return False, False

