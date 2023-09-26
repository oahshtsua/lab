from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Util.Padding import pad, unpad


def gen_rsa_keys(n: int = 1024) -> (bytes, bytes):
    """Generate a pair (private, public) of RSA keys of size `n`."""
    key = RSA.generate(n)
    priv_key = key.export_key()
    pub_key = key.public_key().export_key()
    return priv_key, pub_key


def rsa_encrypt(pub_key: bytes, msg: bytes) -> bytes:
    """Encrypts `msg` using the `pub_key` key using RSA encryption
    and returns a cipher text."""
    key = RSA.importKey(pub_key)
    cipher = PKCS1_OAEP.new(key)
    return cipher.encrypt(msg)


def rsa_decrypt(priv_key: bytes, cipher_txt: bytes) -> bytes:
    """Decrypts `cipher_txt` obtained from RSA encryption using the `priv_key` key
    and returns a plain text."""
    key = RSA.importKey(priv_key)
    cipher = PKCS1_OAEP.new(key)
    return cipher.decrypt(cipher_txt)


def aes_encrypt(key: bytes, msg: bytes) -> (bytes, bytes):
    """Encrypts `msg` using `key` for AES encryption and returns a cipher text."""
    cipher = AES.new(key, AES.MODE_CBC)
    iv = cipher.iv
    cipher_txt = cipher.encrypt(pad(msg, AES.block_size))
    return iv, cipher_txt


def aes_decrypt(cipher_txt: bytes, key: bytes, iv: bytes) -> bytes:
    """Decrypts an AES encrypted `cipher_txt` using the `key` and `iv`."""
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plain_txt = unpad(cipher.decrypt(cipher_txt), AES.block_size)
    return plain_txt
