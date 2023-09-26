import pytest
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes

from utils import (aes_decrypt, aes_encrypt, gen_rsa_keys, rsa_decrypt,
                   rsa_encrypt)


@pytest.mark.parametrize("key_size", [1024, 2048, 3072, 4096])
def test_rsa_keys_generation(key_size):
    priv, _ = gen_rsa_keys(key_size)
    priv_key = RSA.importKey(priv)

    assert priv_key.size_in_bits() == key_size


def test_rsa_keys_can_encrypt():
    priv_key, _ = gen_rsa_keys()
    key = RSA.importKey(priv_key)
    assert key.can_encrypt()


def test_rsa_encryption_decryption():
    message = "Hello world 123 !!".encode()
    priv_key, pub_key = gen_rsa_keys()
    cipher_txt = rsa_encrypt(pub_key, message)
    assert rsa_decrypt(priv_key, cipher_txt) == message


def test_aes_encryption_decryption():
    key = get_random_bytes(16)
    message = "Hello world 123 !!".encode()
    iv, cipher_txt = aes_encrypt(key, message)
    assert aes_decrypt(cipher_txt, key, iv) == message


def test_long_input_aes_encryption_decryption():
    key = get_random_bytes(16)
    with open("lorem.txt", "r") as fp:
        message = fp.read().encode()
    iv, cipher_txt = aes_encrypt(key, message)
    assert aes_decrypt(cipher_txt, key, iv) == message
