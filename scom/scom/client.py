import socket

import utils
from constants import DISCONNECT_MSG, HEADER_SIZE, PORT


def send_msg(soc, msg: bytes) -> None:
    """
    Helper function for sending message to a socket object.

    First sends the message length so that the socket object knows how big
    the message it so that it can adjust the buffer size for receiving the response.
    """
    msg_len = str(len(msg)).ljust(HEADER_SIZE).encode()
    soc.send(msg_len)
    soc.send(msg)


if __name__ == "__main__":
    host_addr = socket.gethostbyname(socket.gethostname())
    client_soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_soc.connect((host_addr, PORT))

    # Generate private and public key for the client
    priv_key, pub_key = utils.gen_rsa_keys()

    # Send the client's public key
    send_msg(client_soc, pub_key)

    # Receive the encrypted session key
    session_key_len = client_soc.recv(HEADER_SIZE).decode()
    encrypted_session_key = client_soc.recv(int(session_key_len))
    session_key = utils.rsa_decrypt(priv_key, encrypted_session_key)

    while True:
        user_input = input("> ")

        iv, msg_cipher = utils.aes_encrypt(session_key, user_input.encode())
        send_msg(client_soc, iv)
        send_msg(client_soc, msg_cipher)

        if user_input == DISCONNECT_MSG:
            break

        # Receive encrypted response from the server
        cipher_response_iv_len = client_soc.recv(HEADER_SIZE).decode()
        cipher_response_iv = client_soc.recv(int(cipher_response_iv_len))

        cipher_response_len = client_soc.recv(HEADER_SIZE).decode()
        cipher_response = client_soc.recv(int(cipher_response_len))

        # Decode the encrypted response from the server using the session_key
        response = utils.aes_decrypt(cipher_response, session_key, cipher_response_iv)
        print(response.decode())
    print("Client exiting...")
