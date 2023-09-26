import configparser

parser = configparser.ConfigParser()
parser.read("scom.conf")

HEADER_SIZE = int(parser.get("scom_config", "header_size"))
PORT = int(parser.get("scom_config", "port"))
DISCONNECT_MSG = parser.get("scom_config", "disconnect_msg")
