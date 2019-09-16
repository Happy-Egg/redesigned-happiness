# coding=<encoding name>
from aip import AipSpeech

# 百度智能云账户参数配置
APP_ID = '17134145'
API_KEY = 'yDTW0ljcQd24ZKyaHYRTDleX'
SECRET_KEY = 'O6de7NZmhxd6KZILjZj2oHoqITdRoHyg'
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)


# 读取文件
def get_file_content(file_path):
    with open(file_path, 'rb') as fp:
        return fp.read()


# 识别本地文件
def do_tts(file_path, file_format):
    ret = client.asr(get_file_content(file_path), file_format, 16000, {
        'dev_pid': 1537,
    })
    return ret['result'][0].encode('utf-8')
