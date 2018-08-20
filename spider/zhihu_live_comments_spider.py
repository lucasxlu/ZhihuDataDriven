import os
import random
import time

import pandas as pd
import requests


def loadJson(url):
    """
    request a ZhihuLiveComments URL and response with JSON
    :param url:
    :return:
    """
    headers = {
        "upgrade-insecure-requests": '1',
        "content-type": "application/json; charset=utf-8",
        "User-Agent": "Mozilla/5.0 (iPad; CPU OS 11_0 like Mac OS X) AppleWebKit/604.1.34 (KHTML, like Gecko) Version/11.0 Mobile/15A5341f Safari/604.1",
    }
    cookies = dict(
        cookies_are='')

    r = requests.get(url, headers=headers, cookies=cookies)
    r.encoding = 'UTF-8'
    page = r.json()

    return page


title = ["id", "created_at", "score", "content"]


def write_to_csv(data, live_id):
    """
    output to CSV
    :param data:
    :param live_id:
    :return:
    """
    if not os.path.isdir("./ZhihuLiveComments") or not os.path.exists('./ZhihuLiveComments'):
        os.makedirs('./ZhihuLiveComments')

    records = []
    for line in data:
        print('record:', line)
        records.append([line['id'], line['created_at'], line['content'], line['score']])

    col = [
        u'id',
        u'time',
        u'score',
        u'content']

    print(records)
    df = pd.DataFrame(records, columns=col)
    df.to_excel('./ZhihuLiveComments/%s.xlsx' % str(live_id), index=False, encoding='UTF-8')


def parseJson(live_id):
    """
    parse fields from JSON
    :param live_id:
    :return:
    """
    data = []
    num = 0
    is_end = False
    while not is_end:
        url = 'https://api.zhihu.com/lives/%s/reviews?limit=10&offset=' % str(live_id) + str(10 * num)
        num += 1
        page = loadJson(url)
        is_end = page['paging']['is_end']
        for i in page['data']:
            data.append({'content': i['content'], 'score': i['score'], 'created_at': i['created_at'], 'id': i['id']})

        write_to_csv(data, live_id)
    time.sleep(random.randint(3, 5))


if __name__ == '__main__':
    ids = pd.read_excel('../ZhihuLiveDB.xlsx', index_col=None)['id']
    for id in ids:
        print('processing ZhihuLive %s ...' % str(id))
        try:
            parseJson(id)
            time.sleep(random.randint(3, 8))
        except:
            pass
