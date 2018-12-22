import torch
import os

import requests
import numpy as np
from sklearn.externals import joblib

from analysis.models import MLP

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ZhihuLivePredictor:
    def __init__(self, pretrained_model='./model/svr.pkl'):
        self.model = joblib.load(pretrained_model)

    def infer(self, zhihu_live):
        """
        infer a Zhihu Live's score
        Note: The input param of zhihu_live should be:
        {
            "zhihu_live_id": 00000,
            "attr": [0, 1, 2, 3, 4, 5]
        }
        :param zhihu_live:
        :return:
        """
        pass


def predict_score(zhihu_live_id):
    """
    predict a Zhihu Live's score with ML model
    :Note: Normalization need to be done!!!
    :param zhihu_live_id:
    :return:
    """
    req_url = 'https://api.zhihu.com/lives/%s' % str(zhihu_live_id).strip()
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Host': 'api.zhihu.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36',
        'Upgrade-Insecure-Requests': '1'
    }
    cookies = dict(
        cookies_are='')
    response = requests.get(req_url, headers=headers, cookies=cookies)
    if response.status_code == 200:
        live = response.json()
        print(live)
        if live['review']['count'] < 18:
            print('The number of scored people is scarce, please buy this Live carefully!')
        else:
            if os.path.exists('./model/zhihu_live_mlp.pth'):
                net = MLP()
                net.load_state_dict(torch.load('./model/zhihu_live_mlp.pth'))
                input = np.array([live['duration'], live['reply_message_count'], 1 if live['source'] == 'admin' else 0,
                                  int(live['purchasable']), int(live['is_refundable']), int(live['has_authenticated']),
                                  0 if live['speaker']['member']['user_type'] == 'organization' else 1,
                                  live['speaker']['member']['gender'], len(live['speaker']['member']['badge']),
                                  live['speaker_audio_message_count'], live['attachment_count'], live['liked_num'],
                                  int(live['is_commercial']), live['audition_message_count'],
                                  int(live['is_audition_open']),
                                  live['seats']['taken'], live['seats']['max'], live['speaker_message_count'],
                                  live['fee']['amount'],
                                  live['fee']['original_price'] / 100, int(live['has_audition']),
                                  int(live['has_feedback']), live['review']['count']], dtype=np.float32)
                if torch.cuda.is_available():
                    net = net.to(DEVICE)
                    input = torch.from_numpy(input).to(DEVICE)

                score = net.forward(input)
                print('Score is %d' % score)
    else:
        print(response.status_code)
