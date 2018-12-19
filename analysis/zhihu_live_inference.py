from sklearn.externals import joblib


class ZhihuLivePredictor:
    def __init__(self, pretrained_model):
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
