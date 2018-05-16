#coding=utf-8

import requests
import json
from colorCode import evaluate
import os
class Palette:
    '''
    palette format: [[44,43,44],[90,83,82],"N","N","N"]
    '''
    def __init__(self , palette):
        self.palette = palette

    def palette_score(self , palette):
        palette_str = [' '.join(str(x * 1.0 / 255) for x in item) for item in palette]
        pastr = [item + '\n' for item in palette_str]
        with open('input.txt' , 'w') as f:
            f.write('1\n')
            f.writelines(pastr)
        score = evaluate.ratingPalette(os.path.abspath('input.txt') , 'res.txt')
        return score



    def getRecColorFromColormind(self):
        url = 'http://colormind.io/api/'
        data = {}
        headers = {}
        headers['Content-Type'] = 'text/plain;charset=UTF-8'
        headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36'

        data['model'] = 'default'
        data['input'] = self.palette

        r = requests.post(url , headers = headers , data = json.dumps(data))

        if r.status_code == 200:
            res = json.loads(r.text)['result']
        else:
            res = []

        return res
    def color_topic_select(self , hex_color,show_topics = True):
        topics_str = """
    Topic 0: 0xfeb842 0xfc28e6 0xf9998a 0xf70a2e 0xecccbe 0xf1eb76 0xd0a3ca 0xf47ad2 0xe7ae06 0xea3d62
    Topic 1: 0xdffff2 0xe28f4e 0xe51eaa 0xdd7096 0xe7ae06 0xd851de 0xdae13a 0x370a3a 0xd5c282 0x3c28f2
    Topic 2: 0xdae13a 0xd851de 0xd5c282 0xd33326 0xd0a3ca 0xdd7096 0xe7ae06 0xe51eaa 0xdffff2 0xe28f4e
    Topic 3: 0xe147a 0xbeb846 0xc3d6fe 0xc147a2 0xb851e 0xbc28ea 0xc8f5b6 0x133332 0xcb8512 0xb9998e
    Topic 4: 0xce146e 0xd0a3ca 0xd5c282 0xcb8512 0xd33326 0xc8f5b6 0xd851de 0xdae13a 0xc6665a 0xc3d6fe
    Topic 5: 0xa7ae0a 0xacccc2 0xa28f52 0x9d709a 0xa51eae 0xb1eb7a 0xb47ad6 0xaf5c1e 0x9ae13e 0x9ffff6
    Topic 6: 0x628f56 0x67ae0e 0x651eb2 0x6a3d6a 0x6f5c22 0x5ffffa 0x6cccc6 0x747ada 0x5d709e 0x799992
    Topic 7: 0xb47ad6 0x86665e 0x8b8516 0x90a3ce 0xb1eb7a 0x799992 0xacccc2 0xa7ae0a 0x8e1472 0xa28f52
    Topic 8: 0x147ae 0x3d70a 0x66666 0x8f5c2 0xb851e 0xe147a 0x466662 0x43d706 0x3eb84e 0x48f5be
    Topic 9: 0xdae13a 0xdd7096 0xd5c282 0xd851de 0xdffff2 0xd33326 0xd0a3ca 0xcb8512 0xc8f5b6 0xce146e
    Topic 10: 0x1851ea 0x1ffffe 0x251eb6 0x1d70a2 0x228f5a 0x1ae146 0x2a3d6e 0x2f5c26 0x15c28e 0xe147a
    Topic 11: 0xb9998e 0xbeb846 0xc147a2 0xc3d6fe 0xb70a32 0xbc28ea 0xf70a2e 0xcb8512 0xb47ad6 0xce146e
    Topic 12: 0x466662 0x3c28f2 0x53332e 0x370a3a 0x4147aa 0x4e1476 0x31eb82 0x50a3d2 0x4b851a 0x48f5be
    Topic 13: 0xc6665a 0xc8f5b6 0xd0a3ca 0xcb8512 0xce146e 0xd5c282 0xc3d6fe 0xbeb846 0xd33326 0xd851de
    Topic 14: 0xe7ae06 0xe51eaa 0xea3d62 0xecccbe 0xe28f4e 0xdd7096 0xdffff2 0xf70a2e 0xf47ad2 0xf1eb76
    Topic 15: 0xef5c1a 0xf1eb76 0x9d709a 0x9ae13e 0xf70a2e 0xf47ad2 0xa28f52 0x9ffff6 0x95c286 0xecccbe
    Topic 16: 0xc3d6fe 0xbeb846 0xc8f5b6 0xb9998e 0xc147a2 0xb1eb7a 0xb70a32 0xbc28ea 0xb47ad6 0xcb8512
    Topic 17: 0x133332 0x2a3d6e 0x1ffffe 0x1ae146 0x251eb6 0x1851ea 0x2f5c26 0x4147aa 0x1d70a2 0x27ae12
    Topic 18: 0xf70a2e 0xecccbe 0xf1eb76 0xf47ad2 0xea3d62 0xf9998a 0xef5c1a 0xe7ae06 0xe51eaa 0xe28f4e
    Topic 19: 0xf1eb76 0xef5c1a 0xecccbe 0xf47ad2 0xf70a2e 0xea3d62 0xdffff2 0xdd7096 0xe7ae06 0xd0a3ca
    """
        def search_topic_by_color(hex_color,topic_filtered):
            def get_center_hex(hexcolor):
                span = max_hex_val/color_span
                for i in range(0,color_span):
                    if hexcolor > i * span and hexcolor <= (i+1) * span:
                        return (i*span+(i+1)*span)/2
                return span/2
            topics = []
            center_hex_color = str(hex(get_center_hex(hex_color)))
            for i_toc in range(len(topic_filtered)):
                if center_hex_color in topic_filtered[i_toc]:
                    topics.append(i_toc)
            return topics
        def rgb2hex(r,g,b):
            return (r << 16) + (g << 8) + b
        def hex2rgb(hexcolor):
            rgb = [(hexcolor >> 16) & 0xff,
                   (hexcolor >> 8) & 0xff,
                   hexcolor & 0xff
             ]
            return rgb
        
        def show_topics_by_index(topic_filtered,indexes):
            for i,topic_words in enumerate(topic_filtered):
                if i in indexes:
                    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

        max_hex_val =rgb2hex(255,255,255)
        color_span = 100
        topics = []
        for topic_str in topics_str.split("\n")[1:-1]:
            topic =  topic_str[topic_str.index(":")+1:].strip().split()
            topics.append(topic)
        selected_topic_indexes = search_topic_by_color(hex_color, topics)
        if show_topics:
            show_topics_by_index(topics, selected_topic_indexes)
        return selected_topic_indexes

    #两个参数：输入十六进制颜色，和是否显示主题名称
    #example：color_topic_select(0xf1eb76, show_topics= False)

if __name__ == '__main__':
    p = Palette([[44,43,44],[90,83,82],"N","N","N"])
    print p.color_topic_select(0xf1eb76, show_topics= True)