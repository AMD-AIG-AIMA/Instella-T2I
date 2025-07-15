# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.

import json
import base64, binascii
from PIL import Image
import io

def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}
    return _f

def pilimg_from_base64(imagestring):
    try:
        jpgbytestring = base64.b64decode(imagestring, validate=True)
    except binascii.Error:
        jpgbytestring = imagestring
    # jpgbytestring = base64.b64decode(imagestring)
    image = Image.open(io.BytesIO(jpgbytestring))
    image = image.convert('RGB')
    return image

class WebdatasetFilter:
    def __init__(self, min_size=1024, max_pwatermark=0.5):
        self.min_size = min_size
        self.max_pwatermark = max_pwatermark

    def get_value(self, x, name, is_list=False):
        if is_list:
            return x.get(name)[0]
        else:
            return x.get(name)
    def __call__(self, x):
        if self.min_size <= 0:
            return True
        try:
            if "meta" in x:
                x_json = json.loads(x["meta"])
                if 'WIDTH' in x_json:
                    height =  x_json.get("WIDTH", 0.0)
                    is_list = True if  isinstance(height, list) else False

                    filter_size = self.get_value(x_json, "WIDTH", is_list) >= self.min_size and self.get_value(x_json, "HEIGHT", is_list) >= self.min_size

                    filter_watermark = self.get_value(x_json, "pwatermark", is_list) <= self.max_pwatermark

                    return filter_size and filter_watermark
                else:
                    filter_size = x_json.get("width") >= self.min_size and x_json.get("height") >= self.min_size
                    return filter_size
            
            elif "json" in x:
                x_json = json.loads(x["json"])
                if 'WIDTH' in x_json:
                    height =  x_json.get("WIDTH", 0.0)
                    is_list = True if  isinstance(height, list) else False

                    filter_size = self.get_value(x_json, "WIDTH", is_list) >= self.min_size and self.get_value(x_json, "HEIGHT", is_list) >= self.min_size

                    filter_watermark = self.get_value(x_json, "pwatermark", is_list) <= self.max_pwatermark

                    return filter_size and filter_watermark
                else:
                    filter_size = x_json.get("width") >= self.min_size and x_json.get("height") >= self.min_size
                    return filter_size
                
            else:
                return False
        except Exception:
            return False
        
class AspectRatioFilter:
    def __init__(self, max_ar):
        self.max_ar = max_ar

    def get_value(self, x, name, is_list=False):
        if is_list:
            return x.get(name)[0]
        else:
            return x.get(name)
    def __call__(self, x):
        try:
            if "meta" in x:
                x_json = json.loads(x["meta"])
                if 'WIDTH' in x_json:
                    height =  x_json.get("WIDTH", 0.0)
                    is_list = True if  isinstance(height, list) else False

                    height = self.get_value(x_json, "HEIGHT", is_list) + 1e-8
                    width = self.get_value(x_json, "WIDTH", is_list) + 1e-8

                    filter_flag = max(height/width, width/height) < self.max_ar
                    return filter_flag
                else:
                    width = x_json.get("width") + 1e-8
                    height = x_json.get("height") + 1e-8
                    filter_flag = max(height/width, width/height) < self.max_ar
                    return filter_flag
            
            elif "json" in x:
                x_json = json.loads(x["json"])
                if 'WIDTH' in x_json:
                    height =  x_json.get("WIDTH", 0.0)
                    is_list = True if  isinstance(height, list) else False

                    height = self.get_value(x_json, "HEIGHT", is_list) + 1e-8
                    width = self.get_value(x_json, "WIDTH", is_list) + 1e-8

                    filter_flag = max(height/width, width/height) < self.max_ar
                    return filter_flag
                else:
                    width = x_json.get("width") + 1e-8
                    height = x_json.get("height") + 1e-8
                    filter_flag = max(height/width, width/height) < self.max_ar
                    return filter_flag
                
            else:
                return False
        except Exception:
            return False